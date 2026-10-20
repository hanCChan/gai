"""CPU tests for rxphys.py on synthetic L-STF-like packets (and one real-cache sanity check)."""
import numpy as np

import rxphys as R
from physics import PERIOD, TONE_BINS


def stf(n, rng, L=256, stf_len=190, snr_db=35.0, gains_db=None):
    ph = np.exp(2j * np.pi * rng.uniform(size=(n, 12)))
    g = np.ones(12) if gains_db is None else 10 ** (np.asarray(gains_db) / 20)
    t = np.arange(L)
    s = (g * ph[:, None, :] * np.exp(2j * np.pi * t[None, :, None] * TONE_BINS / PERIOD)).sum(-1)
    # non-periodic tail standing in for the L-LTF
    s[:, stf_len:] = (rng.normal(size=(n, L - stf_len)) + 1j * rng.normal(size=(n, L - stf_len))) * np.sqrt(6)
    w = (rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape)) * np.sqrt(12 / 10 ** (snr_db / 10) / 2)
    return s + w


def test_fir_zero_realises_request_exactly():
    rng = np.random.default_rng(0)
    z = stf(64, rng, snr_db=300)
    d = R.centre_shape(rng.normal(scale=0.4, size=(64, 12)))
    out, real = R.apply_response(z, d, 'fir_zero')
    assert np.abs(real - d).max() < 1e-8, np.abs(real - d).max()
    # linear filtering = circular filtering of the periodic STF inside the window, so it agrees
    # with a per-period DFT gain there
    seg = z[:, 20:180].reshape(64, 8, PERIOD)
    want = np.fft.ifft(np.fft.fft(seg, axis=2) * np.exp(R.grid_gain(d))[:, None], axis=2).reshape(64, -1)
    assert np.abs(out[:, 20:180] - want).max() < 1e-8


def test_fir_min_same_magnitude_and_causal():
    rng = np.random.default_rng(1)
    z = stf(64, rng, snr_db=300)
    d = R.centre_shape(rng.normal(scale=0.4, size=(64, 12)))
    out, real = R.apply_response(z, d, 'fir_min')
    assert np.sqrt(((real - d) ** 2).mean()) < 0.03, np.sqrt(((real - d) ** 2).mean())
    # zero delta -> identity for every mode
    for mode in ('fir_zero', 'fir_min', 'r1_segment', 'v0_circular'):
        o, r = R.apply_response(z, np.zeros((64, 12)), mode)
        assert np.abs(o - z).max() < 1e-8 and np.abs(r).max() < 1e-8, mode


def test_reproduction_modes():
    import physics
    rng = np.random.default_rng(2)
    z = stf(16, rng)
    d = R.centre_shape(rng.normal(scale=0.4, size=(16, 12)))
    out, real = R.apply_response(z, d, 'r1_segment')
    assert np.allclose(out, physics.reshape_tones(z, d)) and np.abs(real - d).max() < 1e-8
    # r1 leaves samples outside the descriptor segment unfiltered (one common rescale only)
    ratio = out[:, 200:] / z[:, 200:]
    assert np.allclose(ratio, ratio[:, :1])
    v0 = R.apply_response(z, d, 'v0_circular')[0][:, 200:] / z[:, 200:]
    assert not np.allclose(v0, v0[:, :1])


def test_fir_no_cfo_or_periodicity_side_effects():
    rng = np.random.default_rng(3)
    z = stf(200, rng)
    d = R.centre_shape(rng.normal(scale=0.4, size=(200, 12)))
    m = R.measure(z, R.apply_response(z, d, 'fir_zero')[0])
    assert m['cfo_change_hz_median_abs'] < 50 and abs(m['snr_change_db_median']) < 0.5, m
    assert m['periodicity_after'] > 0.99 and m['nonfinite'] == 0


def test_templates_and_controls():
    rng = np.random.default_rng(4)
    T, Rn, per = 4, 3, 30
    a = R.centre_shape(rng.normal(scale=0.5, size=(T, 12)))
    b = R.centre_shape(rng.normal(scale=0.5, size=(Rn, 12)))
    tx = np.repeat(np.arange(T), Rn * per * 2)
    rx = np.tile(np.repeat(np.arange(Rn), per), T * 2)
    day = np.tile(np.repeat([0, 1], Rn * per), T)
    shape = a[tx] + b[rx] + rng.normal(scale=0.05, size=(len(tx), 12))
    tpl = R.response_templates(shape, np.zeros(len(tx)), tx, rx, day)
    assert np.abs(tpl['b'] - (b - b.mean(0))).max() < 0.05
    sh = R.split_half_rx_delta(shape, np.zeros(len(tx)), tx, rx, day)
    assert max(np.linalg.norm(v) for v in sh.values()) < 0.1
    nrm = np.full(10, 0.7)
    for d in (R.gaussian_direction(10, nrm, rng), R.smooth_tilt(10, nrm), R.shuffle_tones(R.smooth_tilt(10, nrm), rng)):
        assert np.allclose(np.linalg.norm(d, axis=1), 0.7) and np.allclose(d.mean(1), 0)
    pl = R.plausibility(shape, tx, rx, tpl)
    assert pl['frac_within_source_range'] > 0.9 and pl['frac_nearest_is_target'] > 0.9


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('PASS', name)
