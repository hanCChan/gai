"""CPU tests for physics.py on synthetic 25 Msps L-STF-like packets with known impairments."""
import numpy as np
from physics import (FS, PERIOD, TONE_BINS, to_complex, to_cache, stf_descriptors, shift_cfo,
                     reshape_tones, add_noise_to_snr, grid_distance)


def stf(n, rng, gains_db=None, cfo=0.0, snr_db=30.0, offset=0, L=256):
    """Periodic-20 signal on the 12 STF tones, random tone phases, optional tone gains/CFO/noise."""
    k = TONE_BINS
    phase = np.exp(2j * np.pi * rng.uniform(size=12))
    g = np.ones(12) if gains_db is None else 10 ** (np.asarray(gains_db) / 20)
    t = np.arange(L + offset)
    s = (g * phase * np.exp(2j * np.pi * np.outer(t, k) / PERIOD)).sum(1)[offset:]
    s = np.tile(s, (n, 1)) * np.exp(2j * np.pi * cfo * np.arange(L) / FS)
    p = np.mean(np.abs(s) ** 2)
    w = (rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape)) * np.sqrt(p / 10 ** (snr_db / 10) / 2)
    return s + w


def test_round_trip_cache_layout():
    z = stf(3, np.random.default_rng(0))
    x = to_cache(z)
    assert x.shape == (3, 1, 256, 2) and np.allclose(np.mean(np.sum(x ** 2, -1), -1), 1, atol=1e-5)
    assert np.allclose(to_complex(x), z / np.sqrt(np.mean(np.abs(z) ** 2, 1, keepdims=True)), atol=1e-5)


def test_cfo_snr_and_tone_shape_recovered():
    rng = np.random.default_rng(1)
    gains = rng.normal(scale=2.0, size=12)
    for cfo, snr, off in [(0.0, 30.0, 0), (37e3, 20.0, 7), (-52e3, 12.0, 13)]:
        d = stf_descriptors(stf(200, rng, gains, cfo, snr, off))
        assert abs(np.median(d['cfo_hz']) - cfo) < 300, (cfo, np.median(d['cfo_hz']))
        assert abs(np.median(d['snr_db']) - snr) < 1.0, (snr, np.median(d['snr_db']))
        want = np.log(10 ** (gains / 10)); want -= want.mean()
        assert np.abs(np.median(d['tone_shape'], 0) - want).max() < 0.25


def test_interventions():
    rng = np.random.default_rng(2)
    z = stf(100, rng, cfo=10e3, snr_db=35)
    d0 = stf_descriptors(z)
    assert abs(np.median(stf_descriptors(shift_cfo(z, 15e3))['cfo_hz']) - 25e3) < 300
    delta = rng.normal(scale=1.0, size=(100, 12))
    d1 = stf_descriptors(reshape_tones(z, delta))
    got = d1['tone_shape'] - d0['tone_shape']
    assert np.abs(np.median(got - (delta - delta.mean(1, keepdims=True)), 0)).max() < 0.15
    d2 = stf_descriptors(add_noise_to_snr(z, d0['snr_db'], 15.0, rng))
    assert abs(np.median(d2['snr_db']) - 15.0) < 1.0
    assert np.allclose(add_noise_to_snr(z, d0['snr_db'], 70.0, rng), z)  # never raises SNR


def test_grid_distance():
    assert grid_distance('1-1', '4-5') == 5.0


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('PASS', name)
