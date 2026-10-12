#!/usr/bin/env python3
"""Verify sign, ambiguity, energy invariance and access guards on synthetic data."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shen_stage1.frequency import periodic_phase_offset
from shen_stage1.protocol import development_path


def main():
    fs, lag = 1_000_000, 1024
    rng = np.random.default_rng(20261012)
    basic = rng.normal(size=lag) + 1j * rng.normal(size=lag)
    clean = np.tile(basic, 8)
    n = np.arange(len(clean))
    frequencies = np.array([-1000., -480., -100., 0., 75., 480., 600., 1000.])
    signals = clean[None, :] * np.exp(2j * np.pi * frequencies[:, None] * n[None, :] / fs)
    before = signals.copy()
    result = periodic_phase_offset(signals, lag, fs)
    period = fs / lag
    expected = (frequencies + period / 2) % period - period / 2
    np.testing.assert_allclose(result['all_pairs_offset_hz_under_assumed_fs'], expected, atol=1e-8)
    np.testing.assert_allclose(result['first_pair_offset_hz_under_assumed_fs'], expected, atol=1e-8)
    np.testing.assert_allclose(result['coherence'], 1, atol=1e-12)
    np.testing.assert_array_equal(signals, before)
    scaled = periodic_phase_offset(signals * 0.037, lag, fs)
    np.testing.assert_allclose(scaled['all_pairs_offset_hz_under_assumed_fs'], expected, atol=1e-8)
    zero = periodic_phase_offset(np.zeros((1, 8192)), lag, fs)
    assert not zero['valid'][0] and np.isnan(zero['all_pairs_offset_hz_under_assumed_fs'][0])
    denied = 0
    for rel in ['Test/n210_1_day3_wireless_test.h5', 'Test/rtl_6_day4_wireless_test.h5', 'n210_1_test.h5']:
        try:
            development_path(ROOT / 'data/external/shen_receiver_agnostic', rel)
        except PermissionError:
            denied += 1
    assert denied == 3
    print('PASS: frequency sign, first-pair Eq.13, modulo ambiguity, amplitude invariance, input immutability, zero-signal handling, final-test guards.')
    print('Synthetic 1000 Hz is estimated as 23.4375 Hz: a small estimate does not prove compensation.')


if __name__ == '__main__':
    main()
