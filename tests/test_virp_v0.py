"""Numerical controls and scientifically consequential V0 invariants."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from virp.spectral import basis, bounded_log_response, cis_details, correlation, perturb


class V0Numerics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261012)
        self.x = rng.normal(size=(3, 8192)) + 1j * rng.normal(size=(3, 8192))

    def test_zero_and_constant_gain_controls(self):
        clean, _ = cis_details(self.x)
        for g in (0., 3 * np.log(10) / 20):
            changed = perturb(self.x, np.full(8192, g))
            got, _ = cis_details(changed)
            np.testing.assert_allclose(clean, got, atol=1e-10, rtol=0)
            np.testing.assert_allclose(correlation(self.x, changed), 1., atol=1e-14)

    def test_bound_without_frequency_clipping_or_amplification(self):
        log_h, factor = bounded_log_response(basis()[:, 0] * 20)
        db = 20 / np.log(10) * log_h
        self.assertLessEqual(np.max(np.abs(db)), 3 + 1e-12)
        self.assertLessEqual(factor, 1)
        np.testing.assert_allclose(log_h, basis()[:, 0] * log_h[0], atol=1e-14)
        tiny, factor = bounded_log_response(basis()[:, 0] * .001)
        self.assertEqual(factor, 1.)
        np.testing.assert_allclose(tiny, basis()[:, 0] * .001, atol=1e-16)

    def test_fft_multiplication_contract(self):
        log_h, _ = bounded_log_response(basis() @ np.arange(6) * .1)
        got = perturb(self.x, log_h)
        expected = np.fft.fft(self.x, axis=-1) * np.exp(log_h)
        np.testing.assert_allclose(np.fft.fft(got, axis=-1), expected, atol=2e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
