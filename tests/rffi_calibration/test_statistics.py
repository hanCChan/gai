"""Synthetic tests of the calibration statistics. No real signal, no GPU.

Covers EXPERIMENT_PROTOCOL.md §7 acceptance rows: mean/noise correction, posterior limits,
logdet/numerics, single-RX rejection, transfer-prior assumptions, dtype.
"""
import math

import numpy as np
import pytest
import torch

from rffi_calibration.statistics import (MissingClassError, SingleReceiverError, StatsConfig, cell_moments,
                                         diag_gaussian_logpdf, fit_source_stats, full_gaussian_logpdf,
                                         gaussian_condition_full, psd_project, safe_cholesky, support_means,
                                         transfer_prior, posterior_predictive)


def _cells(means, n, seed=0, sig=1.0):
    """Features for class means (T, R, d) with n packets per cell and diagonal noise."""
    rng = np.random.default_rng(seed)
    T, R, d = means.shape
    z, y, c = [], [], []
    for t in range(T):
        for r in range(R):
            z.append(means[t, r] + sig * rng.standard_normal((n, d)))
            y += [t] * n
            c += [r] * n
    return np.concatenate(z), np.array(y), np.array(c)


def test_cell_moments_match_numpy():
    z = np.random.default_rng(1).standard_normal((40, 3))
    y = np.repeat([0, 1], 20)
    c = np.tile([0, 1], 20)
    n, u, v = cell_moments(z, y, c, 2, 2)
    for t in range(2):
        for r in range(2):
            m = (y == t) & (c == r)
            assert int(n[t, r]) == m.sum()
            np.testing.assert_allclose(u[t, r].numpy(), z[m].mean(0))
            np.testing.assert_allclose(v[t, r].numpy(), z[m].var(0, ddof=1))


def test_between_covariance_subtracts_cell_mean_noise():
    """K_raw = B - mean_r C recovers the true between-receiver variance of the cell means, and B alone
    over-counts it by the sampling variance of a cell mean."""
    rng = np.random.default_rng(2)
    d, R, n, k_true, sig = 3, 8, 2000, 0.5, 2.0
    shifts = k_true * rng.standard_normal((R, d))           # fixed receiver shifts (the true d[t,r])
    means = np.stack([shifts + 0.1 * t for t in range(3)])
    z, y, c = _cells(means, n, seed=3, sig=sig)
    st = fit_source_stats(z, y, c, 3, StatsConfig(kappa_K=0.0, pooled=True))
    k_hat = st.diag['K_raw'].numpy()
    true_var = shifts.var(0, ddof=1).mean()
    assert abs(k_hat.mean() - true_var) < 0.05, (k_hat.mean(), true_var)
    assert abs(st.diag['B'].numpy().mean() - (true_var + sig ** 2 / n)) < 0.05


def test_single_receiver_is_rejected_not_silently_zeroed():
    z, y, c = _cells(np.zeros((2, 1, 3)), 5)
    with pytest.raises(SingleReceiverError, match='2 usable source receivers'):
        fit_source_stats(z, y, c, 2)


def test_missing_class_and_missing_cell_fail_loudly():
    z, y, c = _cells(np.zeros((2, 3, 3)), 5)
    with pytest.raises(MissingClassError):
        fit_source_stats(z, y, c, 3)                      # class 2 has no records
    keep = ~((y == 0) & (c == 0))                          # class 0 on one receiver only -> still 2 RX
    fit_source_stats(z[keep], y[keep], c[keep], 2)
    keep = ~((y == 0) & (c <= 1))                          # class 0 on one receiver
    with pytest.raises(SingleReceiverError):
        fit_source_stats(z[keep], y[keep], c[keep], 2)


def test_pooled_shrinkage_and_floor_are_finite_and_positive():
    z, y, c = _cells(np.zeros((3, 4, 5)), 6)
    st = fit_source_stats(z, y, c, 3)
    assert st.sigma.min() > 0 and st.A.min() >= 0
    assert torch.isfinite(st.sigma).all() and torch.isfinite(st.A).all()
    assert st.mu.dtype == torch.float64 and st.A.dtype == torch.float64


def test_finite_center_correction_increases_prior_variance():
    z, y, c = _cells(np.random.default_rng(4).standard_normal((2, 3, 4)), 8)
    a = fit_source_stats(z, y, c, 2, StatsConfig(finite_center=True))
    b = fit_source_stats(z, y, c, 2, StatsConfig(finite_center=False))
    assert bool((a.A >= b.A - 1e-12).all())
    np.testing.assert_allclose(a.diag['U'].numpy(), a.A.numpy() - b.A.numpy(), atol=1e-12)


def test_posterior_predictive_moments_match_monte_carlo():
    """H4 posterior mean/predictive vs. an independent Monte Carlo of the same model and a textbook
    closed form. Three independent checks: (a) exact algebra, (b) posterior of the centre given sampled
    supports, (c) the predictive law z ~ N(theta_post, Sigma + V_post)."""
    rng = np.random.default_rng(5)
    d, T = 3, 1
    mu = torch.zeros(T, d, dtype=torch.float64)
    A = torch.tensor([[0.8, 0.2, 0.4]], dtype=torch.float64)
    S = torch.tensor([[1.0, 0.5, 0.25]], dtype=torch.float64)
    from rffi_calibration.statistics import SourceStats
    st = SourceStats(mu=mu, sigma=S, A=A, sigma_pool=S[0], floor=1e-8, mode='multi_rx')
    s = torch.tensor([[1.0, -0.5, 0.7]], dtype=torch.float64)
    m = torch.tensor([4.0], dtype=torch.float64)
    theta, Q = posterior_predictive(st, s, m)
    G, V = A / (A + S / m), A - (A / (A + S / m)) * A
    # (a) exact algebra
    np.testing.assert_allclose(theta.numpy()[0], (mu + G * (s - mu)).numpy()[0], atol=1e-12)
    np.testing.assert_allclose(Q.numpy()[0], (S + V).numpy()[0], atol=1e-12)
    draws = 200000
    # (b) conditional centre given the sampled support mean
    s_draw = s + (S / m).sqrt() * torch.as_tensor(rng.standard_normal((draws, d)))
    centre_draw = mu + G * (s_draw - mu) + V.sqrt() * torch.as_tensor(rng.standard_normal((draws, d)))
    se = float(V.sqrt().max()) / math.sqrt(draws) * 5
    np.testing.assert_allclose(centre_draw.mean(0).numpy(), theta.numpy()[0], atol=se)
    # Var(centre | s) = V + G^2 Var(s) = V + G^2 Sigma/m  (law of total variance)
    np.testing.assert_allclose(centre_draw.var(0).numpy(), (V + G ** 2 * S / m).numpy()[0], rtol=0.08)
    # (c) the predictive law itself
    z = theta[0] + Q[0].sqrt() * torch.as_tensor(rng.standard_normal((draws, d)))
    np.testing.assert_allclose(z.mean(0).numpy(), theta.numpy()[0], atol=float(Q.sqrt().max()) / math.sqrt(draws) * 5)
    np.testing.assert_allclose(z.var(0).numpy(), Q.numpy()[0], rtol=0.03)


def test_posterior_limits_zero_shot_and_large_m():
    from rffi_calibration.statistics import SourceStats
    mu, A, S = (torch.ones(1, 2, dtype=torch.float64), torch.full((1, 2), .5, dtype=torch.float64),
                torch.full((1, 2), 2., dtype=torch.float64))
    st = SourceStats(mu=mu, sigma=S, A=A, sigma_pool=S[0], floor=1e-8, mode='multi_rx')
    s = torch.full((1, 2), 9., dtype=torch.float64)
    th0, Q0 = posterior_predictive(st, s, torch.zeros(1, dtype=torch.float64))
    np.testing.assert_allclose(th0.numpy(), mu.numpy())            # m = 0 -> prior centre
    np.testing.assert_allclose(Q0.numpy(), (S + A).numpy())
    th_inf, Q_inf = posterior_predictive(st, s, torch.tensor([1e9], dtype=torch.float64))
    np.testing.assert_allclose(th_inf.numpy(), s.numpy(), rtol=1e-6)   # m -> inf -> the support mean
    np.testing.assert_allclose(Q_inf.numpy(), S.numpy(), rtol=1e-6)


def test_diagonal_posterior_matches_full_gaussian_conditioning():
    """Independent closed form: full-covariance conditioning of a DIAGONAL Gaussian must reproduce the
    diagonal head elementwise (both scalar and vector modes)."""
    rng = np.random.default_rng(6)
    d = 5
    mu = torch.as_tensor(rng.standard_normal(d))
    A = torch.as_tensor(0.1 + rng.random(d))
    S = torch.as_tensor(0.3 + rng.random(d))
    s = torch.as_tensor(rng.standard_normal(d))
    for m in (1, 3, 10):
        th, Q = gaussian_condition_full(mu, torch.diag(A), torch.diag(S), s, m)
        Rn = S / m
        G = A / (A + Rn)
        np.testing.assert_allclose(th.numpy(), (mu + G * (s - mu)).numpy(), atol=1e-10)
        np.testing.assert_allclose(Q.numpy(), torch.diag(S + A * Rn / (A + Rn)).numpy(), atol=1e-10)


def test_gaussian_scores_are_normalised_and_finite():
    rng = np.random.default_rng(7)
    z = torch.as_tensor(rng.standard_normal((17, 4)))
    center = torch.as_tensor(rng.standard_normal((3, 4)))
    var = torch.full((3, 4), 0.7, dtype=torch.float64)
    ll = diag_gaussian_logpdf(z, center, var)
    assert ll.shape == (17, 3) and torch.isfinite(ll).all()
    # matches scipy's multivariate normal density on the diagonal case
    from scipy.stats import multivariate_normal
    for t in range(3):
        ref = multivariate_normal(center[t].numpy(), np.diag(var[t].numpy())).logpdf(z.numpy())
        np.testing.assert_allclose(ll[:, t].numpy(), ref, atol=1e-9)
    p = torch.softmax(ll, -1)
    np.testing.assert_allclose(p.sum(-1).numpy(), 1.0, atol=1e-12)


def test_full_gaussian_logpdf_and_cholesky_jitter():
    rng = np.random.default_rng(8)
    d = 4
    G = torch.as_tensor(rng.standard_normal((d, d + 2)))
    S = G @ G.T + torch.eye(d, dtype=torch.float64)
    L, jit = safe_cholesky(S)
    assert jit == 0
    z = torch.as_tensor(rng.standard_normal((6, d)))
    center = torch.zeros(1, d, dtype=torch.float64)
    from scipy.stats import multivariate_normal
    ref = multivariate_normal(np.zeros(d), S.numpy()).logpdf(z.numpy())
    np.testing.assert_allclose(full_gaussian_logpdf(z, center, L)[:, 0].numpy(), ref, atol=1e-9)
    # genuinely PSD-singular (rank 2 in 4-D): jitter escalation must recover a finite factor
    H = torch.as_tensor(rng.standard_normal((d, 2)))
    _, j_raw = safe_cholesky(H @ H.T)
    assert j_raw > 0
    Lr, jr = safe_cholesky(psd_project(H @ H.T))
    assert jr == 0 and torch.isfinite(Lr).all()


def test_transfer_prior_factor_two_and_registration_noise():
    pool = fit_source_stats(*_cells(np.random.default_rng(9).standard_normal((2, 4, 3)), 50), 2)
    zr = np.random.default_rng(10).standard_normal((20, 3))
    yr = np.array([0] * 10 + [1] * 10)
    tr = transfer_prior(pool, zr, yr, 2)
    np.testing.assert_allclose(tr.A.numpy(), (2 * pool.diag['K_pool'] + pool.sigma_pool / 10).numpy()[None].repeat(2, 0))
    assert tr.mode == 'transfer_pooled_prior'
    u = np.stack([zr[:10].mean(0), zr[10:].mean(0)])
    np.testing.assert_allclose(tr.mu.numpy(), u, atol=1e-12)


def test_support_means_and_counts():
    z = torch.as_tensor([[0., 1.], [2., 1.], [5., 5.], [7., 5.]])
    s, m, v = support_means(z, torch.tensor([0, 0, 1, 1]), 2)
    np.testing.assert_allclose(s.numpy(), [[1., 1.], [6., 5.]])
    np.testing.assert_allclose(v.numpy(), [[2., 0.], [2., 0.]])
    s, m, v = support_means(z[:3], torch.tensor([0, 0, 1]), 2)
    assert m.tolist() == [2., 1.] and v[1].abs().sum() == 0          # m=1 -> no sample variance
