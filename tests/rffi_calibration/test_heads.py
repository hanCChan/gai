"""Head-level tests: exact parity with the historical P2E proto rule, leakage-free API, edge cases,
and affine behaviour of the analytic heads."""
import numpy as np
import pytest
import torch

from rffi_calibration.heads import (GaussianShrink, LegacyP2EProto, NativeHead, NIWPredictive, PosteriorPredictive,
                                    ScalarShrink, SimpleCNAPSFrozen, SimpleShotCL2N, SourceMeanLDA,
                                    TargetPrototype, default_heads)
from rffi_calibration.statistics import StatsConfig, MissingClassError

from conftest import p2e_eigenrx                    # historical P2E reference implementation


def _toy(T=4, R=3, n=40, d=6, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    proto = 4 * rng.standard_normal((T, d))
    shift = 1.5 * rng.standard_normal((T, R, d))
    z, y, c = [], [], []
    for t in range(T):
        for r in range(R):
            z.append(proto[t] + shift[t, r] + noise * rng.standard_normal((n, d)))
            y += [t] * n
            c += [r] * n
    return np.concatenate(z), np.array(y), np.array(c)


def _support(z, y, m, seed=1):
    rng = np.random.default_rng(seed)
    ix = np.concatenate([rng.choice(np.flatnonzero(y == t), m, replace=False) for t in np.unique(y)])
    return torch.as_tensor(z[ix]), y[ix]


def test_legacy_proto_matches_p2e_eigenrx_bitwise():
    """H2L must reproduce p2e_fewshot 'proto' (EigenRX + m/(m+2) shrinkage) as deployed in P2E."""
    z, y, c = _toy()
    zs, ys = _support(z[:400], y[:400], 5)
    T = 4
    er = p2e_eigenrx.EigenRX(z, y, c, T)
    gbar = np.stack([er.proj(zs.numpy())[ys == t].mean(0) for t in range(T)])
    m = 5
    legacy_means = er.mu + m / (m + 2.) * (gbar - er.mu)
    legacy = er.loglik(er.proj(z), legacy_means, np.zeros(T))
    head = LegacyP2EProto().fit(torch.as_tensor(z), y, c, T)
    cal = head.adapt(zs, ys)
    ours = cal.log_scores(z).numpy()
    # same ranking as the historical log-likelihood (shared covariance -> logdet and priors are constants)
    assert (ours.argmax(1) == legacy.argmax(1)).mean() == 1.0
    np.testing.assert_allclose(ours - ours.max(1, keepdims=True),
                               legacy - legacy.max(1, keepdims=True), rtol=1e-8, atol=1e-8)


def test_scalar_shrink_legacy_mode_reduces_to_the_same_centres():
    z, y, c = _toy()
    zs, ys = _support(z[:400], y[:400], 3)
    T = 4
    a = ScalarShrink('legacy').fit(torch.as_tensor(z), y, c, T).adapt(zs, ys)
    s, m, _ = __import__('rffi_calibration.statistics', fromlist=['support_means']).support_means(zs, ys, T)
    mu = torch.as_tensor(np.stack([z[y == t].mean(0) for t in range(T)]))
    np.testing.assert_allclose(a.center.numpy(), (mu + 3 / 5 * (s - mu)).numpy(), atol=1e-10)


def test_query_labels_cannot_change_predictions():
    """Shuffling the true query labels changes nothing: no head and no metric input path sees them."""
    z, y, c = _toy()
    zs, ys = _support(z, y, 2)
    zq, yq = torch.as_tensor(z), y.copy()
    for h in default_heads():
        head = h.fit(torch.as_tensor(z), y, c, 4)
        p1 = head.adapt(zs, ys).predict_proba(zq)
        head2 = h.fit(torch.as_tensor(z), y, c, 4)
        p2 = head2.adapt(zs, ys).predict_proba(zq)
        np.testing.assert_array_equal(p1.numpy(), p2.numpy())
        perm = np.random.default_rng(0).permutation(len(yq))
        zq_perm = zq[perm]
        p3 = head.adapt(zs, ys).predict_proba(zq_perm)
        np.testing.assert_allclose(p3.numpy(), p1.numpy()[perm], atol=1e-12)


def test_heads_are_inductive_in_batch_size_and_order():
    z, y, c = _toy()
    zs, ys = _support(z, y, 5)
    zq = torch.as_tensor(z[:60])
    for h in (PosteriorPredictive(), ScalarShrink(.5), SourceMeanLDA(), TargetPrototype(), GaussianShrink(),
              SimpleShotCL2N(), SimpleCNAPSFrozen(), NIWPredictive(4.)):
        head = h.fit(torch.as_tensor(z), y, c, 4)
        cal = head.adapt(zs, ys)
        full = cal.predict_proba(zq)
        one = torch.cat([cal.predict_proba(zq[i:i + 1]) for i in range(len(zq))])
        np.testing.assert_allclose(full.numpy(), one.numpy(), atol=1e-12)
        rev = cal.predict_proba(torch.flip(zq, [0]))
        np.testing.assert_allclose(rev.numpy(), full.numpy()[::-1], atol=1e-12)


def test_zero_shot_budget_and_missing_support():
    z, y, c = _toy()
    zt, yt = z, y
    T = 4
    # heads that are defined at m = 0
    for h in (PosteriorPredictive(), PosteriorPredictive(StatsConfig(finite_center=False)),
              PosteriorPredictive(StatsConfig(pooled=True)), SourceMeanLDA(), NIWPredictive(1.)):
        head = h.fit(torch.as_tensor(z), y, c, T)
        p = head.adapt(torch.zeros(0, z.shape[1]), np.zeros(0, int)).predict_proba(torch.as_tensor(zt))
        assert np.isfinite(p.numpy()).all()
        assert abs(float(p.sum(1).mean()) - 1.) < 1e-9
    # target prototypes are undefined without support: explicit error, never a silent fallback
    for h in (TargetPrototype(), ScalarShrink(.5), LegacyP2EProto(), GaussianShrink(), SimpleCNAPSFrozen()):
        head = h.fit(torch.as_tensor(z), y, c, T)
        with pytest.raises(MissingClassError):
            head.adapt(torch.zeros(0, z.shape[1]), np.zeros(0, int))
    # partial support is also refused (H1/H2 assume every class is enrolled)
    zs, ys = _support(z, y, 1)
    keep = ys != 2
    with pytest.raises(MissingClassError):
        TargetPrototype().fit(torch.as_tensor(z), y, c, T).adapt(zs[torch.as_tensor(keep)], ys[keep])


def test_native_head_rejects_a_class_count_mismatch():
    W, b = torch.zeros(10, 5), torch.zeros(10)
    with pytest.raises(ValueError, match='not applicable'):
        NativeHead(W, b).fit(torch.zeros(2, 5), np.zeros(2, int), np.zeros(2, int), 25)


def test_h4_variants_differ_exactly_as_designed():
    z, y, c = _toy(R=5, n=60)
    base = PosteriorPredictive(StatsConfig()).fit(torch.as_tensor(z), y, c, 4)
    ak = PosteriorPredictive(StatsConfig(finite_center=False)).fit(torch.as_tensor(z), y, c, 4)
    pooled = PosteriorPredictive(StatsConfig(pooled=True)).fit(torch.as_tensor(z), y, c, 4)
    assert bool((base.stats.A >= ak.stats.A).all())
    assert base.stats.diag['K'].shape == pooled.stats.diag['K'].shape == (4, 6)
    assert pooled.stats.diag['K'].std(0).max() < 1e-12                     # pooled -> identical rows
    assert base.stats.diag['K'].std(0).max() > 0


def test_affine_equivariance_of_the_analytic_heads():
    """Inference kernel: given statistics transformed TOGETHER with the features (c -> Mc+t, S -> M S M^T),
    full-Gaussian class probabilities are invariant under any invertible affine map. Fitted pipelines are
    NOT claimed invariant: PCA truncation, isotropic shrinkage and absolute variance floors break it, so
    for diagonal heads only per-coordinate scaling + permutation is tested end to end."""
    from rffi_calibration.heads import FullGaussianCalibrated
    from rffi_calibration.statistics import safe_cholesky
    z, y, c = _toy()
    zs, ys = _support(z, y, 5)
    rng = np.random.default_rng(3)
    M = torch.as_tensor(np.eye(6) + 0.3 * rng.standard_normal((6, 6)))
    t = torch.as_tensor(0.2 * rng.standard_normal(6))
    centers = torch.as_tensor(rng.standard_normal((4, 6)))
    G = torch.as_tensor(rng.standard_normal((6, 8)))
    S = G @ G.T / 8 + 0.1 * torch.eye(6, dtype=torch.float64)
    zq = torch.as_tensor(z[:50])
    lp = torch.full((4,), -np.log(4.), dtype=torch.float64)
    p1 = FullGaussianCalibrated(centers, safe_cholesky(S)[0], lp).predict_proba(zq)
    p2 = FullGaussianCalibrated(centers @ M.T + t, safe_cholesky(M @ S @ M.T)[0], lp).predict_proba(zq @ M.T + t)
    np.testing.assert_allclose(p1.numpy(), p2.numpy(), atol=1e-10)
    # diagonal heads: per-coordinate scaling + permutation (a similarity, not a general affine map).
    # SimpleShot CL2N and the Euclidean prototype are deliberately excluded: L2 / Euclidean geometry is
    # not scale-equivariant by construction.
    sc = np.array([1., 2., .5, 3., 1.5, .8])
    perm = np.array([2, 0, 5, 1, 4, 3])
    g = lambda a: a[:, perm] * sc[perm]
    for h in (PosteriorPredictive(), TargetPrototype('pooled'), GaussianShrink(), ScalarShrink(.5)):
        p1 = h.fit(torch.as_tensor(z), y, c, 4).adapt(zs, ys).predict_proba(torch.as_tensor(z))
        h2 = {TargetPrototype: lambda: TargetPrototype('pooled'), ScalarShrink: lambda: ScalarShrink(.5)}.get(
            type(h), type(h))()
        p2 = h2.fit(torch.as_tensor(g(z)), y, c, 4).adapt(torch.as_tensor(g(zs.numpy())), ys).predict_proba(
            torch.as_tensor(g(z)))
        np.testing.assert_allclose(p2.numpy(), p1.numpy(), atol=1e-8, err_msg=h.name)


def test_gaussian_shrink_m1_has_no_sample_variance_path():
    z, y, c = _toy()
    zs, ys = _support(z, y, 1)
    head = GaussianShrink().fit(torch.as_tensor(z), y, c, 4).adapt(zs, ys)
    assert np.isfinite(head.var.numpy()).all()
    ref = GaussianShrink().fit(torch.as_tensor(z), y, c, 4)
    np.testing.assert_allclose(head.var.numpy(), (1 + 1.) * ref.M.numpy(), rtol=1e-12)   # b = 0, Q = 2 M


def test_simplecnaps_equals_a_shared_covariance_for_m1():
    z, y, c = _toy()
    zs, ys = _support(z, y, 1)
    cal = SimpleCNAPSFrozen().fit(torch.as_tensor(z), y, c, 4).adapt(zs, ys)
    Ls = cal.chol
    for t in range(1, len(Ls)):
        np.testing.assert_allclose(Ls[t].numpy(), Ls[0].numpy(), atol=1e-12)


def test_native_head_on_source_features_reproduces_the_linear_classifier():
    z, y, c = _toy()
    rng = np.random.default_rng(4)
    W = torch.as_tensor(rng.standard_normal((4, 6)))
    b = torch.as_tensor(rng.standard_normal(4))
    head = NativeHead(W, b).fit(torch.as_tensor(z), y, c, 4)
    zq = torch.as_tensor(z[:20])
    np.testing.assert_allclose(head.adapt().predict_logits(zq).numpy(), (z[:20] @ W.numpy().T + b.numpy()), atol=1e-12)
