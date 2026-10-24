"""D1 diagnostic matrix: anchors, D_PP vs historical H4-pooled, math limits, invariances, numerics,
leakage and source-statistics rules (plan §10). Synthetic data only."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rffi_calibration.heads import (LegacyP2EProto, PosteriorPredictive, SimpleCNAPSFrozen, TargetPrototype)
from rffi_calibration.sampling import RecordTable, build_manifest
from rffi_calibration.statistics import SingleReceiverError, StatsConfig, gaussian_condition_full
from rffi_post_a.diagnostic_heads import SLOT_DEFS, adapt_variant, diagnostic_variants, score_terms
from rffi_post_a.diagnostic_matrix import anchor_check, fit_replays, metric_layer_center_errors, run_task
from rffi_post_a.diagnostic_stats import (NotApplicable, date_decomposition, half_split_stability, lag_correlation,
                                          loro_stability, panel_summary, source_diagnostics)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = json.loads((ROOT / 'configs/rffi_post_a/diagnostic_matrix.json').read_text())


def toy(T=4, R=5, n=40, d=6, seed=0, dates=2):
    rng = np.random.default_rng(seed)
    proto = 3 * rng.standard_normal((T, d))
    shift = rng.standard_normal((T, R, d)) * rng.uniform(.2, 1.5, (T, 1, d))
    scale = rng.uniform(.5, 2., (T, 1, d))
    z, y, rx, date, pos = [], [], [], [], []
    for t in range(T):
        for r in range(R):
            for dd in range(dates):
                z.append(proto[t] + shift[t, r] + 0.3 * dd + scale[t, 0] * rng.standard_normal((n, d)))
                y += [t] * n
                rx += [r] * n
                date += [dd] * n
                pos += list(range(dd * 1000, dd * 1000 + n))
    return np.concatenate(z), np.array(y), np.array(rx), np.array(date), np.array(pos)


def support(z, y, m, seed=1):
    rng = np.random.default_rng(seed)
    ix = np.concatenate([rng.choice(np.flatnonzero(y == t), m, replace=False) for t in np.unique(y)])
    return torch.as_tensor(z[ix]), y[ix]


@pytest.fixture(scope='module')
def data():
    z, y, rx, date, pos = toy()
    src = rx < 4
    b = source_diagnostics(z[src], y[src], rx[src], dict(held_rx=[4], num_classes=4))
    return z, y, rx, date, pos, src, b


def _scores(v, zs, ys, zq):
    return score_terms(adapt_variant(v, zs, ys), zq)['score'].numpy()


def _norm(a):
    return a - a.max(1, keepdims=True)


# ---------------------------------------------------------------- historical equivalence
@pytest.mark.parametrize('m', [0, 1, 5])
def test_anchor_slots_reproduce_h4_h1_h2l(data, m):
    z, y, rx, _, _, src, b = data
    V = diagnostic_variants(b, MATRIX)
    zt, yt = z[rx == 4], y[rx == 4]
    zs, ys = support(zt, yt, m) if m else (torch.zeros(0, 6, dtype=torch.float64), np.zeros(0, int))
    zq = torch.as_tensor(zt)
    h4 = PosteriorPredictive().fit(torch.as_tensor(z[src]), y[src], rx[src], 4).adapt(zs, ys).log_scores(zq).numpy()
    for slot in ('D_CC', 'M_TQ'):
        np.testing.assert_allclose(_norm(_scores(V[slot], zs, ys, zq)), _norm(h4), atol=1e-8, rtol=0)
    if m:
        h1 = TargetPrototype().fit(torch.as_tensor(z[src]), y[src], rx[src], 4).adapt(zs, ys).log_scores(zq).numpy()
        np.testing.assert_allclose(_norm(_scores(V['M_SP'], zs, ys, zq)), _norm(h1), atol=1e-8, rtol=0)
        h2l = LegacyP2EProto().fit(torch.as_tensor(z[src]), y[src], rx[src], 4).adapt(zs, ys).log_scores(zq).numpy()
        np.testing.assert_allclose(_norm(_scores(V['C_PF_L'], zs, ys, zq)), _norm(h2l), atol=1e-8, rtol=0)


def test_d_pp_differs_from_historical_h4_pooled_only_in_u_sampling(data):
    z, y, rx, _, _, src, b = data
    hist = PosteriorPredictive(StatsConfig(pooled=True)).fit(torch.as_tensor(z[src]), y[src], rx[src], 4).stats
    # historical pooled U uses Sigma_pool * sum_r 1/n / R^2; D_PP uses the class Omega
    from rffi_post_a.diagnostic_heads import _stats, _Kp, _Sp
    dpp = _stats(b, _Kp(b), _Sp(b))
    np.testing.assert_allclose(dpp.sigma.numpy(), hist.sigma.numpy())
    hist_omega = hist.A - b.K_pool - b.K_pool / b.n_rx[:, None]
    np.testing.assert_allclose((dpp.A - b.omega).numpy(), (hist.A - hist_omega).numpy(), atol=1e-12)  # same K-part
    assert not torch.allclose(hist_omega, b.omega)                     # the two U definitions do differ here
    np.testing.assert_allclose(hist_omega.numpy(), (b.sigma_pool * (1. / b.cell_n).sum(1)[:, None] / b.n_rx[:, None] ** 2).numpy(),
                               rtol=1e-10)


def test_panel_reproduces_h4_and_documents_shrinkage_weights(data):
    *_, b = data
    np.testing.assert_allclose((b.K + b.U_K + b.omega).numpy(), b.A.numpy(), atol=1e-12)
    np.testing.assert_allclose(b.a_K.flatten().numpy(), np.full(4, 3 / 11))
    np.testing.assert_allclose(b.a_S.flatten().numpy(), np.full(4, (4 * 79) / (4 * 79 + 6)))  # 4 RX x (80-1) df
    p = panel_summary(b)
    assert set(p['G']) == {1, 2, 5, 10} and 0 <= p['frac_K_raw_clamped'] <= 1
    assert 'K0_eig' in p and p['pca_k'] >= 2


# ---------------------------------------------------------------- mathematics
def test_pooled_bayes_full_matches_diag_when_moments_are_diagonal(data):
    *_, b = data
    pb = b.pooled_bayes
    A0, S0 = torch.diag(torch.diagonal(pb['A0'])), torch.diag(torch.diagonal(pb['Sigma0']))
    mu, s = b.pca_mu[0], b.pca_mu[1]
    th, Q = gaussian_condition_full(mu, A0, S0, s, 3)
    G = torch.diagonal(A0) / (torch.diagonal(A0) + torch.diagonal(S0) / 3)
    np.testing.assert_allclose(th.numpy(), (mu + G * (s - mu)).numpy(), atol=1e-10)
    np.testing.assert_allclose(torch.diagonal(Q).numpy(), (torch.diagonal(S0) + torch.diagonal(A0) * (torch.diagonal(S0) / 3) /
                                                          (torch.diagonal(A0) + torch.diagonal(S0) / 3)).numpy(), atol=1e-10)


def test_k0_is_psd_and_zero_shot_bayes_slots_use_prior(data):
    z, y, rx, *_, b = data
    assert float(torch.linalg.eigvalsh(b.pooled_bayes['K0']).min()) >= -1e-10
    V = diagnostic_variants(b, MATRIX)
    empty = (torch.zeros(0, 6, dtype=torch.float64), np.zeros(0, int))
    sc = adapt_variant(V['C_PF_B'], *empty)
    np.testing.assert_allclose(sc.center.numpy(), b.pca_mu.numpy(), atol=1e-12)
    for slot in ('M_SP', 'M_SQ', 'C_RD_S', 'C_PF_L'):
        with pytest.raises(NotApplicable):
            adapt_variant(V[slot], *empty)


def test_h5b_one_shot_uses_a_shared_metric():
    """Plan §2.2 point 4: with one support point per class S_k=0 and lambda=1/2, so Q_k is shared and the
    (omitted) log det would be a common constant."""
    z, y, rx, *_ = toy()
    zs, ys = support(z, y, 1)
    cal = SimpleCNAPSFrozen().fit(torch.as_tensor(z), y, rx, 4).adapt(zs, ys)
    for t in range(1, 4):
        np.testing.assert_allclose(cal.chol[t].numpy(), cal.chol[0].numpy(), atol=1e-12)


def test_full_conditioning_affine_equivariance_given_parameters():
    """Given (centre, A, Sigma) transformed with the features, full Gaussian conditioning gives the same
    class probabilities; log scores differ by a common Jacobian constant only."""
    from rffi_post_a.diagnostic_heads import FullScorer
    from rffi_calibration.statistics import safe_cholesky
    rng = np.random.default_rng(3)
    k, T = 4, 3
    M = torch.as_tensor(np.eye(k) + .4 * rng.standard_normal((k, k)))
    c0 = torch.as_tensor(rng.standard_normal(k))
    mus = torch.as_tensor(rng.standard_normal((T, k)))
    ss = torch.as_tensor(rng.standard_normal((T, k)))
    G1, G2 = (torch.as_tensor(rng.standard_normal((k, k + 1))) for _ in range(2))
    A, S = G1 @ G1.T / k, G2 @ G2.T / k + .1 * torch.eye(k, dtype=torch.float64)
    zq = torch.as_tensor(rng.standard_normal((9, k)))

    def scorer(f, Ml):
        cs, ls = [], []
        for t in range(T):
            th, Q = gaussian_condition_full(f(mus[t]), Ml @ A @ Ml.T, Ml @ S @ Ml.T, f(ss[t]), 2)
            cs.append(th)
            ls.append(safe_cholesky(Q)[0])
        return FullScorer(torch.stack(cs), torch.stack(ls), [0.])
    s1 = score_terms(scorer(lambda v: v, torch.eye(k, dtype=torch.float64)), zq)['score']
    s2 = score_terms(scorer(lambda v: M @ v + c0, M), zq @ M.T + c0)['score']
    np.testing.assert_allclose(torch.softmax(s1, -1).numpy(), torch.softmax(s2, -1).numpy(), atol=1e-10)
    diff = (s1 - s2).numpy()
    np.testing.assert_allclose(diff, np.full_like(diff, diff[0, 0]), atol=1e-9)


def test_diag_slots_invariant_to_common_shift_and_permutation_and_sign(data):
    z, y, rx, _, _, src, _ = data
    perm = np.array([3, 0, 5, 1, 4, 2])
    sign = np.array([1, -1, 1, -1, -1, 1])
    f = lambda a: (a + 2.5)[:, perm] * sign
    b1 = source_diagnostics(z[src], y[src], rx[src], dict(held_rx=[4], num_classes=4))
    b2 = source_diagnostics(f(z[src]), y[src], rx[src], dict(held_rx=[4], num_classes=4))
    zt, yt = z[rx == 4], y[rx == 4]
    zs, ys = support(zt, yt, 2)
    for slot in ('D_CC', 'D_PC', 'D_CP', 'D_PP', 'M_SP', 'M_TP', 'M_SQ', 'M_TSIG', 'M_NOLOGDET', 'C_RD_S', 'C_RD_L'):
        p1 = torch.softmax(score_terms(adapt_variant(diagnostic_variants(b1, MATRIX)[slot], zs, ys), torch.as_tensor(zt))['score'], -1)
        p2 = torch.softmax(score_terms(adapt_variant(diagnostic_variants(b2, MATRIX)[slot], torch.as_tensor(f(zs.numpy())), ys),
                                       torch.as_tensor(f(zt)))['score'], -1)
        np.testing.assert_allclose(p1.numpy(), p2.numpy(), atol=1e-8, err_msg=slot)


# ---------------------------------------------------------------- numerics
def test_non_finite_input_is_rejected_and_types_checked(data):
    *_, b = data
    V = diagnostic_variants(b, MATRIX)
    zq = torch.zeros(3, 6, dtype=torch.float64)
    zq[1, 2] = float('nan')
    with pytest.raises(FloatingPointError):
        score_terms(adapt_variant(V['D_CC'], torch.zeros(0, 6, dtype=torch.float64), np.zeros(0, int)), zq)
    with pytest.raises(TypeError):
        score_terms(adapt_variant(V['D_CC'], torch.zeros(0, 6, dtype=torch.float64), np.zeros(0, int)), np.zeros((2, 6)))


def test_near_singular_k_is_allowed_and_jitter_recorded():
    z, y, rx, *_ = toy(R=3, d=6)
    z = z.copy()
    z[:, 5] = z[:, 4] * 1.0 + 1e-9 * np.random.default_rng(0).standard_normal(len(z))   # collinear feature
    b = source_diagnostics(z, y, rx, dict(num_classes=4))
    V = diagnostic_variants(b, MATRIX)
    zs, ys = support(z, y, 2)
    sc = adapt_variant(V['C_PF_B'], zs, ys)
    assert all(j >= 0 for j in sc.jitter) and torch.isfinite(sc.chol).all()
    sc2 = adapt_variant(V['C_RF_S'], zs, ys)
    assert torch.isfinite(score_terms(sc2, torch.as_tensor(z[:5]))['score']).all()


def test_registry_rejects_unknown_or_duplicate_slots(data):
    *_, b = data
    with pytest.raises(KeyError):
        diagnostic_variants(b, dict(slots=[dict(slot='D_BEST')]))
    with pytest.raises(ValueError):
        diagnostic_variants(b, dict(slots=[dict(slot='D_CC'), dict(slot='D_CC')]))
    assert MATRIX['registered_slot_count'] == len(MATRIX['slots']) == 20
    assert set(s['slot'] for s in MATRIX['slots']) == set(SLOT_DEFS)


# ---------------------------------------------------------------- leakage / inductive
def _task(data, n_draws=3):
    z, y, rx, date, pos, src, b = data
    t = rx == 4
    tab = RecordTable('toy', y[t], rx[t], date[t], pos[t], 'cache_position',
                      capture=np.array([f'rx{r}_d{d}' for r, d in zip(rx[t], date[t])]))
    man = build_manifest(tab, 4, 4, {'type': 'by_date', 'support_dates': [0], 'query_dates': [1]},
                         q_per_class=20, n_draws=n_draws)
    return z[t], y[t], tab.ids, man


def test_query_label_permutation_changes_no_score(data):
    z, y, rx, _, _, src, b = data
    zt, yt, ids, man = _task(data)
    reps = fit_replays(MATRIX['replays'], z[src], y[src], rx[src], 4)
    m2 = dict(MATRIX, budgets=[0, 1, 5])
    _, _, s1 = run_task(b, m2, reps, zt, yt, ids, man, 4, keep_scores=True)
    lut = {s: i for i, s in enumerate(ids)}
    q = np.array([lut[i] for i in man['query_ids']])
    y2 = yt.copy()
    y2[q] = np.random.default_rng(0).permutation(y2[q])
    _, _, s2 = run_task(b, m2, reps, zt, y2, ids, man, 4, keep_scores=True)
    assert s1.keys() == s2.keys() and len(s1) > 0
    for k in s1:
        np.testing.assert_array_equal(s1[k], s2[k])


def test_scorers_are_inductive_and_never_see_ids(data):
    *_, b = data
    z, y, rx, *_ = data
    V = diagnostic_variants(b, MATRIX)
    zs, ys = support(z[rx == 4], y[rx == 4], 5)
    zq = torch.as_tensor(z[:40])
    for name, v in V.items():
        sc = adapt_variant(v, zs, ys)
        full = score_terms(sc, zq)['score']
        one = torch.cat([score_terms(sc, zq[i:i + 1])['score'] for i in range(len(zq))])
        extra = score_terms(sc, torch.cat([zq, torch.as_tensor(z[100:130])]))['score'][:len(zq)]
        np.testing.assert_allclose(full.numpy(), one.numpy(), atol=1e-10, err_msg=name)
        np.testing.assert_allclose(full.numpy(), extra.numpy(), atol=1e-10, err_msg=name)
        with pytest.raises(TypeError):
            score_terms(sc, list(map(str, range(3))))


def test_anchor_check_on_a_full_task(data):
    z, y, rx, _, _, src, b = data
    zt, yt, ids, man = _task(data, n_draws=2)
    reps = fit_replays(MATRIX['replays'], z[src], y[src], rx[src], 4)
    rows, na, scores = run_task(b, MATRIX, reps, zt, yt, ids, man, 4, keep_scores=True)
    chk = anchor_check(scores, MATRIX['anchor_pairs'])
    assert all(v['ok'] for v in chk.values()), chk
    assert {r['slot'] for r in na} >= {'M_SP', 'M_SQ', 'C_RD_S', 'C_PF_L'}           # m=0 N/A, never zero
    assert not any(r['m'] == 0 and r['slot'] in ('M_SP', 'C_RD_S') for r in rows)
    errs = metric_layer_center_errors(b, zt, yt, ids, man, budgets=(1,))
    assert {e['center'] for e in errs} == {'mu', 's', 'theta'}


# ---------------------------------------------------------------- source statistics
def test_held_rx_and_single_rx_are_refused():
    z, y, rx, *_ = toy()
    with pytest.raises(ValueError, match='held RX'):
        source_diagnostics(z, y, rx, dict(held_rx=[4], num_classes=4))
    with pytest.raises(SingleReceiverError):
        source_diagnostics(z[rx == 0], y[rx == 0], rx[rx == 0], dict(num_classes=4))


def test_dates_are_not_extra_receivers():
    z, y, rx, date, pos = toy(R=2)
    b = source_diagnostics(z, y, rx, dict(num_classes=4))
    assert b.n_rx.tolist() == [2., 2., 2., 2.]
    dd = date_decomposition(z, y, rx, date)
    assert dd['cells'] == 8 and 0 < dd['cross_date_share_median'] < 1


def test_stability_block_runs_and_is_bounded():
    z, y, rx, date, pos = toy()
    rows = loro_stability(z, y, rx, 4)
    assert len(rows) == 5 and all(-1 <= r['spearman_K'] <= 1 for r in rows)
    hs = half_split_stability(z, y, rx, pos, 4)
    assert hs['rel_sigma'] >= 0
    lc = lag_correlation(z, y, rx, pos)
    assert set(lc) == {'lag1', 'lag2', 'lag4'} and abs(lc['lag1']['median']) < .5
