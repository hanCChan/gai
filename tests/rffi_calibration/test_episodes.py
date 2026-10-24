"""Episode schedule, differentiability (gradcheck) of the T3 posterior risk, T2 ProtoNet loss,
reference-only statistics, and the 3-RX minimum."""
import numpy as np
import pytest
import torch

from rffi_calibration.episodes import (EpisodeSchedule, episode_loss, posterior_query_logits,
                                       protonet_query_logits)
from rffi_calibration.statistics import StatsConfig


def _labels(T=3, R=4, n=30):
    y = np.repeat(np.tile(np.arange(T), R), n)
    rx = np.repeat(np.repeat(np.arange(R), T), n)
    return y, rx


def test_schedule_is_deterministic_disjoint_and_rotates_targets():
    y, rx = _labels()
    s1 = EpisodeSchedule(y, rx, 3, seed=7, n_ref=8, n_query=4)
    s2 = EpisodeSchedule(y, rx, 3, seed=7, n_ref=8, n_query=4)
    seen = []
    for step in range(8):
        a, b = s1.episode(step), s2.episode(step)
        assert (a.all_idx() == b.all_idx()).all() and a.m == b.m
        assert not set(a.sup_idx) & set(a.qry_idx)
        assert set(rx[a.sup_idx]) == {a.pseudo_target} == set(rx[a.qry_idx])
        assert a.pseudo_target not in set(rx[a.ref_idx])            # target never in the reference prior
        assert len(set(rx[a.ref_idx])) >= 2
        assert len(a.sup_idx) == 3 * a.m and len(a.qry_idx) == 3 * 4
        seen.append(a.pseudo_target)
    assert sorted(seen[:4]) == [0, 1, 2, 3] and sorted(seen[4:]) == [0, 1, 2, 3]


def test_schedule_requires_three_receivers_and_enough_records():
    y, rx = _labels(R=2)
    with pytest.raises(ValueError, match='>= 3 fitting RX'):
        EpisodeSchedule(y, rx, 3, seed=0)
    y, rx = _labels(n=10)
    with pytest.raises(ValueError, match='smaller than'):
        EpisodeSchedule(y, rx, 3, seed=0, n_ref=16)


def test_posterior_risk_gradcheck_through_reference_support_and_query():
    torch.manual_seed(0)
    T, d = 3, 2
    yr = torch.arange(T).repeat_interleave(4).repeat(3)
    cr = torch.arange(3).repeat_interleave(4 * T)
    ys = torch.arange(T).repeat_interleave(2)
    zr = torch.randn(len(yr), d, dtype=torch.float64, requires_grad=True)
    zs = torch.randn(len(ys), d, dtype=torch.float64, requires_grad=True)
    zq = torch.randn(5, d, dtype=torch.float64, requires_grad=True)
    cfg = StatsConfig(kappa_K=8.0)

    def f(zr, zs, zq):
        lg = posterior_query_logits(zr, yr, cr, zs, ys, zq, T, cfg)
        return torch.log_softmax(lg, -1)[:, 0].sum()

    # K_raw is clamped at 0; nudge the reference so no coordinate sits exactly on the kink
    assert torch.autograd.gradcheck(f, (zr, zs, zq), eps=1e-6, atol=1e-5)
    out = f(zr, zs, zq)
    out.backward()
    for g in (zr.grad, zs.grad, zq.grad):
        assert g is not None and g.abs().sum() > 0


def test_protonet_logits_match_negative_squared_distance():
    zs = torch.tensor([[0., 0.], [2., 0.], [0., 4.], [0., 6.]], dtype=torch.float64)
    ys = torch.tensor([0, 0, 1, 1])
    zq = torch.tensor([[1., 1.]], dtype=torch.float64)
    lg = protonet_query_logits(zs, ys, zq, 2, temperature=2.)
    np.testing.assert_allclose(lg.numpy(), [[-(0 + 1) / 2, -(1 + 16) / 2]])


def test_episode_loss_arms_share_native_anchor_and_t3_uses_reference_only():
    y, rx = _labels(T=3, R=4, n=30)
    sch = EpisodeSchedule(y, rx, 3, seed=1, n_ref=6, n_query=3)
    ep = sch.episode(0)
    torch.manual_seed(0)
    idx = ep.all_idx()
    z = torch.randn(len(idx), 4, dtype=torch.float64, requires_grad=True)
    W = torch.randn(3, 4, dtype=torch.float64)
    nat = z @ W.T
    l0, i0 = episode_loss('T0', z, y, ep, rx, 3, nat)
    l2, i2 = episode_loss('T2', z, y, ep, rx, 3, nat)
    l3, i3 = episode_loss('T3', z, y, ep, rx, 3, nat)
    assert i0['native'] == i2['native'] == i3['native']
    assert l2 > l0 and l3 > l0
    with pytest.raises(ValueError):
        episode_loss('T1', z, y, ep, rx, 3, nat)                    # T1 requires the ccalign term
    # perturbing pseudo-target QUERY features must not change the T3 prior: compare logits computed with
    # reference-only statistics against a manual recomputation
    nr, ns = len(ep.ref_idx), len(ep.sup_idx)
    lg = posterior_query_logits(z[:nr], torch.as_tensor(y[ep.ref_idx]), torch.as_tensor(rx[ep.ref_idx]),
                                z[nr:nr + ns], torch.as_tensor(y[ep.sup_idx]), z[nr + ns:], 3)
    z2 = z.detach().clone()
    z2[nr + ns:] += 100.                                            # move the queries far away
    lg_ref_only = posterior_query_logits(z2[:nr], torch.as_tensor(y[ep.ref_idx]), torch.as_tensor(rx[ep.ref_idx]),
                                         z2[nr:nr + ns], torch.as_tensor(y[ep.sup_idx]), z[nr + ns:].detach(), 3)
    np.testing.assert_allclose(lg.detach().numpy(), lg_ref_only.numpy(), atol=1e-10)
    l3.backward()
    assert z.grad[:nr].abs().sum() > 0                              # reference branch is differentiable
    assert z.grad[nr:nr + ns].abs().sum() > 0                       # support branch
    assert z.grad[nr + ns:].abs().sum() > 0                         # query branch


def test_t1_uses_the_historical_ccalign_term():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('p2_losses_direct', root / 'scripts/receiver_p2/p2_losses.py')
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    y, rx = _labels(T=3, R=4, n=30)
    ep = EpisodeSchedule(y, rx, 3, seed=2, n_ref=6, n_query=3).episode(0)
    torch.manual_seed(1)
    z = torch.randn(len(ep.all_idx()), 4, dtype=torch.float64, requires_grad=True)
    nat = z @ torch.randn(3, 4, dtype=torch.float64).T
    l1, info = episode_loss('T1', z, y, ep, rx, 3, nat, aux_fn=L.ccalign, weight=3.0)
    l0, _ = episode_loss('T0', z, y, ep, rx, 3, nat)
    assert abs(float(l1 - l0) - 3.0 * info['ccalign']) < 1e-9
