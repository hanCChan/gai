"""Training-path smoke tests: real encoder wrappers + episode losses, random tensors, CPU, no data."""
import numpy as np
import pytest
import torch

from rffi_calibration.encoders import build_shen_featurecnn, build_wisig_official
from rffi_calibration.episodes import EpisodeSchedule, episode_loss


def _ccalign():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('p2_losses_direct2', root / 'scripts/receiver_p2/p2_losses.py')
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    return L.ccalign


@pytest.mark.parametrize('dataset', ['wisig', 'shen'])
@pytest.mark.parametrize('arm', ['T0', 'T1', 'T2', 'T3'])
def test_one_optimizer_step_per_arm(dataset, arm):
    torch.manual_seed(0)
    T, R, n = 10, 3, 12
    y = np.repeat(np.tile(np.arange(T), R), n)
    rx = np.repeat(np.repeat(np.arange(R), T), n)
    shape = (1, 256, 2) if dataset == 'wisig' else (1, 52, 126)
    w = build_wisig_official(classes=T) if dataset == 'wisig' else build_shen_featurecnn()
    opt = torch.optim.Adam(w.parameters(), lr=1e-3)
    ep = EpisodeSchedule(y, rx, T, seed=3, n_ref=2, n_query=1, budgets=(1, 2)).episode(0)
    x = torch.randn(len(ep.all_idx()), *shape)
    w.train()
    z, logits = w.train_forward(x)
    assert z.shape[1] == w.feature_dim and logits.shape == (len(x), T)
    loss, info = episode_loss(arm, z.double(), y, ep, rx, T, logits.float(),
                              aux_fn=_ccalign() if arm == 'T1' else None)
    loss = loss + w.regularization()
    before = [p.detach().clone() for p in w.parameters()]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(w.parameters(), 5., error_if_nonfinite=True)
    opt.step()
    assert torch.isfinite(loss)
    assert any((p.detach() != b).any() for p, b in zip(w.parameters(), before))
    if arm in ('T2', 'T3'):
        assert 'episode' in info
