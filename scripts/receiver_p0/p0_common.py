"""Shared, dataset-agnostic helpers for the P0 pair-margin localization phase.

Inference and documented source-only refits only; no backbone training here.
All rules frozen in docs/receiver_p0/PROTOCOL_P0.md before any P0 output.

Terminology (per packet with true class t, on receiver r):
  margin_{t->u}(x) = logit_t(x) - logit_u(x)          "ordered pair" (t,u), u != t
  q_{r,t,u}        = 10% quantile of margin_{t->u} over packets with y = t on RX r
  conf_{r,t->u}    = P(pred = u | y = t, RX = r)      directional confusion rate
Shen: 90 ordered pairs per (RX, model); the true-10-class prediction must be checked
alongside margins (a positive margin vs u does not preclude confusion into a third class).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

GAI = Path('/workspace/rl1/gai')
ROOT = Path('/workspace/rl1/gai_retrain')   # frozen R0/R1 retrain tree with checkpoints


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_root(root=ROOT):
    """Put the frozen retrain tree's src and scripts on sys.path (idempotent)."""
    root = Path(root)
    entry = str(root / 'src')
    if entry not in sys.path:
        sys.path[:0] = [entry, str(root / 'scripts')]
    return root


def registration():
    return json.loads((ROOT / 'configs/receiver_r01/registration.json').read_text())


def jobs(dataset='shen', stage='formal'):
    reg = registration()
    out = []
    for j in reg[f'{stage}_jobs']:
        if j['dataset'] == dataset:
            out.append(dict(j, ckpt=ROOT / 'outputs/receiver_r01/formal' / j['run_id']))
    return out


def load_model(ckpt):
    """Identical guard to D phase: completed status + checkpoint hash match."""
    import torch
    ckpt = Path(ckpt)
    status = json.loads((ckpt / 'status.json').read_text())
    assert status['state'] == 'completed', (ckpt, status.get('state'))
    assert sha(ckpt / 'selected.pt') == status['checkpoint_sha256'], f'checkpoint hash mismatch: {ckpt}'
    return torch.load(ckpt / 'selected.pt', map_location='cpu', weights_only=False), status


def build_model(job, ckpt_obj):
    from b1_baselines.model import FeatureCNN
    from train_receiver_r01 import build
    if job['dataset'] == 'shen':
        model = FeatureCNN()
    elif job['backbone'] == 'official':
        from wisig_w01_common import OfficialCNN
        model = OfficialCNN(10)
    else:
        from receiver_r01.core import WiSigResNet
        model = WiSigResNet()
    model.load_state_dict(ckpt_obj['model'])
    model.eval()
    return model, ckpt_obj['scaler']


def shen_fold_pools(reg, fold, scaler=None, partition='all'):
    """Feature/label/RX tensors for the fold's source RX (train split only, source-only
    diagnostics) and outer evaluation RX (all packets), reusing receiver_r01.data.shen.

    partition: 'train' = first 160 per TX (fitting packets), 'val' = last 40 per TX,
               'all' = 200. Source pools use 'train' by default (risk rules must not
               touch validation packets used for checkpoint selection).
    """
    from receiver_r01.data import shen as load_shen
    f = reg['shen_folds'][fold]
    keys = reg['shen_allowed']
    src, _ = load_shen(reg, f['inner_train'] + f['inner_validation'], scaler, partition)
    out, _ = load_shen(reg, f['outer_evaluation'], scaler, 'all')
    return src, out


def run_batches(fn, x, device, batch=512):
    import torch
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            outs.append(fn(torch.as_tensor(np.ascontiguousarray(x[i:i + batch]), device=device)).float().cpu())
    return torch.cat(outs).numpy() if outs else np.zeros((0,))


def pair_index(n=10):
    """All ordered pairs (t, u), t != u, as arrays for vectorized margin computation."""
    ts, us = np.triu_indices(n, k=1)
    return np.concatenate([ts, us]), np.concatenate([us, ts])   # both directions


def margin_stats(logits, y, rx, n=10, qs=(0.1, 0.5)):
    """Per (rx, t, u): mean, low quantiles, P(margin<0), confusion rate conf_{t->u}.

    Returns dict of {(rx, t, u): stats} plus per-(rx,t) full-prediction recall.
    """
    P = logits.argmax(1)
    all_t, all_u = pair_index(n)
    out = {}
    for r in np.unique(rx):
        m = rx == r
        lr, yr, pr = logits[m], y[m], P[m]
        for t in range(n):
            mt = yr == t
            if not mt.any():
                continue
            lt = lr[mt]
            for u in range(n):
                if u == t:
                    continue
                marg = lt[:, t] - lt[:, u]
                out[(int(r), t, u)] = dict(
                    n=int(mt.sum()),
                    mean=float(marg.mean()),
                    q10=float(np.quantile(marg, 0.1)),
                    q50=float(np.quantile(marg, 0.5)),
                    frac_neg=float((marg < 0).mean()),
                    conf=float((pr[mt] == u).mean()),
                )
    return out


def effective_rank(f):
    ev = np.clip(np.linalg.eigvalsh(np.cov(f.T)), 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30))


def class_stats(features, y, rx, t, u):
    """Between-mean distance and within scatter ratio in a feature space (d_{r,t,u})."""
    m = rx
    ft, fu = features[m & (y == t)], features[m & (y == u)]
    if len(ft) == 0 or len(fu) == 0:
        return None
    mu_t, mu_u = ft.mean(0), fu.mean(0)
    s_t = np.trace(np.cov(ft.T)) if len(ft) > 1 else 0.0
    s_u = np.trace(np.cov(fu.T)) if len(fu) > 1 else 0.0
    d = float(np.linalg.norm(mu_t - mu_u) / (np.sqrt(s_t + s_u) + 1e-8))
    return dict(d_class=d, dist=float(np.linalg.norm(mu_t - mu_u)), within=float(s_t + s_u))


REFIT = dict(steps=500, lr=3e-3, l2=1e-4, seed=20261019)   # identical to interaction_l0 run_l0


def refit_head(make_head, f, y, device, steps=None, lr=None):
    """Source-only refit of a fresh head on frozen features (r1/L0 recipe, reused verbatim)."""
    import torch
    from torch.nn import functional as F
    torch.manual_seed(REFIT['seed'])
    h = make_head().to(device)
    opt = torch.optim.Adam(h.parameters(), lr=lr or REFIT['lr'])
    X = torch.as_tensor(np.asarray(f, dtype=np.float32), device=device)
    Y = torch.as_tensor(y, device=device)
    losses = []
    for step in range(steps or REFIT['steps']):
        opt.zero_grad()
        loss = F.cross_entropy(h(X), Y) + REFIT['l2'] * sum((p ** 2).sum() for n, p in h.named_parameters() if 'weight' in n)
        loss.backward()
        opt.step()
        if step in (0, (steps or REFIT['steps']) - 1):
            losses.append(float(loss.detach()))
    h.eval()
    return h, dict(refit_loss_first=losses[0], refit_loss_last=losses[-1])


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def manifest(paths, extra=None):
    """Code SHA manifest for reproducibility (same convention as D phase)."""
    m = {p.name: sha(p) for p in paths}
    if extra:
        m.update(extra)
    return m
