"""Shared, dataset-agnostic helpers for the D phase (D1 WiSig response interventions, D2 Shen
frozen representations, D2+ CIS input atlas).

No network fitting anywhere in this file. Models are only loaded and evaluated; the one trained
object is the documented source-only refit head of run_l0.py (same recipe, reused below).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

TO_D_MODEL_TYPES = ('OfficialCNN', 'WiSigResNet', 'FeatureCNN')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_root(root):
    """Put the R0/R1 worktree's src and scripts on sys.path and return Path(root)."""
    root = Path(root)
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    return root


def load_model(ckpt):
    """Load a frozen checkpoint exactly as run_l0.py did: status must be completed and the
    selected checkpoint hash must match, or we refuse to evaluate."""
    import torch
    ckpt = Path(ckpt)
    status = json.loads((ckpt / 'status.json').read_text())
    assert status['state'] == 'completed', (ckpt, status.get('state'))
    assert sha(ckpt / 'selected.pt') == status['checkpoint_sha256'], f'checkpoint hash mismatch: {ckpt}'
    return torch.load(ckpt / 'selected.pt', map_location='cpu', weights_only=False), status


def split(model):
    """(encoder, head, fresh same-capacity head factory); model(x) == head(encoder(x))."""
    import torch.nn.functional as F
    from torch import nn
    name = type(model).__name__
    if name == 'OfficialCNN':
        return model.encode, (lambda f: model.head(model.drop(f))), (lambda: nn.Linear(80, 10))
    if name == 'WiSigResNet':
        return (lambda x: model.head[:3](model.body(x))), (lambda f: model.head[3:](f)), (lambda: nn.Linear(80, 10))
    if name == 'FeatureCNN':
        return ((lambda x: model.forward_with_features(x)[1]),
                (lambda f: model.classifier(F.relu(model.tx_input(f)))),
                (lambda: nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 10))))
    raise ValueError(f'unexpected model type {name}')


def run_batches(fn, x, device, batch=512):
    import torch
    with torch.no_grad():
        out = [fn(torch.as_tensor(np.ascontiguousarray(x[i:i + batch]), device=device)).float().cpu()
               for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


REFIT = dict(steps=500, lr=3e-3, l2=1e-4, seed=20261019)


def refit_head(make_head, f, y, device):
    """Source-only refit of a fresh head on frozen features (identical recipe to run_l0.py)."""
    import torch
    from torch.nn import functional as F
    torch.manual_seed(REFIT['seed'])
    h = make_head().to(device)
    opt = torch.optim.Adam(h.parameters(), lr=REFIT['lr'])
    X = torch.as_tensor(np.ascontiguousarray(f), device=device)
    Y = torch.as_tensor(np.ascontiguousarray(y), device=device)
    for _ in range(REFIT['steps']):
        opt.zero_grad()
        loss = F.cross_entropy(h(X), Y) + REFIT['l2'] * sum((p ** 2).sum() for n, p in h.named_parameters() if 'weight' in n)
        loss.backward()
        opt.step()
    h.eval()
    return h


def logits_to_probs(lg):
    e = np.exp(lg - lg.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def logit_stats(lg):
    pr = logits_to_probs(lg)
    return dict(mean_logit_norm=float(np.linalg.norm(lg, axis=1).mean()),
                mean_max_prob=float(pr.max(1).mean()))


def domain_key(data):
    """(RX, date) domain id used by the L0 erasure; date -1 (Shen) keeps every RX a domain."""
    return data['rx'] * 10 + (data['day'] + 1)


def rx_equal_accuracy_checked(pred, data):
    from link_core import rx_equal_accuracy
    return rx_equal_accuracy(pred, data['y'], data['rx'], data['day'])


def per_rx_accuracy(pred, data):
    """{rx: accuracy} equal-weight over that RX's dates (Shen: one pool per RX)."""
    out = {}
    for r in np.unique(data['rx']):
        m = data['rx'] == r
        days = np.unique(data['day'][m])
        out[int(r)] = float(np.mean([(pred[m & (data['day'] == d)] == data['y'][m & (data['day'] == d)]).mean()
                                     for d in days]))
    return out


def per_class_recall(pred, data):
    y = data['y']
    return {int(t): float((pred[y == t] == t).mean()) if (y == t).any() else np.nan for t in np.unique(y)}


def spearman(x, y):
    """Rank correlation with tie-averaged ranks (no scipy on this box)."""
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind='stable')
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        # average ties
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return (sums / cnt)[inv]
    x, y = rank(x), rank(y)
    if x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_ci(values, stat=np.mean, draws=2000, seed=20261019, lo=2.5, hi=97.5):
    """Percentile bootstrap over the given units (RX, seed, or seed x RX means)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    if len(v) < 2:
        return float('nan'), float('nan')
    s = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(draws)])
    return float(np.percentile(s, lo)), float(np.percentile(s, hi))


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o
