"""P2 losses: decision-subspace exposure of per-RX class-mean displacements. Pure functions,
unit-tested in test_p2_losses.py. Notation as PROTOCOL_P2 §1: pre = 128-d pre-head features
(ReLU(tx_input(feat512(x)))), W = live classifier weight (10x128), A = W - mean_rows(W),
row(A) = the 9-dim margin (differential-mode) subspace span{w_t - w_u}; D_{t,r} = mu_{t,r} - mu_t.

Mechanism (P2 hypothesis): for a displacement D of class t's mean, the logit-perturbation vector
is A D (mean-zero over classes: common-mode cancels, argmax untouched). All 90 normalized-margin
perturbations are bounded by sigma_max(A) * ||P_row D||, so the exposure

    rho(D) = ||P_row D||^2 / (||D||^2 + eps)

is exactly the fraction of displacement energy that can damage ANY margin. Minimising mean rho
over every (t, r) makes source-RX displacements common-mode -> unseen-RX shifts that look like
source shifts leave every argmax unchanged. Scale-invariant in W and D (no ||w||-shrink cheat),
covers all classes (no capacity re-alibration list), and drops only a row that is exactly the
negative sum of the kept rows (A9 = centered W with the last row removed: row(A9) == row(A)
because sum_i a_i = 0), so P_row = A9^T (A9 A9^T + eta I)^-1 A9 is an exact projector when eta=0
and numerically safe with the small relative ridge used in training.

  rxnull      lambda * mean_{t,r} rho(D_{t,r})                       (B1, main arm)
  cmean       lambda * mean_{t,r} ||P_row D_{t,r}||^2               (B2, no normalization)
  rxnull9     lambda * mean_{(t,u) in P} mean_r (w_hat_tu^T D_{t,r})^2 / (||D_{t,r}||^2+eps)
              selected-pairs coverage ablation of rxnull (B3)
  rxnull_rand control: D replaced by a same-norm random unit direction, seed-driven (deterministic)
"""
import torch
import torch.nn.functional as F


def centered_rows(W):
    return W - W.mean(0, keepdim=True)


def a9(W):
    """First 9 centered rows: row(a9) == row(W - mean) exactly (dropped row = -sum of kept)."""
    return centered_rows(W)[:9]


def proj_row_num(W, D, eta_rel=1e-6):
    """||P_row D||^2 for a batch of displacements D (n, dim), with a small relative ridge.

    P_row = A9^T (A9 A9^T + eta I)^-1 A9, eta = eta_rel * mean(diag(A9 A9^T)) + 1e-12.
    """
    A = a9(W)
    G = A @ A.T
    eta = eta_rel * G.diagonal().mean() + 1e-12
    G = G + eta * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    U = A @ D.T                                    # (9, n)
    Minv_U = torch.linalg.solve(G, U)
    return (U * Minv_U).sum(0)                     # (n,)


def class_disp(pre, y, rb, min_rx=2):
    """D_{t,r} = mu_{t,r} - mu_t for every (t, r) with class t seen on >= min_rx RX.
    Returns a list of (t, D) tensors; E_r D = 0 exactly on balanced batches."""
    out = []
    for t in torch.unique(y):
        mus = []
        for r in torch.unique(rb):
            m = (y == t) & (rb == r)
            if bool(m.any()):
                mus.append(pre[m].mean(0))
        if len(mus) < min_rx:
            continue
        mus = torch.stack(mus)
        D = mus - mus.mean(0, keepdim=True)
        out.extend((int(t), D[i]) for i in range(mus.shape[0]))
    return out


def _rho(W, ds, eps):
    if not ds:
        return None
    D = torch.stack([d for _, d in ds])
    num = proj_row_num(W, D)
    den = D.pow(2).sum(1) + eps
    return num / den


def rxnull(pre, W, y, rb, eps=1e-4):
    """B1: mean_{t,r} ||P_row D||^2/(||D||^2+eps) in [0, 1]."""
    rho = _rho(W, class_disp(pre, y, rb), eps)
    return rho.mean() if rho is not None else pre.sum() * 0.


def cmean(pre, W, y, rb, eps=1e-4):
    """B2: mean_{t,r} ||P_row D||^2 (unnormalized ablation)."""
    ds = class_disp(pre, y, rb)
    if not ds:
        return pre.sum() * 0.
    D = torch.stack([d for _, d in ds])
    return proj_row_num(W, D).mean()


def rxnull9(pre, W, y, rb, P, eps=1e-4):
    """B3: selected-pairs exposure: mean over (t,u) in P, r of (w_hat_tu^T D_{t,r})^2/(||D||^2+eps)."""
    ds = class_disp(pre, y, rb)
    by_t = {}
    for t, d in ds:
        by_t.setdefault(t, []).append(d)
    terms = []
    for (t, u) in P:
        if t not in by_t:
            continue
        w = W[t] - W[u]
        wn = w / (w.norm() + 1e-12)
        for d in by_t[t]:
            terms.append((wn @ d).pow(2) / (d.pow(2).sum() + eps))
    return torch.stack(terms).mean() if terms else pre.sum() * 0.


def rxnull_rand(pre, W, y, rb, seed, eps=1e-4):
    """Control arm: D_{t,r} replaced by a same-norm random unit direction (deterministic from seed)."""
    ds = class_disp(pre, y, rb)
    if not ds:
        return pre.sum() * 0.
    g = torch.Generator()
    g.manual_seed(int(seed))
    fake = []
    for t, d in ds:
        u = torch.randn(d.numel(), generator=g).to(d.dtype)
        u = u / u.norm().clamp_min(1e-12)
        fake.append((t, u * d.norm()))
    rho = _rho(W, fake, eps)
    return rho.mean()


def drift_v0(pre, W, y, rb, P, gamma):
    """P0 drift verbatim (imported here to keep the P2 trainer single-tree; bitwise same as p1)."""
    base = {}
    per_rx = []
    for r in torch.unique(rb):
        for t in torch.unique(yb_of(y, rb, r)):
            m = (y == t) & (rb == r)
            base.setdefault(int(t), []).append(pre[m].mean(0))
            per_rx.append((int(r), int(t), pre[m].mean(0)))
    gmean = {t: torch.stack(v).mean(0) for t, v in base.items()}
    losses = []
    for (t, u) in P:
        if t not in gmean or u not in gmean:
            continue
        wdu = W[t] - W[u]
        for r, tt, mu_r in per_rx:
            if tt != t:
                continue
            delta = (mu_r - gmean[t]) @ wdu
            losses.append(F.softplus(gamma + delta))
    return torch.stack(losses).mean() if losses else pre.sum() * 0.


def yb_of(y, rb, r):
    return y[rb == r]
