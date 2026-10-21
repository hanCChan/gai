"""P1 auxiliary losses on hard ordered pairs P (true t, competitor u). Pure functions, unit-tested
in test_p1_losses.py. Notation: pre = 128-d pre-head features (ReLU(tx_input(f))), W/b = live
linear head, logits = pre @ W.T + b, w_tu = W[t]-W[u], mu_{t,r} = batch mean of pre over y=t, RX=r.

  drift_v0   P0 'drift' verbatim:  mean softplus(gamma + Delta_{t,u,r}),
             Delta = w_tu^T (mu_{t,r} - mean_r mu_{t,r}).  Because mean_r Delta = 0 exactly
             (balanced batches contain every source RX), its Taylor expansion is
             softplus(gamma) + kappa*Var_r(Delta) + O(Delta^3), kappa = s(gamma)(1-s(gamma))/2.
             NOTE (P1 review): +Delta penalizes the FAVOURABLE direction (Delta>0 enlarges the t-vs-u
             margin), and the loss falls when ||w_tu|| shrinks. It is a decision-projection
             dispersion regulariser, not a harmful-drift penalty.
  proj_var   explicit second-order truncation of drift_v0: kappa * mean Var_r(Delta)  (A2).
  margin_risk
             normalized margin m = (l_t - l_u)/(||w_tu|| + eps), packet risk
             softplus((gamma - m)/tau_m).
               tau_r=None : mean over all y=t packets              (A3, RX-agnostic)
               tau_r>0    : per-RX mean L_r, then tau_r*log mean_r exp(L_r/tau_r)   (A4, worst-RX)
             A4 -> A3 as tau_r -> inf on balanced batches; A4 >= A3 (Jensen).
  pair_p0_buggy
             the P0 'pair' branch kept only for replay/tests: logsumexp over RX of the batch-min
             margin; gamma unused; minimising it LOWERS hard-pair margins (sign error).
"""
import math

import torch
import torch.nn.functional as F


def kappa(gamma):
    s = 1.0 / (1.0 + math.exp(-gamma))
    return s * (1 - s) / 2


def drift_v0(pre, W, yb, rb, P, gamma):
    """Verbatim P0 l_drift (same loop order, same ops) so P0 drift fits replay bitwise."""
    base = {}
    per_rx = []
    for r in torch.unique(rb):
        for t in torch.unique(yb[rb == r]):
            m = (yb == t) & (rb == r)
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


def _deltas(pre, W, yb, rb, P):
    """Delta_{t,u,r} as a list of (|R|,) tensors, one per (t,u) in P."""
    out = []
    rxs = torch.unique(rb)
    for (t, u) in P:
        mus = [pre[(yb == t) & (rb == r)].mean(0) for r in rxs if bool(((yb == t) & (rb == r)).any())]
        if len(mus) < 2:
            continue
        mu = torch.stack(mus)
        out.append((mu - mu.mean(0)) @ (W[t] - W[u]))
    return out


def proj_var(pre, W, yb, rb, P, gamma):
    """kappa(gamma) * mean over P of Var_r(Delta) (population variance)."""
    d = _deltas(pre, W, yb, rb, P)
    if not d:
        return pre.sum() * 0.
    return kappa(gamma) * torch.stack([x.pow(2).mean() for x in d]).mean()


def norm_margin(logits, W, t, u, eps=1e-6):
    return (logits[:, t] - logits[:, u]) / ((W[t] - W[u]).norm() + eps)


def margin_risk(logits, W, yb, rb, P, gamma, tau_m, tau_r=None):
    per_pair = []
    for (t, u) in P:
        mt = yb == t
        if not bool(mt.any()):
            continue
        risk = F.softplus((gamma - norm_margin(logits[mt], W, t, u)) / tau_m)
        if tau_r is None:
            per_pair.append(risk.mean())
            continue
        rt = rb[mt]
        Lr = torch.stack([risk[rt == r].mean() for r in torch.unique(rt)])
        per_pair.append(tau_r * (torch.logsumexp(Lr / tau_r, 0) - math.log(len(Lr))))
    return torch.stack(per_pair).mean() if per_pair else logits.sum() * 0.


def pair_p0_buggy(logits, yb, rb, P, tau=1.0):
    per_pair = []
    for (t, u) in P:
        qs = []
        for r in torch.unique(rb):
            m = (yb == t) & (rb == r)
            if int(m.sum()) == 0:
                continue
            qs.append((logits[m, t] - logits[m, u]).min())
        if qs:
            per_pair.append(torch.logsumexp(torch.stack(qs) / tau, 0) * tau)
    return torch.stack(per_pair).mean() if per_pair else logits.sum() * 0.
