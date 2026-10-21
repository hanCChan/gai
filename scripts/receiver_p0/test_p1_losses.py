#!/usr/bin/env python3
"""Numerical tests for p1_losses (CPU, float64). Run: python3 test_p1_losses.py"""
import math
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import p1_losses as L   # noqa: E402

torch.manual_seed(0)
D = torch.float64
P = [(1, 0), (3, 2)]
GAMMA, TAU_M, TAU_R = 1.0, 0.5, 0.5


def batch(n_rx=5, n_per=2, n_cls=10, dim=16):
    """Balanced batch like balanced_schedule: n_per packets per (RX, TX)."""
    y = torch.arange(n_cls).repeat_interleave(n_per).repeat(n_rx)
    r = torch.arange(n_rx).repeat_interleave(n_cls * n_per)
    pre = torch.rand(len(y), dim, dtype=D)
    W = torch.randn(n_cls, dim, dtype=D)
    b = torch.randn(n_cls, dtype=D)
    return pre, W, b, y, r


def risk(logits, W, y, r, tau_r=TAU_R, pairs=P):
    return L.margin_risk(logits, W, y, r, pairs, GAMMA, TAU_M, tau_r)


def check(name, cond):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond:
        check.failed = True


check.failed = False
pre, W, b, y, r = batch()
logits = pre @ W.T + b

# 1. true-class logit down -> risk up ; 2. competitor logit up -> risk up (A3 and A4)
for tr in (None, TAU_R):
    base = risk(logits, W, y, r, tr)
    lo = logits.clone(); lo[y == 1, 1] -= 0.5
    check(f'true logit down raises risk (tau_r={tr})', risk(lo, W, y, r, tr) > base)
    lo = logits.clone(); lo[y == 1, 0] += 0.5
    check(f'competitor logit up raises risk (tau_r={tr})', risk(lo, W, y, r, tr) > base)

# 3. worse RX raises aggregate and gets larger gradient weight under A4
lg = logits.clone().requires_grad_(True)
lo = lg.clone()
mask = (y == 1) & (r == 2)
lo = lo + torch.zeros_like(lo).index_put((mask.nonzero()[:, 0], torch.full((int(mask.sum()),), 1)),
                                         torch.full((int(mask.sum()),), -2.0, dtype=D))
a4 = risk(lo, W, y, r, TAU_R, [(1, 0)])
check('degrading one RX raises A4', a4 > risk(logits, W, y, r, TAU_R, [(1, 0)]))
a4.backward()
g = lg.grad[:, 1].abs()
w_bad = g[mask].mean()
w_ok = g[(y == 1) & (r != 2)].mean()
check('A4 gradient concentrates on the worse RX', w_bad > 1.5 * w_ok)
lg.grad = None
a3 = risk(lo, W, y, r, None, [(1, 0)])
check('A4 >= A3 (Jensen)', a4.detach() >= a3.detach() - 1e-12)
check('A4 -> A3 as tau_r -> inf', abs(float(risk(logits, W, y, r, 1e6) - risk(logits, W, y, r, None))) < 1e-5)

# 4. class relabeling invariance: swap labels 0<->1 and the pair direction
perm = torch.arange(10); perm[0], perm[1] = 1, 0
check('relabel invariance', torch.allclose(risk(logits, W, y, r, TAU_R, [(1, 0)]),
                                           risk(logits[:, perm], W[perm], perm[y], r, TAU_R, [(0, 1)])))

# 5. gradcheck on logits and W
f = lambda lg_, W_: L.margin_risk(lg_, W_, y, r, P, GAMMA, TAU_M, TAU_R)
check('gradcheck A4', torch.autograd.gradcheck(f, (logits.clone().requires_grad_(True), W.clone().requires_grad_(True))))
f3 = lambda lg_, W_: L.margin_risk(lg_, W_, y, r, P, GAMMA, TAU_M, None)
check('gradcheck A3', torch.autograd.gradcheck(f3, (logits.clone().requires_grad_(True), W.clone().requires_grad_(True))))

# 6. P0 pair branch bug: loss INCREASES with margin (so minimising lowers margins); gamma unused
base = L.pair_p0_buggy(logits, y, r, P)
lo = logits.clone(); lo[y == 1, 1] += 1.0; lo[y == 3, 3] += 1.0
check('P0 pair (buggy): larger margins give LARGER loss', L.pair_p0_buggy(lo, y, r, P) > base)

# 7. drift_v0 ~ softplus(gamma) + kappa*Var for small, centred Delta; proj_var equals that term
g_d = 0.5
small = pre * 0.02
dv = L.drift_v0(small, W, y, r, P, g_d)
pv = L.proj_var(small, W, y, r, P, g_d)
excess = float(dv - torch.nn.functional.softplus(torch.tensor(g_d, dtype=D)))
check(f'drift excess ~= kappa*Var (excess={excess:.3e}, var term={float(pv):.3e})', abs(excess - float(pv)) < 0.05 * float(pv))
check('drift excess >= 0 (convexity, centred Delta)', excess >= -1e-12)

# 8. drift penalises the favourable direction: softplus(g+d) > softplus(g-d)
sp = torch.nn.functional.softplus
check('drift sign: Delta>0 (margin-enlarging) costs more than Delta<0', float(sp(torch.tensor(g_d + .3)) - sp(torch.tensor(g_d - .3))) > 0)

# 9. drift / var can be lowered by shrinking ||w_tu||
W2 = W.clone(); mid = (W2[1] + W2[0]) / 2; W2[1] = mid + (W2[1] - mid) * .5; W2[0] = mid + (W2[0] - mid) * .5
check('proj_var drops when ||w_1-w_0|| halves (P={(1,0)})',
      float(L.proj_var(pre, W2, y, r, [(1, 0)], g_d)) < 0.3 * float(L.proj_var(pre, W, y, r, [(1, 0)], g_d)))

sys.exit(1 if check.failed else 0)
