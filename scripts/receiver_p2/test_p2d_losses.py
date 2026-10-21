#!/usr/bin/env python3
"""Numerical tests for the P2D losses rxccp / tailrisk (CPU, float64). Run: python3 test_p2d_losses.py"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import p2d_losses as D   # noqa: E402

torch.manual_seed(7)
DTYPE = torch.float64
y = torch.arange(10).repeat_interleave(2).repeat(5)
r = torch.arange(5).repeat_interleave(20)
pre = torch.rand(len(y), 128, dtype=DTYPE)
W = torch.randn(10, 128, dtype=DTYPE)
b = torch.randn(10, dtype=DTYPE)
logits = pre @ W.T + b
failed = []


def check(name, cond):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond:
        failed.append(name)


# 1. shift extraction
sh0 = D.class_logit_shifts(pre, W, y, r)
check('class_logit_shifts keys == 10 classes', sorted(sh0) == list(range(10)))
check('shifts are common-mode-free (sum over RX == 0 per class)',
      all(float(sh0[t].sum(0).abs().max()) < 1e-9 for t in sh0))
r1 = torch.zeros_like(r)
check('rxccp == 0 when only one RX present', float(D.rxccp(logits, pre, W, y, r1)) == 0.)

# no per-RX displacement -> rxccp == plain CE
pre_const = torch.zeros_like(pre)
for t in range(10):
    pre_const[y == t] = torch.randn(128, dtype=DTYPE)
check('rxccp == plain CE when no per-RX displacement',
      abs(float(D.rxccp(logits, pre_const, W, y, r)) - float(F.cross_entropy(logits, y))) < 1e-12)

# 2. hand-built 3-class case with a KNOWN harmful shift: verify the max picks it
y3 = torch.tensor([0, 0, 1, 1, 2, 2])
r3 = torch.tensor([0, 1, 0, 1, 0, 1])
W3 = torch.cat([torch.eye(3, dtype=DTYPE), torch.zeros(3, 3, dtype=DTYPE)], 1)
pre3 = torch.zeros(6, 6, dtype=DTYPE)
pre3[0, 0] = 2.0; pre3[1, 0] = 2.5     # class 0, rx0/rx1 (rx1 shifted along its own logit)
pre3[2, 1] = 1.5; pre3[3, 1] = 1.5
pre3[4, 2] = 1.0; pre3[5, 2] = 1.0
z3 = pre3 @ W3.T
sh3 = D.class_logit_shifts(pre3, W3, y3, r3)
check('3-class shifts present', sorted(sh3) == [0, 1, 2])
check('classes 1 and 2 have zero shifts', all(float(sh3[t].abs().max()) == 0 for t in (1, 2)))
val = D.rxccp(z3, pre3, W3, y3, r3)
# manual: candidates for class 0 are D[r] = mu_r - mu_mean (rx0 = -half, rx1 = +half);
# the worst (max per-class CE) is the rx1 shift = +(mu1-mu0)/2 in logit space
mu00 = pre3[(y3 == 0) & (r3 == 0)].mean(0)
mu01 = pre3[(y3 == 0) & (r3 == 1)].mean(0)
delta_cand = [ (mu00 - (mu00+mu01)/2) @ D.centered_rows(W3).T,
               (mu01 - (mu00+mu01)/2) @ D.centered_rows(W3).T ]
m = y3 == 0
ces = [float(F.cross_entropy(z3[m] + d, y3[m])) for d in delta_cand]
worst = delta_cand[int(np.argmax(ces))]
manual = (float(F.cross_entropy(z3[m] + worst, y3[m])) * 2 + float(F.cross_entropy(z3[y3 != 0], y3[y3 != 0])) * 4) / 6
check('rxccp == manual worst-shift CE on hand case', abs(float(val) - manual) < 1e-9)

# 3. tailrisk: negative margin tail at same mean -> higher risk
torch.manual_seed(3)
zA = torch.randn(200, 10, dtype=DTYPE) * 0.01
zA[:, 0] += 1.0
yA = torch.zeros(200, dtype=torch.long)
rA = torch.arange(2).repeat_interleave(100)
tA = D.tailrisk(zA, yA, rA)
zB = zA.clone()
zB[::4, 0] -= 4.0        # heavy negative tail spread across BOTH RX cells (mean drops a bit,
                         # but the point is tail mass far below zero)
tB = D.tailrisk(zB, yA, rA)
check('tailrisk increases with negative margin tail at same mean', float(tB) > float(tA))
check('tailrisk in (0, 1)', 0. < float(tA) < 1.)
tr1 = D.tailrisk(zA[:100], yA[:100], torch.zeros(100, dtype=torch.long))
check('tailrisk is None when no class sees 2 RX', tr1 is None)

# 4. gradient flow
pre_g = pre.clone().requires_grad_(True)
W_g = W.clone().requires_grad_(True)
lg = pre_g @ W_g.T
loss = D.rxccp(lg, pre_g, W_g, y, r)
loss.backward()
check('rxccp has gradients on pre', float(pre_g.grad.abs().sum()) > 0)
check('rxccp has gradients on W', float(W_g.grad.abs().sum()) > 0)

# 5. invariance: adding a constant to all logits (pure common mode) leaves rxccp exactly
#    unchanged (softmax invariant; shifts are common-mode-free by construction)
v0 = float(D.rxccp(logits, pre, W, y, r))
v1 = float(D.rxccp(logits + 3.0, pre, W, y, r))
check('rxccp invariant to common-mode logit shift', abs(v1 - v0) < 1e-12)

# 6. harmful vs beneficial shift: rxccp sees the difference where kappa does not
#    class 0 mean margin +0.2; shift A moves it to +0.7 (beneficial), shift B to -0.3 (harmful).
torch.manual_seed(0)
Wd = torch.zeros(2, 4, dtype=DTYPE)
Wd[0, 0] = 1.0; Wd[1, 1] = 1.0
y2 = torch.tensor([0, 0, 1, 1])
r2 = torch.tensor([0, 1, 0, 1])
base_f = torch.zeros(4, 4, dtype=DTYPE)
base_f[:, 3] = 0.4           # shared offset, killed by A
# class 0: rx0 mean at (0.0), rx1 mean displaced along feature 0 (its own logit dir) by +0.5
#   -> margin 0-vs-1 goes 0.2 -> 0.7 (beneficial)
fA = base_f.clone()
fA[0, 0] = 0.0; fA[1, 0] = 0.5
fA[2, 1] = 0.0; fA[3, 1] = 0.0
# class 0 rx1 displaced by -0.5: margin 0.2 -> -0.3 (harmful, flips argmax for the mean packet)
fB = base_f.clone()
fB[0, 0] = 0.0; fB[1, 0] = -0.5
fB[2, 1] = 0.0; fB[3, 1] = 0.0
z2 = fA @ Wd.T
va = float(D.rxccp(fA @ Wd.T, fA, Wd, y2, r2))
vb = float(D.rxccp(fB @ Wd.T, fB, Wd, y2, r2))
check('rxccp penalises the harmful direction more than the beneficial one', vb > va)

# 7. rxgeo: geometry-only variant has gradients and penalises the harmful direction too
pre_g7 = pre.clone().requires_grad_(True)
W_g7 = W.clone().requires_grad_(True)
lg7 = pre_g7 @ W_g7.T
loss7 = D.rxgeo(lg7, pre_g7, W_g7, y, r)
loss7.backward()
check('rxgeo has gradients', float(W_g7.grad.abs().sum()) > 0)
# 2-class hand case: fB harmful (margin 0.2 -> -0.3) must give larger rxgeo than fA beneficial
Wd = torch.zeros(2, 4, dtype=DTYPE); Wd[0, 0] = 1.0; Wd[1, 1] = 1.0
y2 = torch.tensor([0, 0, 1, 1]); r2 = torch.tensor([0, 1, 0, 1])
base_f = torch.zeros(4, 4, dtype=DTYPE); base_f[:, 3] = 0.4
fA = base_f.clone(); fA[0, 0] = 0.0; fA[1, 0] = 0.5; fA[2, 1] = 0.0; fA[3, 1] = 0.0
fB = base_f.clone(); fB[0, 0] = 0.0; fB[1, 0] = -0.5; fB[2, 1] = 0.0; fB[3, 1] = 0.0
ga = float(D.rxgeo(fA @ Wd.T, fA, Wd, y2, r2))
gb = float(D.rxgeo(fB @ Wd.T, fB, Wd, y2, r2))
check('rxgeo penalises the harmful direction more than the beneficial one', gb > ga)

n_pass = len([1])  # placeholder; count real
print('\n%d failed' % len(failed))
sys.exit(1 if failed else 0)
