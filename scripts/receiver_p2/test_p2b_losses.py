#!/usr/bin/env python3
"""Numerical tests for the P2B losses rxlogit / ccalign (CPU, float64). Run: python3 test_p2b_losses.py"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import p2_losses as L   # noqa: E402

torch.manual_seed(1)
D = torch.float64
y = torch.arange(10).repeat_interleave(2).repeat(5)
r = torch.arange(5).repeat_interleave(20)
pre = torch.rand(len(y), 128, dtype=D)
W = torch.randn(10, 128, dtype=D)
b = torch.randn(10, dtype=D)
failed = []


def check(name, cond):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond:
        failed.append(name)


A = L.centered_rows(W)
P_row = torch.linalg.pinv(A) @ A
P_null = torch.eye(128, dtype=D) - P_row
k0 = L.rxlogit(pre, W, y, r)
c0 = L.ccalign(pre, y, r)

# 1. rxlogit invariant to the function-preserving reparameterisation that breaks rho
for s in (0.25, 4.0, 16.0):
    p2 = pre @ P_row + s * (pre @ P_null)
    check(f'rxlogit invariant under null rescale s={s}', abs(float(L.rxlogit(p2, W, y, r) - k0)) < 1e-9 * float(k0))
    check(f'rxnull (rho) NOT invariant under s={s}', abs(float(L.rxnull(p2, W, y, r) - L.rxnull(pre, W, y, r))) > 1e-3)

# 2. scale invariance
check('rxlogit scale-invariant in W (rel 1e-5, eps effect)', abs(float(L.rxlogit(pre, 7 * W, y, r) / k0 - 1)) < 1e-5)
check('rxlogit scale-invariant in h (rel 1e-5)', abs(float(L.rxlogit(3 * pre, W, y, r) / k0 - 1)) < 1e-5)
check('ccalign scale-invariant in h (rel 1e-5)', abs(float(L.ccalign(5 * pre, y, r) / c0 - 1)) < 1e-5)

# 3. rxlogit equals kappa computed from centred logits
z = pre @ W.T + b
zc = z - z.mean(1, keepdim=True)
mz = torch.stack([torch.stack([zc[(y == t) & (r == q)].mean(0) for q in range(5)]) for t in range(10)])
AD = mz - mz.mean(1, keepdim=True)
AC = mz.mean(1) - mz.mean(1).mean(0, keepdim=True)
kap = AD.pow(2).sum(-1).mean() / AC.pow(2).sum(-1).mean()
check('rxlogit == kappa from logits (rel 1e-5)', abs(float(kap / k0 - 1)) < 1e-5)

# 4. displacement inside null(A): rxlogit ~ 0, ccalign > 0
pre_n = pre.clone()
mu_t = torch.stack([pre[y == t].mean(0) for t in range(10)])
for t in range(10):
    for q in range(5):
        m = (y == t) & (r == q)
        pre_n[m] = pre[m] - pre[m].mean(0) + mu_t[t] + P_null @ torch.randn(128, dtype=D) * 0.3
check('null(A) displacement: rxlogit ~ 0', float(L.rxlogit(pre_n, W, y, r)) < 1e-12)
check('null(A) displacement: ccalign > 0', float(L.ccalign(pre_n, y, r)) > 1e-3)

# 5. no RX displacement: both zero
pre_0 = pre.clone()
for t in range(10):
    for q in range(5):
        m = (y == t) & (r == q)
        pre_0[m] = pre[m] - pre[m].mean(0) + mu_t[t]
check('no RX displacement: rxlogit == 0', float(L.rxlogit(pre_0, W, y, r)) < 1e-20)
check('no RX displacement: ccalign == 0', float(L.ccalign(pre_0, y, r)) < 1e-20)

# 6. gradcheck
check('gradcheck rxlogit', torch.autograd.gradcheck(lambda W_, p_: L.rxlogit(p_, W_, y, r),
                                                    (W.clone().requires_grad_(True), pre.clone().requires_grad_(True))))
check('gradcheck ccalign', torch.autograd.gradcheck(lambda p_: L.ccalign(p_, y, r), (pre.clone().requires_grad_(True),)))

# 7. relabel invariance
perm = torch.randperm(10)
inv = torch.argsort(perm)
check('rxlogit relabel invariance', abs(float(L.rxlogit(pre, W[inv], perm[y], r) - k0)) < 1e-9)

sys.exit(1 if failed else 0)
