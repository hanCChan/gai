#!/usr/bin/env python3
"""Numerical tests for p2_losses (CPU, float64). Run: python3 test_p2_losses.py"""
import math
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import p2_losses as L   # noqa: E402

torch.manual_seed(0)
D = torch.float64
EPS = 1e-4
P = [(1, 0), (3, 2)]


def batch(n_rx=5, n_per=2, n_cls=10, dim=128):
    y = torch.arange(n_cls).repeat_interleave(n_per).repeat(n_rx)
    r = torch.arange(n_rx).repeat_interleave(n_cls * n_per)
    pre = torch.rand(len(y), dim, dtype=D)
    W = torch.randn(n_cls, dim, dtype=D)
    b = torch.randn(n_cls, dtype=D)
    return pre, W, b, y, r


def check(name, cond):
    print(('PASS ' if cond else 'FAIL ') + name)
    if not cond:
        check.failed = True


check.failed = False
pre, W, b, y, r = batch()

# 0. a9 row space == full centered row space: A10 D == 0 iff a9 D == 0 (common-mode cancellation)
A10 = L.centered_rows(W)
A9 = L.a9(W)
d0 = pre[0] * 0.3
check('a9 rank equals A10 rank', torch.linalg.matrix_rank(A9) == torch.linalg.matrix_rank(A10))
check('A10 D == 0 iff a9 D == 0', torch.allclose(A10 @ d0, torch.zeros(10, dtype=D)) ==
      bool((A9 @ d0).abs().max() < 1e-9))
# common-mode is structurally impossible: for ANY displacement delta, the logit perturbation
# A delta is exactly zero-mean (1^T A = 0 by row-centering), so argmax damage is purely
# differential-mode and rho measures ALL of it. Check the identity + the null(A) case.
one = torch.ones(10, dtype=D)
check('1^T A = 0 (perturbations exactly differential-mode)', float((one @ A10).abs().max()) < 1e-12)
q, _ = torch.linalg.qr(A9.T)                          # (128, 9) orthonormal basis of row(A)
dc = torch.randn(128, dtype=D)
dc = dc - q @ (q.T @ dc)                              # direction in null(A): NO margin can respond
dc = dc / (dc.norm() + 1e-12)
check('null(A) direction: zero logit perturbation', (A10 @ dc).abs().max() < 1e-12)
rho_c = L._rho(W, [(0, dc)], EPS)
check(f'rho(null(A) D) ~ 0 (got {float(rho_c[0]):.2e})', float(rho_c[0]) < 1e-4)
# orthogonal-to-row(A) D: zero exposure
q, _ = torch.linalg.qr(A9.T)                          # (128, 9) orthonormal basis of row(A)
perp = torch.randn(128, dtype=D)
perp = perp - q @ (q.T @ perp)
perp = perp / perp.norm()
rho_p = L._rho(W, [(0, perp)], EPS)
check(f'rho(D orthogonal to row(A)) ~ 0 (got {float(rho_p[0]):.2e})', float(rho_p[0]) < 1e-3)
# inside-row-space D: rho ~ 1
inside = q[:, 0] * 0.7
rho_i = L._rho(W, [(0, inside)], EPS)
check(f'rho(D inside row(A)) ~ 1 (got {float(rho_i[0]):.4f})', float(rho_i[0]) > 0.99)

# 1. bounds: rxnull in [0,1]
rx = L.rxnull(pre, W, y, r, EPS)
check('rxnull in [0,1]', 0.0 <= float(rx) <= 1.0)

# 2. scale invariance in W and D (tolerance = O(eps/||D||^2) second-order effect)
rx2 = L.rxnull(pre * 17.0, W, y, r, EPS)
rx3 = L.rxnull(pre, W * 100.0, y, r, EPS)
check(f'rho scale-invariant in ||D|| (rx={float(rx):.5f}, rx2={float(rx2):.5f})',
      abs(float(rx2) - float(rx)) < 1e-3)
check(f'rho scale-invariant in ||W|| (rx3={float(rx3):.5f})', abs(float(rx3) - float(rx)) < 1e-8)

# 3. relabel invariance
perm = torch.arange(10)
perm[0], perm[1] = 1, 0
check('relabel invariance', torch.allclose(L.rxnull(pre, W, y, r, EPS),
                                           L.rxnull(pre, W[perm], perm[y], r, EPS)))

# 4. gradcheck
Wg = W.clone().requires_grad_(True)
pg = pre.clone().requires_grad_(True)
f = lambda W_, p_: L.rxnull(p_, W_, y, r, EPS)
check('gradcheck rxnull', torch.autograd.gradcheck(f, (Wg, pg)))

# 5. rank-deficient W (two identical classes -> rank 8 subspace): no NaN, bounded
Wr = W.clone()
Wr[2] = Wr[3]
rr = L.rxnull(pre, Wr, y, r, EPS)
check(f'rank-deficient W: finite and in [0,1] (got {float(rr):.4f})',
      torch.isfinite(rr) and 0.0 <= float(rr) <= 1.0)

# 6. worst-RX concentration: degrading one RX's displacement raises the loss
pre_bad = pre.clone()
m = (y == 1) & (r == 2)
pre_bad[m] += q[:, 3] * 2.0                      # push class-1 mean on RX 2 along a margin direction
check('displacing one RX along a margin direction raises rxnull',
      float(L.rxnull(pre_bad, W, y, r, EPS)) > float(rx))

# 7. cmean unnormalized drops with feature shrink (the cheat rxnull blocks); ||D||^2 >> eps both times
check('cmean shrinks with ||pre||, rxnull does not',
      float(L.cmean(pre * 0.1, W, y, r, EPS)) < 0.02 * float(L.cmean(pre, W, y, r, EPS)) and
      abs(float(L.rxnull(pre * 0.1, W, y, r, EPS)) - float(rx)) < 1e-3)

# 8. rxnull9 equals per-pair margin damage, zero for null(A) displacement
nm = L.rxnull9(pre, W, y, r, P, EPS)
check('rxnull9 finite in [0,1]', torch.isfinite(nm) and 0.0 <= float(nm) <= 1.0)
check('rxnull9 ~ 0 for null(A) D', float(L.rxnull9(torch.stack([dc, dc]),
      W, torch.tensor([0, 0]), torch.tensor([0, 1]), P, EPS)) < 1e-3)
# full-coverage link: rxnull9 over ALL 90 pairs should co-vary with rxnull on real W draws.
# On random W the selected pair directions sample only a rotation of the subspace, so only a
# positive (not near-perfect) rank correlation is asserted; the mechanism-level link on real
# displacements is G2's job (gate), not this unit test's.
allp = [(t, u) for t in range(10) for u in range(10) if t != u]
import numpy as np   # noqa: E402
Ws_ = [torch.randn(10, 128, dtype=D) for _ in range(12)]
a9v = [float(L.rxnull(pre, Wi, y, r, EPS)) for Wi in Ws_]
b9v = [float(L.rxnull9(pre, Wi, y, r, allp, EPS)) for Wi in Ws_]
rho_sp = float(np.corrcoef(np.argsort(np.argsort(a9v)), np.argsort(np.argsort(b9v)))[0, 1])
check(f'rxnull9(all 90) positively rank-correlated with rxnull (rho={rho_sp:.2f})', rho_sp > 0.3)

# 9. random-direction control: E[rho] ~ rank/128 = 9/128 under isotropic g; deterministic in seed
rs = [float(L.rxnull_rand(pre, W, y, r, s, EPS)) for s in (123, 123, 456)]
check(f'rxnull_rand deterministic per seed (0.123={rs[0]:.4f}, repeat={rs[1]:.4f})', rs[0] == rs[1])
check(f'rxnull_rand differs across seeds ({rs[2]:.4f})', rs[0] != rs[2])
mc = float(torch.tensor([float(L.rxnull_rand(pre, W, y, r, 1000 + i, EPS)) for i in range(30)]).mean())
check(f'rxnull_rand mean ~ 9/128 = 0.070 (got {mc:.4f})', abs(mc - 9 / 128) < 0.02)

# 10. drift_v0 verbatim check: matches p1_losses.drift_v0 bitwise on the same inputs
sys.path[:0] = [str(Path(__file__).resolve().parent.parent / 'receiver_p0')]
import p1_losses as L1   # noqa: E402
check('drift_v0 == p1 drift_v0 (bitwise)', torch.equal(
    L.drift_v0(pre, W, y, r, P, 0.5), L1.drift_v0(pre, W, y, r, P, 0.5)))

# 11. proj_row_num == ||P_row D||^2 via explicit projector (same relative ridge)
A = L.a9(W)
G = A @ A.T
eta = 1e-6 * G.diagonal().mean() + 1e-12
P_row = A.T @ torch.linalg.inv(G + eta * torch.eye(9, dtype=D)) @ A
Dx = torch.stack([pre[0] * 0.4, pre[1] * -0.2])
ref = ((P_row @ Dx.T).T ** 2).sum(1)
mine = L.proj_row_num(W, Dx)
check('proj_row_num == dense projector (rtol 1e-5: solve vs inv paths)', torch.allclose(ref, mine, rtol=1e-5, atol=1e-9))

sys.exit(1 if check.failed else 0)
