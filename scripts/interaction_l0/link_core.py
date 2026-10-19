"""L0 link-shortcut diagnostics: numpy only, no data access, unit-tested on synthetic grids.

Notation: features z [n,d]; tx, rx integer labels; cell (t,r) = packets of TX t seen by RX r
on one date. Cell means M[t,r] decompose exactly (two-way ANOVA, balanced grid) into
grand + TX main a_t + RX main b_r + link interaction i_tr. "Link" = the part of a TX's
representation that is specific to the TX->RX pair, i.e. neither a TX property shared by
all receivers nor a receiver offset shared by all transmitters.
"""
import numpy as np

def complete_grid(tx, rx, mask, min_count=2):
    """TX and RX lists of a complete grid under mask: keep TX with >= min_count packets at
    every RX (two are needed for the split-half cross-fit)."""
    rxs = np.unique(rx[mask])
    txs = [t for t in np.unique(tx[mask]) if all(((tx == t) & (rx == r) & mask).sum() >= min_count for r in rxs)]
    return np.array(txs), rxs


def cell_means(z, tx, rx, mask, txs, rxs, half=None, order=None):
    """[T,R,d] means; half in {0,1} keeps alternate packets of each cell ordered by `order`."""
    out = np.zeros((len(txs), len(rxs), z.shape[1]))
    for i, t in enumerate(txs):
        for j, r in enumerate(rxs):
            idx = np.flatnonzero((tx == t) & (rx == r) & mask)
            if half is not None:
                idx = idx[np.argsort(order[idx], kind='stable')][half::2]
            assert len(idx), (t, r, half)
            out[i, j] = z[idx].mean(0)
    return out


def effects(M):
    g = M.mean((0, 1))
    a = M.mean(1) - g
    b = M.mean(0) - g
    return g, a, b, M - a[:, None] - b[None] - g


def energies(M):
    """Plug-in per-cell energies (biased upward by within-cell noise)."""
    _, a, b, i = effects(M)
    T, R = M.shape[:2]
    return dict(tx=float((a * a).sum() / T), rx=float((b * b).sum() / R), link=float((i * i).sum() / (T * R)))


def crossfit_energies(MA, MB):
    """Unbiased energies from two independent packet halves: <effect_A, effect_B>.

    Within-cell noise is independent across halves, so it cancels in expectation.
    Values can be slightly negative when the true component is ~0.
    """
    _, aA, bA, iA = effects(MA)
    _, aB, bB, iB = effects(MB)
    T, R = MA.shape[:2]
    return dict(tx=float((aA * aB).sum() / T), rx=float((bA * bB).sum() / R), link=float((iA * iB).sum() / (T * R)))


def crossfit_scatter(MA, MB):
    """Noise-free (in expectation) effect scatter matrices [d,d] from two packet halves."""
    _, aA, bA, iA = effects(MA)
    _, aB, bB, iB = effects(MB)
    iA, iB = (i.reshape(-1, MA.shape[2]) for i in (iA, iB))
    sym = lambda u, v: (u.T @ v + v.T @ u) / 2
    return dict(tx=sym(aA, aB), rx=sym(bA, bB), link=sym(iA, iB))


def within_cell_covariance(z, tx, rx, mask, txs, rxs):
    """Pooled within-cell covariance [d,d] and the harmonic-mean packets per cell."""
    d = z.shape[1]; W = np.zeros((d, d)); dof = 0; ns = []
    for t in txs:
        for r in rxs:
            x = z[(tx == t) & (rx == r) & mask]
            x = x - x.mean(0); W += x.T @ x; dof += len(x) - 1; ns.append(len(x))
    return W / dof, len(ns) / np.sum(1 / np.array(ns))


def effect_directions(z, tx, rx, mask, txs, rxs, k, kind='link', q=24, ridge=1e-6):
    """Readout/pattern pairs for the k directions maximising noise-corrected `kind` energy
    over all between-cell energy (kind in link | rx | cond | tx).

    Numerator: plug-in effect scatter minus its expected noise part (dof/n * Sigma_w), with
    Sigma_w pooled over all packets (a two-half cross-fit of a full d x d matrix is too noisy).
    Denominator B: plug-in TX+RX+link between-cell scatter (PSD).
    With (T-1)(R-1) dof comparable to d (WiSig 117 vs 80; Shen 36 vs 512) sample scatter is
    Wishart-dispersed, so the problem is solved inside the top-q principal subspace of B
    (PCA-then-discriminant); q is fixed in advance, never tuned on target receivers.
    Returns readout V [d,k], pattern G V [d,k] with V'GV = I, and the k energy shares.
    """
    M = cell_means(z, tx, rx, mask, txs, rxs)
    T, R, d = M.shape
    _, a, b, i = effects(M)
    flat = i.reshape(-1, d)
    Sw, n = within_cell_covariance(z, tx, rx, mask, txs, rxs)
    scat = dict(tx=(R * a.T @ a, T - 1), rx=(T * b.T @ b, R - 1), link=(flat.T @ flat, (T - 1) * (R - 1)))
    # 'cond' = full class-conditional domain shift mu_tr - mu_t = RX main + link (CIDG-style target)
    scat['cond'] = (scat['rx'][0] + scat['link'][0], scat['rx'][1] + scat['link'][1])
    S = scat[kind][0] - scat[kind][1] / n * Sw
    B = sum(scat[k][0] for k in ('tx', 'rx', 'link'))
    P = np.linalg.eigh(B)[1][:, ::-1][:, :min(q, d)]
    Sq, Bq = P.T @ S @ P, P.T @ B @ P
    Bq = Bq + ridge * np.trace(Bq) / len(Bq) * np.eye(len(Bq))
    Li = np.linalg.inv(np.linalg.cholesky(Bq))
    w, U = np.linalg.eigh(Li @ Sq @ Li.T)
    Vq = Li.T @ U[:, ::-1][:, :k]
    return P @ Vq, P @ Bq @ Vq, w[::-1][:k]


def erase_oblique(z, readout, pattern, center):
    """LEACE-style: zero the readouts readout'(z-c) by moving along the patterns."""
    return z - ((z - center) @ readout) @ pattern.T


def random_basis(d, k, seed):
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(d, k)))[0]


def erase(z, Q, center):
    """Remove the span of Q around `center` (a source-only mean); other coordinates untouched."""
    return z - ((z - center) @ Q) @ Q.T


def loro_prototype_accuracy(z, tx, rx, mask, txs, rxs):
    """Leave-one-source-RX-out nearest-prototype accuracy, RX-equal mean.

    Prototype of TX t for held RX r = mean of t's cell means at the other source RX.
    Uses only source packets; measures how much a TX's representation is link-specific.
    """
    M = cell_means(z, tx, rx, mask, txs, rxs)
    accs = []
    for j, r in enumerate(rxs):
        P = np.delete(M, j, axis=1).mean(1)
        sel = mask & (rx == r) & np.isin(tx, txs)
        d2 = ((z[sel][:, None, :] - P[None]) ** 2).sum(-1)
        accs.append(float((txs[d2.argmin(1)] == tx[sel]).mean()))
    return float(np.mean(accs))


def rx_equal_accuracy(pred, y, rx, day):
    """Official R0/R1 aggregation: accuracy per RX x date, dates equal within RX, then RX equal."""
    per_rx = []
    for r in np.unique(rx):
        per_rx.append(np.mean([(pred[(rx == r) & (day == d)] == y[(rx == r) & (day == d)]).mean()
                               for d in np.unique(day[rx == r])]))
    return float(np.mean(per_rx))


def spearman(x, y):
    rx_, ry_ = (np.argsort(np.argsort(v)).astype(float) for v in (np.asarray(x), np.asarray(y)))
    return float(np.corrcoef(rx_, ry_)[0, 1])


def error_overlap(wrong_a, wrong_b):
    """Jaccard of error sets and the ratio of joint error rate to the independence expectation."""
    both = float((wrong_a & wrong_b).mean())
    either = float((wrong_a | wrong_b).mean())
    expected = float(wrong_a.mean() * wrong_b.mean())
    return dict(jaccard=both / either if either else np.nan, joint=both,
                joint_over_independent=both / expected if expected else np.nan)
