"""CPU tests for link_core on synthetic TX x RX grids with planted effects."""
import numpy as np
from link_core import (effects, energies, crossfit_energies, cell_means, effect_directions, erase_oblique,
                       random_basis, erase, loro_prototype_accuracy, rx_equal_accuracy, spearman,
                       complete_grid, error_overlap)

T, R, D, N = 10, 14, 80, 40


def grid(seed, link_scale=1.0, tx_scale=1.0, noise=1.0, n=N, R_=R, a=None):
    """Planted orthogonal subspaces: TX dims 0-8, RX dims 9-13, link dims 14-21."""
    g = np.random.default_rng(seed)
    if a is None:
        a = np.zeros((T, D)); a[:, :9] = tx_scale * g.normal(size=(T, 9))
    b = np.zeros((R_, D)); b[:, 9:14] = g.normal(size=(R_, 5))
    i = np.zeros((T, R_, D)); i[:, :, 14:22] = link_scale * g.normal(size=(T, R_, 8))
    z, tx, rx, pos = [], [], [], []
    for t in range(T):
        for r in range(R_):
            z.append(a[t] + b[r] + i[t, r] + noise * g.normal(size=(n, D)))
            tx += [t] * n; rx += [r] * n; pos += list(range(n))
    return np.concatenate(z), np.array(tx), np.array(rx), np.array(pos)


def halves(z, tx, rx, pos):
    m = np.ones(len(tx), bool); txs, rxs = complete_grid(tx, rx, m)
    return [cell_means(z, tx, rx, m, txs, rxs, h, pos) for h in (0, 1)], m, txs, rxs


def test_exact_decomposition():
    M = np.random.default_rng(0).normal(size=(T, R, 7))
    g, a, b, i = effects(M)
    assert np.allclose(g + a[:, None] + b[None] + i, M)
    assert np.allclose(i.sum(0), 0) and np.allclose(i.sum(1), 0)


def test_crossfit_unbiased_under_null_link():
    plug, cross = [], []
    for s in range(30):
        z, tx, rx, pos = grid(s, link_scale=0.0)
        (A, B), m, txs, rxs = halves(z, tx, rx, pos)
        plug.append(energies(cell_means(z, tx, rx, m, txs, rxs))['link'])
        cross.append(crossfit_energies(A, B)['link'])
    # plug-in absorbs D*noise^2/N*(T-1)(R-1)/(TR) ~ 1.67; cross-fit centres on 0
    assert np.mean(plug) > 1.4, np.mean(plug)
    assert abs(np.mean(cross)) < 0.1, np.mean(cross)


def test_crossfit_recovers_planted_energy():
    vals = [crossfit_energies(*halves(*grid(s, link_scale=1.0))[0]) for s in range(30)]
    link = np.mean([v['link'] for v in vals]); tx = np.mean([v['tx'] for v in vals])
    # expected per-cell link energy 8*(T-1)(R-1)/(TR) ~ 6.69; TX 9*(T-1)/T = 8.1
    assert abs(link - 8 * 9 * 13 / 140) < 0.4, link
    assert abs(tx - 9 * 0.9) < 1.5, tx


def test_identity_preserving_erasure():
    z, tx, rx, pos = grid(1, link_scale=2.0)
    m = np.ones(len(tx), bool); txs, rxs = complete_grid(tx, rx, m)
    V, G, share = effect_directions(z, tx, rx, m, txs, rxs, 8)
    assert share.min() > 0.7, share  # planted link directions are almost pure link
    assert np.allclose(V.T @ G, np.eye(8), atol=1e-8)
    z2 = erase_oblique(z, V, G, z.mean(0))
    assert np.abs((z2 - z.mean(0)) @ V).max() < 1e-8
    A, B = [cell_means(z2, tx, rx, m, txs, rxs, h, pos) for h in (0, 1)]
    e = crossfit_energies(A, B); e0 = crossfit_energies(*halves(z, tx, rx, pos)[0])
    assert e['link'] < 0.15 * e0['link'] and e['tx'] > 0.85 * e0['tx'], (e, e0)
    # random erasure of the same rank barely changes link energy
    zr = erase(z, random_basis(D, 8, 0), z.mean(0))
    er = crossfit_energies(*[cell_means(zr, tx, rx, m, txs, rxs, h, pos) for h in (0, 1)])
    assert er['link'] > 0.7 * e0['link']


def test_loro_accuracy_falls_with_link():
    accs = []
    for scale in (0.0, 3.0):
        z, tx, rx, pos = grid(2, link_scale=scale)
        m = np.ones(len(tx), bool); txs, rxs = complete_grid(tx, rx, m)
        accs.append(loro_prototype_accuracy(z, tx, rx, m, txs, rxs))
    assert accs[0] > accs[1] + 0.2, accs


def test_erasing_link_helps_new_receivers():
    """Mechanism sketch: prototypes from source RX, test on unseen RX with fresh links, same TX."""
    a = np.zeros((T, D)); a[:, :9] = np.random.default_rng(9).normal(size=(T, 9))
    z, tx, rx, pos = grid(3, link_scale=2.5, a=a)
    zt, tt, _, _ = grid(4, link_scale=2.5, R_=6, a=a)
    m = np.ones(len(tx), bool); txs, rxs = complete_grid(tx, rx, m)
    V, G, _ = effect_directions(z, tx, rx, m, txs, rxs, 8)
    Vr, Gr, _ = effect_directions(z, tx, rx, m, txs, rxs, 8, kind='rx')
    P = np.stack([z[tx == t].mean(0) for t in range(T)]); c = z.mean(0); Qr = random_basis(D, 8, 1)
    def acc(f):
        a_, p_ = f(zt), f(P)
        return float((((a_[:, None] - p_[None]) ** 2).sum(-1).argmin(1) == tt).mean())
    base = acc(lambda x: x)
    erased = acc(lambda x: erase_oblique(x, V, G, c))
    rnd = acc(lambda x: erase(x, Qr, c))
    rxe = acc(lambda x: erase_oblique(x, Vr, Gr, c))
    assert erased > max(base, rnd, rxe) + 0.05, (base, erased, rnd, rxe)
    Vc, Gc, _ = effect_directions(z, tx, rx, m, txs, rxs, 13, kind='cond')  # rx (5) + link (8) dims
    cond = acc(lambda x: erase_oblique(x, Vc, Gc, c))
    assert cond > base + 0.05, (base, cond)
    assert erased > base + 0.05 and erased > rnd + 0.05, (base, erased, rnd)


def test_ties_coverage_and_diagnostics():
    from link_core import average_ranks, coverage
    assert list(average_ranks([3, 1, 3, 2])) == [3.5, 1.0, 3.5, 2.0]
    # with ties, double-argsort ranks would give 0.8 here; the average-rank value is 0.9487
    assert abs(spearman([1, 2, 2, 3], [1, 2, 3, 4]) - 0.9486832980505138) < 1e-12
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
    tx = np.array([0, 0, 1, 1, 0, 0, 1]); rx = np.array([0, 0, 0, 0, 1, 1, 1])
    c = coverage(tx, rx, np.ones(7, bool), n_tx=3)
    assert c['complete_tx'] == [0] and c['dropped_tx'] == [1, 2] and c['insufficient_cells'] == 3
    txs, _ = complete_grid(tx, rx, np.ones(7, bool), min_count=1, restrict=[1])
    assert list(txs) == [1]
    z, tx2, rx2, pos = grid(5, link_scale=1.0); m = np.ones(len(tx2), bool); a, b = complete_grid(tx2, rx2, m)
    info = {}; effect_directions(z, tx2, rx2, m, a, b, 4, info=info)
    assert info['q'] == 24 and np.isfinite(info['cond_Bq']) and info['cells'] == 140


def test_small_helpers():
    pred = np.array([0, 1, 1, 0]); y = np.array([0, 1, 0, 0]); rx = np.array([0, 0, 1, 1]); day = np.zeros(4, int)
    assert rx_equal_accuracy(pred, y, rx, day) == 0.75
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    o = error_overlap(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    assert o['jaccard'] == 1 / 3 and o['joint_over_independent'] == 1.0
    txs, rxs = complete_grid(np.array([0, 1, 0, 0, 1]), np.array([0, 0, 1, 0, 0]), np.ones(5, bool), min_count=1)
    assert list(txs) == [0] and list(rxs) == [0, 1]
    assert len(complete_grid(np.array([0, 0, 0]), np.array([0, 0, 1]), np.ones(3, bool))[0]) == 0


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('PASS', name)
