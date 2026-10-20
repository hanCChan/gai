#!/usr/bin/env python3
"""D2+: Strict CIS input-space atlas for Shen, joined with the saved R0/R1 predictions.

Model-free (no GPU). Rules: docs/receiver_d/PROTOCOL_D.md. Strict CIS cache only, raw IQ never read.

Rows of the 52 x 126 Strict CIS map are STFT bins of 7.8125 kHz (row r <-> bin r-26). Bands:
  in-band     |bin| <= 8  (rows 18..34; inside the |f| <= 62.5 kHz strict filter)
  transition  9..12       (rows 14..17, 35..38; boxcar-window leakage)
  out-of-band >= 13       (the remaining 27 rows; leakage and the 1e-12 floor only)
Per band, the packet's band sub-map is the vector (packets are preamble-aligned; cell-mean maps of
two disjoint halves correlate ~1.0). With source-only statistics of each fold (grand mean, TX mean
over source RX; leave-one-RX-out for the source RX themselves):
  b_r          = mean_t (M[t,r] - g - a_t)                         receiver offset
  d_RX(r)      = ||b_r|| / median_source ||b||                     receiver-condition shift
  d_class(r)   = ||i_{.,r}||_F / ||a||_F, i = M - g - a - b_r      TX-structure distortion
  sep_ratio    = ||M[u,r]-M[v,r]|| / median_source ||M[u,s]-M[v,s]|| for the TX pairs SWAD confuses
Joined with predictions: CE / SWAD outer accuracy per RX, Delta_r = SWAD - CE, per-TX recall,
pair confusion rates. Spearman over the 10 RX (each RX is outer in exactly one fold) with a
permutation p-value; exploratory, n = 10.

    python3 d2plus_atlas.py --root /workspace/rl1/gai_retrain --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'interaction_l0')]
import d_common as C  # noqa: E402
from link_core import crossfit_energies  # noqa: E402

BIN = np.arange(52) - 26
BANDS = dict(in_band=np.abs(BIN) <= 8, transition=(np.abs(BIN) >= 9) & (np.abs(BIN) <= 12), out_of_band=np.abs(BIN) >= 13)
HARDWARE = {'b200_mini': 'AD9364 (USRP B200mini)', 'b210': 'AD9361 (USRP B210)', 'pluto': 'AD9363 (ADALM-Pluto)',
            'rtl': 'R820T/RTL2832U (RTL-SDR)'}
PAIRS = ((1, 0), (3, 2), (8, 7))     # (true, predicted) pairs SWAD collapses at pluto_1 in fold 0
METHODS = ('ce', 'swad', 'pcgrad', 'mldg_rx', 'groupdro')
PERMUTATIONS = 20000


def family(key):
    for k, v in HARDWARE.items():
        if k in key:
            return v
    return '?'


def load(reg):
    root = Path(reg['shen_cache'])
    X, Y = {}, {}
    for key in reg['shen_allowed']:
        X[key] = np.load(root / key / 'strict_cis.npy', mmap_mode='r')
        Y[key] = np.load(root / key / 'labels.npy')
    return X, Y


def band_vectors(X, Y, band, rep):
    """{key: [n,d]} per-packet band representation: the band sub-map ('map') or its per-row std over
    time ('rowstd', invariant to small time shifts)."""
    rows = BANDS[band]
    out = {}
    for key, x in X.items():
        a = np.asarray(x[:, 0][:, rows, :], dtype=np.float64)
        out[key] = a.reshape(len(a), -1) if rep == 'map' else a.std(2)
    return out


def cell_mean(v, y, half=None):
    """[T,d] TX means; half in {0,1} takes alternate packets of each TX."""
    M = []
    for t in range(10):
        idx = np.flatnonzero(y == t)
        idx = idx if half is None else idx[half::2]
        M.append(v[idx].mean(0))
    return np.stack(M)


def stats_for_fold(V, Y, source, all_keys):
    """Source-only decomposition and d_RX / d_class / pair separation for every RX of the fold."""
    Ms = {k: cell_mean(V[k], Y[k]) for k in all_keys}
    def ref(excl=None):
        S = [k for k in source if k != excl]
        M = np.stack([Ms[k] for k in S], 1)                 # [T,R,d]
        g = M.mean((0, 1))
        a = M.mean(1) - g
        b = {k: (Ms[k] - g - a).mean(0) for k in S}
        return g, a, b, M
    g0, a0, b0, M0 = ref()
    out = {}
    for k in all_keys:
        g, a, b, M = ref(excl=k) if k in source else (g0, a0, b0, M0)
        br = (Ms[k] - g - a).mean(0)
        i = Ms[k] - g - a - br
        nb = np.median([np.linalg.norm(v) for v in b.values()])
        sep = {}
        for u, w in PAIRS:
            src_sep = np.median([np.linalg.norm(M[u, j] - M[w, j]) for j in range(M.shape[1])])
            sep[f'sep_{u}_{w}'] = float(np.linalg.norm(Ms[k][u] - Ms[k][w]) / max(src_sep, 1e-30))
        nearest = min(np.linalg.norm(br - v) for v in b.values()) / max(nb, 1e-30)
        out[k] = dict(d_rx=float(np.linalg.norm(br) / max(nb, 1e-30)), d_rx_nearest=float(nearest),
                      d_class=float(np.linalg.norm(i) / max(np.linalg.norm(a), 1e-30)), **sep)
    # cross-fit energies on the fold's source RX (two halves per TX)
    A = np.stack([cell_mean(V[k], Y[k], 0) for k in source], 1)
    B = np.stack([cell_mean(V[k], Y[k], 1) for k in source], 1)
    e = crossfit_energies(A, B)
    tot = e['tx'] + e['rx'] + e['link']
    e.update(tx_share=e['tx'] / tot, rx_share=e['rx'] / tot, link_share=e['link'] / tot)
    return out, e


def perm_spearman(x, y, rng):
    r = C.spearman(x, y)
    null = np.array([C.spearman(x, rng.permutation(np.asarray(y))) for _ in range(PERMUTATIONS)])
    return r, float((np.abs(null) >= abs(r) - 1e-12).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    reg = json.loads((root / 'configs/receiver_r01/registration.json').read_text())
    bad = [x['path'] for x in reg['input_assets'] if 'shen_p2a8' in x['path'] and C.sha(x['path']) != x['sha256']]
    assert not bad, bad
    X, Y = load(reg)
    keys = reg['shen_allowed']

    # ---------------------------------------------------------------- predictions (R0/R1 outputs)
    ev = root / 'outputs/receiver_r01/evaluation'
    pred_rows, pair_rows, tx_rows = [], [], []
    for f in sorted(ev.glob('shen_rank16_CV*_source_pool_outer_RX.npz')):
        name = f.name.replace('_source_pool_outer_RX.npz', '')
        fold = int(name.split('_CV')[1][0])
        method = name.split(f'_CV{fold}_')[1].rsplit('_seed', 1)[0]
        seed = int(name.rsplit('_seed', 1)[1])
        d = np.load(f)
        for rv in np.unique(d['rx']):
            m = d['rx'] == rv
            pr, y = d['prediction'][m], d['y'][m]
            pred_rows.append(dict(fold=fold, method=method, seed=seed, rx_key=keys[rv], acc=float((pr == y).mean())))
            for u, w in PAIRS:
                pair_rows.append(dict(fold=fold, method=method, seed=seed, rx_key=keys[rv], pair=f'{u}->{w}',
                                      rate=float((pr[y == u] == w).mean())))
            for t in range(10):
                tx_rows.append(dict(fold=fold, method=method, seed=seed, rx_key=keys[rv], tx=t,
                                    recall=float((pr[y == t] == t).mean())))
    P = pd.DataFrame(pred_rows)
    PP = pd.DataFrame(pair_rows)
    PT = pd.DataFrame(tx_rows)
    for nm, df in (('outer_acc', P), ('pair_confusion', PP), ('per_tx_recall', PT)):
        df.to_csv(out / f'{nm}.csv', index=False)
    acc = P.groupby(['rx_key', 'method']).acc.mean().unstack()
    acc['fold'] = P.groupby('rx_key').fold.first()
    acc['delta_swad'] = acc['swad'] - acc['ce']
    acc['delta_pcgrad'] = acc['pcgrad'] - acc['ce']
    acc['delta_swad_seed_min'] = (P[P.method == 'swad'].set_index(['rx_key', 'seed']).acc
                                  - P[P.method == 'ce'].set_index(['rx_key', 'seed']).acc).groupby('rx_key').min()
    acc['hardware'] = [family(k) for k in acc.index]

    # ---------------------------------------------------------------- input-space atlas
    atlas, energy_rows = [], []
    for band in BANDS:
        for rep in ('map', 'rowstd'):
            V = band_vectors(X, Y, band, rep)
            for fold, fd in enumerate(reg['shen_folds']):
                source = fd['inner_train'] + fd['inner_validation']
                st, e = stats_for_fold(V, Y, source, keys)
                energy_rows.append(dict(band=band, rep=rep, fold=fold, **e))
                for k, s in st.items():
                    atlas.append(dict(band=band, rep=rep, fold=fold, rx_key=k,
                                      role='source' if k in source else 'outer', **s))
            # all-10-RX energies: diagnostic of how much receiver structure the input still holds
            Aall = np.stack([cell_mean(V[k], Y[k], 0) for k in keys], 1)
            Ball = np.stack([cell_mean(V[k], Y[k], 1) for k in keys], 1)
            e = crossfit_energies(Aall, Ball)
            tot = e['tx'] + e['rx'] + e['link']
            energy_rows.append(dict(band=band, rep=rep, fold='all10', **e, tx_share=e['tx'] / tot,
                                    rx_share=e['rx'] / tot, link_share=e['link'] / tot))
    A = pd.DataFrame(atlas)
    E = pd.DataFrame(energy_rows)
    A.to_csv(out / 'atlas.csv', index=False)
    E.to_csv(out / 'band_energies.csv', index=False)

    # ---------------------------------------------------------------- join: each RX in the fold where it is outer
    rng = np.random.default_rng(20261019)
    J = A[A.role == 'outer'].merge(acc.reset_index()[['rx_key', 'ce', 'swad', 'pcgrad', 'mldg_rx', 'groupdro', 'fold',
                                                     'delta_swad', 'delta_pcgrad', 'delta_swad_seed_min', 'hardware']],
                                   on=['rx_key', 'fold'])
    J.to_csv(out / 'outer_join.csv', index=False)
    corr = []
    for (band, rep), g in J.groupby(['band', 'rep']):
        for xname in ('d_rx', 'd_rx_nearest', 'd_class'):
            for yname, yv in (('ce_error', 1 - g.ce), ('delta_swad', g.delta_swad), ('delta_pcgrad', g.delta_pcgrad)):
                r, pv = perm_spearman(g[xname].values, yv.values, rng)
                corr.append(dict(band=band, rep=rep, x=xname, y=yname, n=len(g), spearman=r, perm_p=pv))
    Cr = pd.DataFrame(corr)
    Cr.to_csv(out / 'correlations.csv', index=False)

    # ---------------------------------------------------------------- Pluto and counterexamples
    S = {}
    for (band, rep), g in J.groupby(['band', 'rep']):
        g = g.set_index('rx_key')
        pl = g.loc['shen_pluto_1_train']
        rank_rx = int((g.d_rx >= pl.d_rx).sum())
        rank_cls = int((g.d_class >= pl.d_class).sum())
        counter = g[(g.d_rx >= pl.d_rx) & (g.delta_swad >= 0) & (g.index != 'shen_pluto_1_train')]
        S[f'{band}/{rep}'] = dict(pluto_d_rx=float(pl.d_rx), pluto_d_rx_rank_of_10=rank_rx,
                                  pluto_d_class=float(pl.d_class), pluto_d_class_rank_of_10=rank_cls,
                                  pluto_sep={c: float(pl[c]) for c in g.columns if c.startswith('sep_')},
                                  counterexamples=list(counter.index),
                                  d_rx_by_rx=g.d_rx.round(3).to_dict(), d_class_by_rx=g.d_class.round(3).to_dict())
    pair = PP.groupby(['rx_key', 'method', 'pair']).rate.mean().unstack('method')
    pair = pair[[m for m in METHODS if m in pair.columns]]
    pair.to_csv(out / 'pair_confusion_by_method.csv')
    tx = PT.groupby(['rx_key', 'method', 'tx']).recall.mean().unstack('method')
    tx['swad_minus_ce'] = tx['swad'] - tx['ce']
    tx.to_csv(out / 'per_tx_recall_by_method.csv')
    pl_tx = tx.loc['shen_pluto_1_train']
    S['pluto_per_tx_swad_minus_ce'] = pl_tx['swad_minus_ce'].round(4).to_dict()
    S['pluto_pairs'] = pair.loc['shen_pluto_1_train'].round(4).to_dict('index')
    # concentration: how much of SWAD's Pluto loss sits in the worst three TX
    loss = (-pl_tx['swad_minus_ce']).clip(lower=0)
    S['pluto_swad_loss_top3_share'] = float(loss.sort_values(ascending=False)[:3].sum() / max(loss.sum(), 1e-12))
    S['per_rx_accuracy'] = acc.round(4).reset_index().to_dict('records')
    S['band_energies'] = E.round(5).to_dict('records')
    (out / 'summary.json').write_text(json.dumps(C.jsonable(S), indent=1))
    pd.set_option('display.width', 220)
    print(acc.round(4).to_string())
    print(E[E.rep == 'map'].round(4).to_string())
    print(Cr[(Cr.rep == 'map')].round(3).to_string())
    for k, v in S.items():
        if '/' in k and k.endswith('map'):
            print(k, {kk: vv for kk, vv in v.items() if kk not in ('d_rx_by_rx', 'd_class_by_rx')})
    print('pluto per-TX SWAD-CE:', S['pluto_per_tx_swad_minus_ce'], 'top3 share', round(S['pluto_swad_loss_top3_share'], 3))
    print('pluto pairs:', S['pluto_pairs'])


if __name__ == '__main__':
    main()
