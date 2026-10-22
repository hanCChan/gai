#!/usr/bin/env python3
"""P2E driver scripts for analyses that were first run inline (audit A7), so every REPORT number is
reproducible from a committed script.

  predictors  source-only statistics (audit/models.csv) vs held/outer accuracy: Spearman within fold,
              partial Spearman given in-domain val acc, family-demeaned Spearman with a within-family
              permutation p (2000 perms). Writes audit/source_predictors.csv.
  input       Shen input statistics per RX (within/between-class CIS variance, in-band SNR estimate),
              TX-independent offset share, and outer-RX offset coverage by the source offset span with
              two controls: a random subspace of the full input space and (fair) a random subspace of
              equal dimension inside the span of the top-32 source packet principal components.
              Writes input/shen_input.csv, input/shen_offset_coverage.csv.

    python3 p2e_analyses.py predictors|input --rep reports/receiver_p2e
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REG = Path('/workspace/rl1/gai_retrain/configs/receiver_r01/registration.json')


def rank(s):
    return pd.Series(np.asarray(s)).rank().values


def sp(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def partial(x, y, z):
    rx_, ry, rz = rank(x), rank(y), rank(z)
    res = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)
    return float(np.corrcoef(res(rx_), res(ry))[0, 1])


def within(d, q, tgt, rng, n_perm=2000):
    a = d[q] - d.groupby('family')[q].transform('mean')
    b = d[tgt] - d.groupby('family')[tgt].transform('mean')
    obs, null = sp(a, b), []
    groups = d.groupby('family').groups
    for _ in range(n_perm):
        bb = b.copy()
        for ix in groups.values():
            bb.loc[ix] = rng.permutation(b.loc[ix].values)
        null.append(sp(a, bb))
    return obs, float((np.abs(null) >= abs(obs)).mean())


def predictors(rep):
    M = pd.read_csv(rep / 'audit' / 'models.csv')
    M['family'] = M.arm.str.replace('_ls', '').str.replace('_rand', '').str.replace('_soft', '')
    rng = np.random.default_rng(0)
    preds = ['src_val_acc', 'src_val_worst', 'src_ce_var', 'kappa', 'src_q10', 'loro_cov']
    rows = []
    for stage, tgts in (('search', ('held_acc', 'outer_acc')), ('formal', ('outer_acc',))):
        for fold in (0, 1):
            d = M[(M.stage == stage) & (M.fold == fold)].reset_index(drop=True)
            for tgt in tgts:
                qs = preds + (['held_acc'] if stage == 'search' and tgt == 'outer_acc' else [])
                for q in qs:
                    w, p = within(d, q, tgt, rng)
                    rows.append(dict(stage=stage, fold=fold, target=tgt, pred=q, n=len(d), n_fam=d.family.nunique(),
                                     spearman=sp(d[q], d[tgt]),
                                     partial_srcacc=partial(d[q], d[tgt], d.src_val_acc) if q != 'src_val_acc' else np.nan,
                                     within_family=w, perm_p=p))
    P = pd.DataFrame(rows)
    P.to_csv(rep / 'audit' / 'source_predictors.csv', index=False)
    print(P.round(3).to_string(index=False))


def input_stats(rep):
    import sys
    sys.path[:0] = [str(Path(__file__).resolve().parent)]
    import rx_physics as P
    reg = json.loads(REG.read_text())
    root = Path(reg['shen_cache'])
    out = rep / 'input'
    out.mkdir(parents=True, exist_ok=True)
    X, Y, rows = {}, {}, []
    for k in reg['shen_allowed']:
        x = np.load(root / k / 'strict_cis.npy').astype(np.float64)[:, 0]
        y = np.load(root / k / 'labels.npy')
        X[k], Y[k] = x, y
        mu = np.stack([x[y == t].mean(0) for t in range(10)])
        snr = P.snr_db(P.to_complex(np.array(np.load(root / k / 'iq.npy', mmap_mode='r')[::4]))).numpy()
        rows.append(dict(rx=k, within=float(np.mean([x[y == t].var(0).mean() for t in range(10)])),
                         between=float(mu.var(0).mean()), snr_med_db=float(np.median(snr))))
    g = np.mean([np.stack([X[k][Y[k] == t].mean(0) for t in range(10)]) for k in X], 0)
    for r in rows:
        D = np.stack([X[r['rx']][Y[r['rx']] == t].mean(0) for t in range(10)]) - g
        com = D.mean(0)
        r['tx_independent_share'] = float((com ** 2).sum() * 10 / (D ** 2).sum())
    pd.DataFrame(rows).to_csv(out / 'shen_input.csv', index=False)
    cov = []
    rng = np.random.default_rng(0)
    for fold in (0, 1):
        f = reg['shen_folds'][fold]
        src = f['inner_train'] + f['inner_validation']
        C = {k: np.stack([X[k][Y[k] == t].mean(0) for t in range(10)]).mean(0).ravel() for k in X}
        S = np.stack([C[k] for k in src])
        m = S.mean(0)
        U = np.linalg.svd(S - m, full_matrices=False)[2]
        U = U[: len(src) - 1]
        pk = np.concatenate([X[k].reshape(len(X[k]), -1)[::10] for k in src])
        Vp = np.linalg.svd(pk - pk.mean(0), full_matrices=False)[2][:32]
        for k in f['outer_evaluation']:
            d = C[k] - m
            n2 = (d ** 2).sum()
            full = np.mean([((d @ np.linalg.qr(rng.standard_normal((len(d), len(U))))[0]) ** 2).sum() / n2
                            for _ in range(20)])
            fair = np.mean([((d @ (Vp.T @ np.linalg.qr(rng.standard_normal((32, len(U))))[0])) ** 2).sum() / n2
                            for _ in range(20)])
            cov.append(dict(fold=fold, rx=k, coverage=float(((d @ U.T) ** 2).sum() / n2), random_full=float(full),
                            random_in_source_pca=float(fair),
                            mag_over_src_rms=float(np.sqrt(n2 / ((S - m) ** 2).sum(1).mean()))))
    C_ = pd.DataFrame(cov)
    C_.to_csv(out / 'shen_offset_coverage.csv', index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(C_.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['predictors', 'input'])
    ap.add_argument('--rep', default='/workspace/rl1/gai/reports/receiver_p2e')
    a = ap.parse_args()
    (predictors if a.what == 'predictors' else input_stats)(Path(a.rep))


if __name__ == '__main__':
    main()
