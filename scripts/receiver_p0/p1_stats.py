#!/usr/bin/env python3
"""P1 re-analysis of P0-C (CPU; reads P0-A CSVs, writes nothing over P0 outputs).

Fixes vs run_p0c.py:
  - risk aggregation reported separately for CE-only (the arm select_pairs.py uses) and for the
    5-method mean that P0-C used; outer failure always the CE arm in the CE-only analysis.
  - R_q and R_conf compared on equal footing (same models, same packets).
  - significance by a TX-relabel (Mantel-type) permutation: O'[t,u] = O[s(t), s(u)] for random
    permutations s of the 10 TX, which keeps the shared-TX dependence of the 90 ordered pairs.
  - receiver-level analysis: each of the 10 outer RX (outer once across the two folds) gets its
    own Spearman(source risk, that RX's confusion); report per-RX values and a sign count of
    R_q minus R_conf.

    python3 p1_stats.py --p0a DIR --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N = 10
SEED = 20261021


def mat(df, col):
    M = np.full((N, N), np.nan)
    for t, u, v in df[['t', 'u', col]].itertuples(index=False):
        M[int(t), int(u)] = v
    return M


def offdiag(M):
    return M[~np.eye(N, dtype=bool)]


def mantel(S, O, n=20000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    rho = stats.spearmanr(offdiag(S), offdiag(O)).statistic
    cnt = 0
    for _ in range(n):
        s = rng.permutation(N)
        if stats.spearmanr(offdiag(S), offdiag(O[np.ix_(s, s)])).statistic >= rho:
            cnt += 1
    return float(rho), (cnt + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p0a', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.concat([pd.read_csv(f) for f in sorted(Path(a.p0a).glob('pairs_shard*.csv'))], ignore_index=True)

    rows, rx_rows = [], []
    for agg in ('ce_only', 'all5'):
        src = pairs[pairs.pool == 'source_train']
        outp = pairs[pairs.pool == 'outer']
        if agg == 'ce_only':
            src, outp = src[src.method == 'ce'], outp[outp.method == 'ce']
        per_run = src.groupby(['fold', 'run_id', 't', 'u']).agg(q=('q10', 'median'), c=('conf', 'mean')).reset_index()
        risk = per_run.groupby(['fold', 't', 'u']).agg(R_q=('q', 'mean'), R_conf=('c', 'mean')).reset_index()
        risk['R_q'] = -risk.R_q
        for fold in (0, 1):
            rk = risk[risk.fold == fold]
            o = outp[outp.fold == fold]
            O = mat(o.groupby(['t', 'u']).conf.mean().reset_index(), 'conf')
            for name in ('R_q', 'R_conf'):
                S = mat(rk, name)
                rho, p = mantel(S, O)
                top_s = set(np.argsort(-offdiag(S))[:10]); top_o = set(np.argsort(-offdiag(O))[:10])
                rows.append(dict(agg=agg, fold=fold, risk=name, spearman=rho, p_tx_perm=p,
                                 top10_recall=len(top_s & top_o) / 10))
            for rxk in sorted(o.rx_key.unique()):
                Orx = mat(o[o.rx_key == rxk].groupby(['t', 'u']).conf.mean().reset_index(), 'conf')
                rq = stats.spearmanr(offdiag(mat(rk, 'R_q')), offdiag(Orx)).statistic
                rc = stats.spearmanr(offdiag(mat(rk, 'R_conf')), offdiag(Orx)).statistic
                rx_rows.append(dict(agg=agg, fold=fold, rx_key=rxk, rho_R_q=rq, rho_R_conf=rc, diff=rq - rc))
    R = pd.DataFrame(rows)
    X = pd.DataFrame(rx_rows)
    R.to_csv(out / 'risk_fair.csv', index=False)
    X.to_csv(out / 'risk_per_rx.csv', index=False)
    summ = {}
    for agg in ('ce_only', 'all5'):
        x = X[X['agg'] == agg]
        summ[agg] = dict(n_rx=len(x), median_rho_R_q=float(x.rho_R_q.median()), median_rho_R_conf=float(x.rho_R_conf.median()),
                         n_rx_R_q_better=int((x['diff'] > 0).sum()),
                         sign_test_p=float(stats.binomtest(int((x['diff'] > 0).sum()), len(x)).pvalue))
    (out / 'risk_fair_summary.json').write_text(json.dumps(summ, indent=1))
    print(R.to_string(index=False))
    print(X.round(3).to_string(index=False))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
