#!/usr/bin/env python3
"""Freeze the hard-pair set P per fold from P0-A source margins, for PoC training.

Frozen rule (docs/receiver_p0/PROTOCOL_P0.md):
  For each fold, take the CE-arm models (3 seeds), compute over the fold's source_train
  margins, per ordered pair (t,u):
      R_q(t,u) = - mean_seed median_r q10_{r,t,u}        (worst-margin risk)
  Union of the top-K ordered pairs by R_q and all ordered pairs with R_neg >= 0.05
  (any source RX flipping >=5% of packets) forms P (both directions kept independently;
  a pair entering as (t,u) does NOT auto-add (u,t)).
  K = 8 (matches SWAD's 4 collapsed-pair count with both directions; sensitivity K=4/16
  in the PoC grid is decided by the grid, not by looking at outer data).

No outer-RX value is read here. Output: pairs_f{fold}.json with provenance.

    python3 select_pairs.py --p0a DIR --out DIR [--k 8]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import p0_common as C   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p0a', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--neg-thresh', type=float, default=0.05)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = pd.concat([pd.read_csv(f) for f in sorted(Path(a.p0a).glob('pairs_shard*.csv'))], ignore_index=True)
    src = pairs[(pairs.pool == 'source_train') & (pairs.method == 'ce')]
    all_p = {}
    for fold in (0, 1):
        d = src[src.fold == fold]
        # per ordered pair: seed-mean of (median over source RX of q10), seed-mean max frac_neg
        agg = d.groupby(['run_id', 't', 'u']).agg(q10_med=('q10', 'median'), neg=('frac_neg', 'max')).reset_index()
        s = agg.groupby(['t', 'u']).agg(R_q=('q10_med', 'mean'), R_neg=('neg', 'mean')).reset_index()
        s['R_q'] = -s.R_q
        topk = set(map(tuple, s.sort_values('R_q', ascending=False).head(a.k)[['t', 'u']].values))
        neg = set(map(tuple, s[s.R_neg >= a.neg_thresh][['t', 'u']].values))
        P = sorted(topk | neg)
        all_p[fold] = dict(pairs=[list(map(int, p)) for p in P],
                           topk=[list(map(int, p)) for p in sorted(topk)],
                           by_neg=[list(map(int, p)) for p in sorted(neg)],
                           R_q_table=s.sort_values('R_q', ascending=False).to_dict('records'))
        print(f'fold {fold}: |P|={len(P)} topk={sorted(topk)} by_neg={sorted(neg)}')
    C.dump_json(out / 'pairs.json', dict(k=a.k, neg_thresh=a.neg_thresh, folds=all_p,
                                          rule='CE-arm source_train margins only; R_q=seed-mean of median-source-RX q10; R_neg=seed-mean max frac_neg'))
    print('wrote', out / 'pairs.json')


if __name__ == '__main__':
    main()
