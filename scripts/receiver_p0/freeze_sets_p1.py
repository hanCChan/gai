#!/usr/bin/env python3
"""Freeze P1 pair sets and margin-loss scale constants. Source data only; no outer RX read.

Per fold:
  Rq        = P0 frozen set (pairs.json; CE source_train, all 5 source RX), |P|=k
  Rq_inner  = same rule on CE source_train margins of the 4 inner_train RX only (the inner
              validation RX used for lambda selection is excluded from the set definition)
  Rconf     = top-k ordered pairs by CE source confusion R_conf (5 source RX)
  rand      = k ordered pairs drawn uniformly (rng seed 20261021 + fold)
Calibration (gamma, tau_m) for the normalized margin m = (l_t-l_u)/||w_t-w_u||, from the 3 frozen
CE checkpoints on source_train packets with y=t of the Rq pairs:
  gamma = median(m), tau_m = IQR(m)/1.349; 'formal' uses 5 source RX, 'inner' the 4 inner RX.

    python3 freeze_sets_p1.py --p0a DIR --pairs pairs.json --out sets_p1.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import p0_common as C   # noqa: E402

C.add_root()


def rule(src, k, neg=0.05):
    agg = src.groupby(['run_id', 't', 'u']).agg(q=('q10', 'median'), n=('frac_neg', 'max')).reset_index()
    s = agg.groupby(['t', 'u']).agg(R_q=('q', 'mean'), R_neg=('n', 'mean')).reset_index()
    s['R_q'] = -s.R_q
    top = set(map(tuple, s.sort_values('R_q', ascending=False).head(k)[['t', 'u']].values))
    return sorted(top | set(map(tuple, s[s.R_neg >= neg][['t', 'u']].values)))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--p0a', required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    reg = C.registration()
    keys = reg['shen_allowed']
    p0 = json.loads(Path(a.pairs).read_text())
    pairs = pd.concat([pd.read_csv(f) for f in sorted(Path(a.p0a).glob('pairs_shard*.csv'))], ignore_index=True)
    src = pairs[(pairs.pool == 'source_train') & (pairs.method == 'ce')]
    device = torch.device('cuda:0')
    res = dict(rule='see docstring', folds={})
    for fold in (0, 1):
        f = reg['shen_folds'][fold]
        Rq = [tuple(p) for p in p0['folds'][str(fold)]['pairs']]
        k = len(Rq)
        d = src[src.fold == fold]
        inner_idx = [keys.index(x) for x in f['inner_train']]
        Rq_inner = rule(d[d.rx.isin(inner_idx)], p0['k'], p0['neg_thresh'])
        rc = d.groupby(['t', 'u']).conf.mean().reset_index().sort_values(['conf', 't', 'u'], ascending=[False, True, True])
        Rconf = sorted(map(tuple, rc.head(k)[['t', 'u']].values))
        all_pairs = [(t, u) for t in range(10) for u in range(10) if t != u]
        rng = np.random.default_rng(20261021 + fold)
        rand = sorted(all_pairs[i] for i in rng.choice(len(all_pairs), k, replace=False))
        calib = {}
        for tag, rx_keys, Pset in (('formal', f['inner_train'] + f['inner_validation'], Rq), ('inner', f['inner_train'], Rq_inner)):
            ms = []
            for j in C.jobs('shen'):
                if j['fold'] != fold or j['method'] != 'ce':
                    continue
                obj, _ = C.load_model(j['ckpt'])
                model, scaler = C.build_model(j, obj)
                model = model.to(device)
                from receiver_r01.data import shen as load_shen
                data, _ = load_shen(reg, rx_keys, scaler, 'train')
                lg = C.run_batches(model, data['x'], device)
                W = model.classifier.weight.detach().cpu().numpy()
                for (t, u) in Pset:
                    m = data['y'] == t
                    ms.append((lg[m, t] - lg[m, u]) / (np.linalg.norm(W[t] - W[u]) + 1e-6))
            ms = np.concatenate(ms)
            q25, q50, q75 = np.quantile(ms, [.25, .5, .75])
            calib[tag] = dict(gamma=float(q50), tau_m=float((q75 - q25) / 1.349), n=int(len(ms)))
        res['folds'][str(fold)] = dict(Rq=[list(map(int, p)) for p in Rq], Rq_inner=[list(map(int, p)) for p in Rq_inner],
                                       Rconf=[list(map(int, p)) for p in Rconf], rand=[list(map(int, p)) for p in rand],
                                       calib=calib)
        print(fold, json.dumps(res['folds'][str(fold)]))
    C.dump_json(a.out, res)


if __name__ == '__main__':
    main()
