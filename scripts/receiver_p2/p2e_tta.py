#!/usr/bin/env python3
"""P2E-T: label-free receiver calibration at deployment (a different setting from DG: the new receiver
supplies N unlabeled packets before identification starts). Inference only.

Per formal checkpoint and outer RX q, on pre-head features h with the frozen head:
  raw        no calibration
  ca_all     h - mean_q(h over all q packets) + mean_S(h over source fitting packets)
  ca_N       same with N random unlabeled q packets (N in 20, 50, 200; 10 draws each)
  ca_imb50   N=50 packets drawn with a Dirichlet(1) class prior (prior shift); 10 draws
Test accuracy always on all q packets.

    python3 p2e_tta.py --formal outputs/receiver_p2e/formal --out reports/receiver_p2e/tta
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'receiver_p0')]
import p0_common as C   # noqa: E402
from audit_p2 import forward   # noqa: E402

C.add_root()


def main():
    import torch
    from b1_baselines.model import FeatureCNN
    from receiver_r01.data import shen as load_shen
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = C.registration()
    device = torch.device('cuda:0')
    rows, cache = [], {}
    for d in sorted(Path(a.formal).glob('shen_rank16_CV*_formal_seed*')):
        if not (d / 'status.json').exists():
            continue
        obj = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)
        j = obj['job']
        if j.get('epochs', 20) != a.epochs:
            continue
        f = reg['shen_folds'][j['fold']]
        key = (j['fold'], round(obj['scaler']['mean'], 12))
        if key not in cache:
            cache = {key: (load_shen(reg, f['inner_train'] + f['inner_validation'], obj['scaler'], 'train')[0],
                           load_shen(reg, f['outer_evaluation'], obj['scaler'], 'all')[0])}
        src, tg = cache[key]
        model = FeatureCNN()
        model.load_state_dict(obj['model'])
        model = model.to(device).eval()
        W = model.classifier.weight.detach().double().cpu().numpy()
        b = model.classifier.bias.detach().double().cpu().numpy()
        mS = forward(model, src['x'], device)[0].mean(0)
        ht, _ = forward(model, tg['x'], device)
        rng = np.random.default_rng(j['seed'])
        for r in np.unique(tg['rx']):
            m = tg['rx'] == r
            h, y = ht[m], tg['y'][m]
            acc = lambda mu: float((((h - mu + mS) @ W.T + b).argmax(1) == y).mean())
            row = dict(fold=j['fold'], arm=j['arm'], lam=j['lam'], seed=j['seed'], rx=reg['shen_allowed'][int(r)],
                       raw=float(((h @ W.T + b).argmax(1) == y).mean()), ca_all=acc(h.mean(0)))
            for n in (20, 50, 200):
                row[f'ca_{n}'] = float(np.mean([acc(h[rng.choice(len(h), n, replace=False)].mean(0)) for _ in range(10)]))
            imb = []
            for _ in range(10):
                p = rng.dirichlet(np.ones(10))
                cls = rng.choice(10, 50, p=p)
                ids = [rng.choice(np.flatnonzero(y == c)) for c in cls]
                imb.append(acc(h[ids].mean(0)))
            row['ca_imb50'] = float(np.mean(imb))
            rows.append(row)
        print(d.name, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / 'tta_per_rx.csv', index=False)
    cols = ['raw', 'ca_20', 'ca_50', 'ca_200', 'ca_all', 'ca_imb50']
    S = R.groupby(['fold', 'arm', 'seed'])[cols].mean().groupby(['fold', 'arm']).mean() * 100
    S.round(2).to_csv(out / 'tta_summary.csv')
    print(S.round(2).to_string())


if __name__ == '__main__':
    main()
