#!/usr/bin/env python3
"""Outer-RX evaluation of P2E formal checkpoints (train_p2e.py). Per run: receiver-equal accuracy,
macro-F1, worst RX, Pluto, and the two pre-registered receiver groups defined from the INPUT data only
(per-RX within-class CIS variance, reports/receiver_p2e/REPORT.md §2): noisy = rtl_1, rtl_2, rtl_3,
rtl_5 (within-class var > 0.010); clean = the rest.

    python3 eval_p2e.py --formal outputs/receiver_p2e/formal --out reports/receiver_p2e/eval
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'receiver_p0')]
import p0_common as C   # noqa: E402

C.add_root()
NOISY = {'shen_rtl_1_train', 'shen_rtl_2_train', 'shen_rtl_3_train', 'shen_rtl_5_train'}


def main():
    import torch
    from b1_baselines.model import FeatureCNN
    from receiver_r01.core import metrics
    from receiver_r01.data import shen as load_shen
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = C.registration()
    device = torch.device('cuda:0')
    old = pd.read_csv(out / 'runs.csv') if (out / 'runs.csv').exists() else None
    done = set(old.run_id) if old is not None else set()
    R, RX = ([] if old is None else old.to_dict('records')), \
            ([] if old is None or not (out / 'per_rx.csv').exists() else pd.read_csv(out / 'per_rx.csv').to_dict('records'))
    cache = {}
    for d in sorted(Path(a.formal).glob('shen_rank16_CV*_formal_seed*')):
        if not (d / 'status.json').exists() or d.name in done:
            continue
        obj = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)
        j = obj['job']
        f = reg['shen_folds'][j['fold']]
        key = (j['fold'], round(obj['scaler']['mean'], 12), round(obj['scaler']['std'], 12))
        if key not in cache:
            cache = {key: load_shen(reg, f['outer_evaluation'], obj['scaler'], 'all')[0]}
        data = cache[key]
        model = FeatureCNN()
        model.load_state_dict(obj['model'])
        model = model.to(device).eval()
        logits = C.run_batches(model, data['x'], device)
        ms = {}
        for r in np.unique(data['rx']):
            m = data['rx'] == r
            k = reg['shen_allowed'][int(r)]
            ms[k] = metrics(logits[m], torch.as_tensor(data['y'][m]))
            RX.append(dict(run_id=d.name, fold=j['fold'], arm=j['arm'], lam=j['lam'], seed=j['seed'],
                           epochs=j['epochs'], rx=k, accuracy=ms[k]['accuracy'], macro_f1=ms[k]['macro_f1']))
        accs = {k: v['accuracy'] for k, v in ms.items()}
        R.append(dict(run_id=d.name, fold=j['fold'], arm=j['arm'], lam=j['lam'], seed=j['seed'], epochs=j['epochs'],
                      acc=float(np.mean(list(accs.values()))),
                      macro_f1=float(np.mean([v['macro_f1'] for v in ms.values()])),
                      worst_rx=float(min(accs.values())), pluto=accs.get('shen_pluto_1_train', np.nan),
                      noisy=float(np.mean([v for k, v in accs.items() if k in NOISY])),
                      clean=float(np.mean([v for k, v in accs.items() if k not in NOISY]))))
        print(d.name, 'acc=%.4f' % R[-1]['acc'], flush=True)
    pd.DataFrame(R).to_csv(out / 'runs.csv', index=False)
    pd.DataFrame(RX).to_csv(out / 'per_rx.csv', index=False)


if __name__ == '__main__':
    main()
