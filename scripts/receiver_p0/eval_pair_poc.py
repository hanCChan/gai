#!/usr/bin/env python3
"""Evaluate PoC checkpoints on the fold's outer RX: accuracy + hard-pair margins/confusions.

Reads ONLY outer RX labels for evaluation after all checkpoints are fixed (same hygiene as
evaluate_receiver_r01.py). Reports, per run and aggregated over seeds:
  - outer accuracy per RX (matched against the frozen CE arm as anchor)
  - per hard pair (t,u) in P and for ALL 90 ordered pairs: conf_{t->u}, q10 margin
  - headline pre-registered comparisons (CE anchor vs each branch x lambda):
      mean outer acc, Pluto-fold acc, mean conf over P (both directions),
      worst-pair conf, count of pairs with conf > 0.3 (collapse count).
No model selection happens here; every trained checkpoint is reported.

    python3 eval_pair_poc.py --poc-out DIR --pairs pairs.json --out DIR
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import p0_common as C   # noqa: E402

C.add_root()


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--poc-out', required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    reg = C.registration()
    keys = reg['shen_allowed']
    pairs_cfg = json.loads(Path(a.pairs).read_text())

    rows, pair_rows = [], []
    for ckpt_dir in sorted(Path(a.poc_out).glob('shen_rank16_CV*')):
        if not (ckpt_dir / 'selected.pt').exists():
            continue
        obj = torch.load(ckpt_dir / 'selected.pt', map_location='cpu', weights_only=False)
        fold = obj['job']['fold']
        branch = obj['job']['branch']
        lam = float(obj['job']['lambda'])
        seed = obj['job']['seed']
        from b1_baselines.model import FeatureCNN
        model = FeatureCNN()
        model.load_state_dict(obj['model'])
        model = model.to(device).eval()
        src, outer = C.shen_fold_pools(reg, fold, obj['scaler'], partition='train')
        logits = C.run_batches(model, outer['x'], device)
        st = C.margin_stats(logits, outer['y'], outer['rx'])
        acc = float((logits.argmax(1) == outer['y']).mean())
        P = [tuple(p) for p in pairs_cfg['folds'][str(fold)]['pairs']]
        conf_p = [st[(r, t, u)]['conf'] for (r, t, u) in st if (t, u) in P] if P else []
        for r in np.unique(outer['rx']):
            rows.append(dict(fold=fold, branch=branch, lam=lam, seed=seed, rx_key=keys[int(r)],
                             accuracy=float((logits[outer['rx'] == r].argmax(1) == outer['y'][outer['rx'] == r]).mean())))
        for (r, t, u), s in st.items():
            pair_rows.append(dict(fold=fold, branch=branch, lam=lam, seed=seed, rx_key=keys[r],
                                  t=t, u=u, **s, in_P=int((t, u) in P)))
        rows.append(dict(fold=fold, branch=branch, lam=lam, seed=seed, rx_key='OVERALL',
                         accuracy=acc, mean_conf_P=float(np.mean(conf_p)) if conf_p else np.nan,
                         max_conf_P=float(np.max(conf_p)) if conf_p else np.nan,
                         n_collapsed_P=int(sum(c > 0.3 for c in conf_p)) if conf_p else 0))
        print(f'eval {ckpt_dir.name}: acc={acc:.4f}', flush=True)
        del model
    pd.DataFrame(rows).to_csv(out / 'runs.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(out / 'pairs.csv', index=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
