#!/usr/bin/env python3
"""P2E-3: receiver-counterfactual fidelity (no training). Does transplanting a receiver's estimated
perturbation onto clean SOURCE validation packets reproduce that receiver's real accuracy?

Perturbation model in raw Strict-CIS units (x = packet, r = its receiver, q = receiver to imitate):
  sig      x - c_r + c_q                 c = class-balanced mean CIS (TX-independent receiver signature)
  sigproj  x - c_r + m + P_U (c_q - m)   only the part of c_q inside the source signature span
  noise    x + sqrt(max(v_q - v_r, 0)) * eps   v = per-pixel within-class variance (diagonal profile)
  both     sig + noise
Source packets: positions 160..199 of every fitting RX (never trained on). Targets: outer RX (dev data,
exploratory) and, source-only, every fitting RX imitated from the others.

    python3 p2e_fidelity.py --out reports/receiver_p2e/fidelity
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
GAI = Path('/workspace/rl1/gai')


def load_raw(reg, key, part):
    root = Path(reg['shen_cache']) / key
    x = np.load(root / 'strict_cis.npy').astype(np.float64)[:, 0]
    y = np.load(root / 'labels.npy')
    sl = {'train': slice(0, 160), 'val': slice(160, 200), 'all': slice(None)}[part]
    ids = np.concatenate([np.flatnonzero(y == t)[sl] for t in range(10)])
    return x[ids], y[ids]


def stats(x, y):
    mu = np.stack([x[y == t].mean(0) for t in range(10)])
    v = np.mean([x[y == t].var(0) for t in range(10)], 0)
    return mu.mean(0), v


def main():
    import torch
    from b1_baselines.model import FeatureCNN
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = C.registration()
    device = torch.device('cuda:0')
    rows = []
    for fold in (0, 1):
        f = reg['shen_folds'][fold]
        src = f['inner_train'] + f['inner_validation']
        allk = src + f['outer_evaluation']
        # receiver statistics: source RX from their fitting packets only; outer RX from all packets
        st = {k: stats(*load_raw(reg, k, 'train' if k in src else 'all')) for k in allk}
        S = np.stack([st[k][0].ravel() for k in src])
        m = S.mean(0)
        U = np.linalg.svd(S - m, full_matrices=False)[2]
        vals = {k: load_raw(reg, k, 'val') for k in src}
        tg_real = {k: load_raw(reg, k, 'all') for k in f['outer_evaluation']}
        runs = [('ce', Path(j['ckpt']) / 'selected.pt', j['seed']) for j in C.jobs('shen')
                if j['method'] == 'ce' and j['fold'] == fold]
        for arm, sub in (('ccalign', 'receiver_p2b'), ('rxnull', 'receiver_p2')):
            for d in sorted((GAI / 'outputs' / sub / 'formal').glob(f'shen_rank16_CV{fold}_{arm}[0-9]*_formal_seed*')):
                runs.append((arm, d / 'selected.pt', int(d.name.split('seed')[1])))
        for arm, path, seed in runs:
            obj = torch.load(path, map_location='cpu', weights_only=False)
            sc = obj['scaler']
            model = FeatureCNN()
            model.load_state_dict(obj['model'])
            model = model.to(device).eval()

            def acc(x, y):
                xb = torch.as_tensor(((x - sc['mean']) / sc['std'])[:, None], dtype=torch.float32, device=device)
                with torch.no_grad():
                    p = torch.cat([model(xb[i:i + 512]).argmax(1) for i in range(0, len(xb), 512)]).cpu().numpy()
                return float((p == y).mean())

            rng = np.random.default_rng(seed)

            def imitate(q, cq, vq):
                res = {}
                for kind in ('sig', 'sigproj', 'noise', 'both'):
                    xs, ys = [], []
                    for r in src:
                        if r == q:
                            continue
                        x, y = vals[r]
                        cr, vr = st[r]
                        z = x.copy()
                        if kind in ('sig', 'both'):
                            z = z - cr + cq
                        if kind == 'sigproj':
                            z = z - cr + (m + (cq.ravel() - m) @ U.T @ U).reshape(cr.shape)
                        if kind in ('noise', 'both'):
                            z = z + np.sqrt(np.clip(vq - vr, 0, None)) * rng.standard_normal(z.shape)
                        xs.append(z)
                        ys.append(y)
                    res[f'acc_{kind}'] = acc(np.concatenate(xs), np.concatenate(ys))
                return res

            base = acc(np.concatenate([vals[r][0] for r in src]), np.concatenate([vals[r][1] for r in src]))
            for q in f['outer_evaluation']:
                rows.append(dict(fold=fold, arm=arm, seed=seed, kind='outer', rx=q, acc_src=base,
                                 acc_real=acc(*tg_real[q]), **imitate(q, *st[q])))
            for q in src:   # source-only fidelity: imitate fitting RX q from the other fitting RX
                others = [r for r in src if r != q]
                rows.append(dict(fold=fold, arm=arm, seed=seed, kind='source', rx=q,
                                 acc_src=acc(np.concatenate([vals[r][0] for r in others]),
                                             np.concatenate([vals[r][1] for r in others])),
                                 acc_real=acc(*vals[q]), **imitate(q, *st[q])))
            print(fold, arm, seed, 'done', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / 'fidelity.csv', index=False)
    G = R.groupby(['kind', 'fold', 'rx', 'arm']).mean(numeric_only=True).drop(columns='seed').round(3)
    G.to_csv(out / 'fidelity_summary.csv')
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(G.to_string())


if __name__ == '__main__':
    main()
