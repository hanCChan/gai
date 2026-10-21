#!/usr/bin/env python3
"""Outer-RX evaluation of P1 formal checkpoints (+ frozen R0/R1 CE = bitwise P1 ce anchor).

Run only after every formal checkpoint exists (asserted against the formal job list).
Per run: RX-equal accuracy and macro-F1, worst outer RX, per RX/TX recall, all 90 ordered-pair
confusions and normalized-margin stats per RX, P-set (Rq formal set of the fold) mean confusion and
normalized margin mean/q10, non-P mean confusion.

    python3 eval_p1.py --formal DIR --jobs jobs_formal.txt --sets sets_p1.json --out DIR
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
    from b1_baselines.model import FeatureCNN
    from receiver_r01.core import metrics
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal', required=True)
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--sets', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = C.registration()
    keys = reg['shen_allowed']
    sets = json.loads(Path(a.sets).read_text())['folds']
    device = torch.device('cuda:0')

    runs = []
    n_jobs = 0
    for d in sorted(Path(a.formal).glob('shen_rank16_CV*_formal_seed*')):
        assert (d / 'status.json').exists(), f'incomplete: {d}'
        j = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)['job']
        runs.append(dict(path=d / 'selected.pt', fold=j['fold'], method=j['method'], lam=j['lam'],
                         lam2=j['lam2'] if j['method'] == 'tail_drift' else 0., set=j['set'], seed=j['seed']))
        n_jobs += 1
    expected = sum(1 for l in Path(a.jobs).read_text().splitlines() if l.strip() and not l.startswith('#'))
    assert n_jobs == expected, (n_jobs, expected)
    for j in C.jobs('shen'):
        if j['method'] == 'ce':
            C.load_model(j['ckpt'])   # hash guard
            runs.append(dict(path=Path(j['ckpt']) / 'selected.pt', fold=j['fold'], method='ce', lam=0., lam2=0., set='Rq', seed=j['seed']))

    R, RX, TX, PR = [], [], [], []
    for ru in runs:
        obj = torch.load(ru['path'], map_location='cpu', weights_only=False)
        model = FeatureCNN()
        model.load_state_dict(obj['model'])
        model = model.to(device).eval()
        _, outer = C.shen_fold_pools(reg, ru['fold'], obj['scaler'])
        lg = C.run_batches(model, outer['x'], device)
        W = model.classifier.weight.detach().cpu().numpy()
        meta = {k: ru[k] for k in ('fold', 'method', 'lam', 'lam2', 'set', 'seed')}
        P = set(map(tuple, sets[str(ru['fold'])]['Rq']))
        own = set(map(tuple, sets[str(ru['fold'])][ru['set']]))
        accs, f1s = [], []
        pconf, nconf, oconf, pm, pq = [], [], [], [], []
        for r in np.unique(outer['rx']):
            m = outer['rx'] == r
            met = metrics(lg[m], outer['y'][m])
            accs.append(met['accuracy']); f1s.append(met['macro_f1'])
            RX.append(dict(meta, rx_key=keys[r], accuracy=met['accuracy'], macro_f1=met['macro_f1']))
            for t in range(10):
                TX.append(dict(meta, rx_key=keys[r], tx=t, recall=met['recall'][t]))
            cm = np.array(met['confusion'])
            yr, lr = outer['y'][m], lg[m]
            for t in range(10):
                lt = lr[yr == t]
                for u in range(10):
                    if u == t:
                        continue
                    nm = (lt[:, t] - lt[:, u]) / (np.linalg.norm(W[t] - W[u]) + 1e-6)
                    c = cm[t, u] / max(cm[t].sum(), 1)
                    PR.append(dict(meta, rx_key=keys[r], t=t, u=u, conf=c, nm_mean=float(nm.mean()),
                                   nm_q10=float(np.quantile(nm, .1)), in_P=int((t, u) in P)))
                    (pconf if (t, u) in P else nconf).append(c)
                    if (t, u) in own:
                        oconf.append(c)
                    if (t, u) in P:
                        pm.append(nm.mean()); pq.append(np.quantile(nm, .1))
        pl = [x['accuracy'] for x in RX[-5:] if x['rx_key'] == 'shen_pluto_1_train']
        R.append(dict(meta, acc=float(np.mean(accs)), macro_f1=float(np.mean(f1s)), worst_rx_acc=float(np.min(accs)),
                      pluto_acc=pl[0] if pl else np.nan, P_conf=float(np.mean(pconf)), nonP_conf=float(np.mean(nconf)),
                      own_set_conf=float(np.mean(oconf)), P_nm_mean=float(np.mean(pm)), P_nm_q10=float(np.mean(pq)),
                      w_tu_P=float(np.mean([np.linalg.norm(W[t] - W[u]) for t, u in P]))))
        print('eval', meta, round(R[-1]['acc'], 4), flush=True)
    for name, rows in (('runs', R), ('per_rx', RX), ('per_rx_tx', TX), ('pairs', PR)):
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
