#!/usr/bin/env python3
"""P2E auto-chain: receiver-calibrated counterfactual training, with source-only nested selection and
pre-registered gates. Every rule below is fixed before any P2E fit; stages are resumable (completed
runs are skipped, partial run directories are removed and rerun).

E1  loro     every arm x lam grid x 5 held source RX x 2 folds, seed 20261016
E1s select   lam* per (fold, arm) = min mean held-RX CE over the 5 LORO folds (tie -> smaller lam);
             G1 (source-only gate) = arm's mean held-RX acc at lam* > ccalign's in BOTH folds
E2  formal   every arm at lam*, 2 folds x 5 seeds (20261013, 14, 15, 17, 18); anchors retrained
E3  eval     eval_p2e.py (outer RX: dev data, read only here)
E4  decide   "X beats Y" = seed-mean d(acc) > 0 in both folds AND >= 8/10 paired (fold, seed) d > 0
             GO_METHOD   some new arm beats ccalign, passes G1, and beats its own control
                         (noise, ccanoise -> noiseiso; sig -> sigrand)
             GENERIC     some new arm beats ccalign but is not GO (fails G1 or its control; audit B7)
             STOP        no new arm beats ccalign
E5  horizon  (after GO_METHOD or GENERIC) best new arm, ccalign, ce at 40 epochs, 5 seeds, same rule
Then the chain stops: WiSig transfer needs its own trainer and is not part of this chain.

--round e6 (registered after E4 = STOP, REPORT §5): hypothesis = the clean-RX cost of calibrated noise
comes from training CE on noisy inputs, not from noise invariance. Arms ncons / ccancons (CE on clean
packets + KL consistency of a calibrated-noise view), control nconsiso; E1 arms and anchors reused.
E6 decision: GO_METHOD iff ncons or ccancons beats ccalign AND beats nconsiso on acc (8/10 rule) AND
its clean-group seed-mean d vs ccalign >= -0.5 pp in both folds; else STOP_NOISE_LINE.

    python3 run_p2e_chain.py [--round e6] [--workers 8] [--dry-run]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GAI = HERE.parent.parent
OUT = GAI / 'outputs' / 'receiver_p2e'
REP = GAI / 'reports' / 'receiver_p2e'
SEARCH_SEED = 20261016
SEEDS = (20261013, 20261014, 20261015, 20261017, 20261018)
GRID = {'ce': (0.,), 'ls': (0.,), 'ccalign': (0.3, 1., 3.), 'noise': (1., 2., 4.), 'noiseiso': (1., 2., 4.),
        'ccanoise': (1., 2., 4.), 'sig': (1., 2., 4.), 'sigrand': (1., 2., 4.)}
NEW = ('noise', 'ccanoise', 'sig')
CONTROL = {'noise': 'noiseiso', 'ccanoise': 'noiseiso', 'sig': 'sigrand'}
GRID_E6 = {'ncons': (0.3, 1., 3.), 'nconsiso': (0.3, 1., 3.), 'ccancons': (0.3, 1., 3.)}
NEW_E6 = ('ncons', 'ccancons')
CONTROL_E6 = {'ncons': 'nconsiso', 'ccancons': 'nconsiso'}

sys.path[:0] = [str(HERE)]
from train_p2e import run_id   # noqa: E402


def queue(jobs, out, workers, dry):
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for j in jobs:
        d = out / run_id(argparse.Namespace(**j))
        if (d / 'status.json').exists():
            continue
        if d.exists():
            shutil.rmtree(d)
        todo.append(j)
    print(f'[queue] {out.name}: {len(jobs)} jobs, {len(todo)} to run', flush=True)
    if dry or not todo:
        return
    lock, failed = threading.Lock(), []

    def worker(k):
        while True:
            with lock:
                if not todo:
                    return
                j = todo.pop(0)
            args = [f'--{k_.replace("_", "-")}={v}' for k_, v in j.items()]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % 4), CUBLAS_WORKSPACE_CONFIG=':4096:8')
            r = subprocess.run([sys.executable, str(HERE / 'train_p2e.py'), *args, f'--out={out}'],
                               env=env, capture_output=True, text=True)
            msg = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-400:]
            with lock:
                print(('OK   ' if r.returncode == 0 else 'FAIL ') + ' '.join(args) + ' :: ' + msg[:300], flush=True)
                if r.returncode:
                    failed.append(j)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(workers)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    if failed:
        raise SystemExit(f'{len(failed)} jobs failed in {out.name}')


def status_table(out):
    import torch
    rows = []
    for st in sorted(out.glob('*/status.json')):
        s = json.loads(st.read_text())
        j = torch.load(st.parent / 'selected.pt', map_location='cpu', weights_only=False)['job']
        rows.append(dict(fold=j['fold'], arm=j['arm'], lam=j['lam'], held=j['held'], seed=j['seed'],
                         epochs=j['epochs'], **{k: s[k] for k in ('best_val_CE', 'best_val_acc', 'best_epoch')},
                         held_CE=s.get('held_CE'), held_acc=s.get('held_acc'), held_rx=s.get('held_rx')))
    return pd.DataFrame(rows)


def select(T, grid=GRID):
    T = T[[a in grid and any(abs(l - g) < 1e-9 for g in grid[a]) for a, l in zip(T.arm, T.lam)]]
    G = T.groupby(['fold', 'arm', 'lam'])[['held_CE', 'held_acc']].mean().reset_index()
    sel = {}
    for (fold, arm), d in G.groupby(['fold', 'arm']):
        b = d.sort_values(['held_CE', 'lam']).iloc[0]
        sel[f'{fold}_{arm}'] = dict(lam=float(b.lam), held_CE=float(b.held_CE), held_acc=float(b.held_acc))
    for fold in (0, 1):
        ref = sel[f'{fold}_ccalign']['held_acc']
        for arm in grid:
            sel[f'{fold}_{arm}']['beats_ccalign_loro'] = bool(sel[f'{fold}_{arm}']['held_acc'] > ref)
    g1 = {arm: all(sel[f'{f}_{arm}']['beats_ccalign_loro'] for f in (0, 1)) for arm in grid}
    return sel, g1, G


def beats(R, x, y, metric='acc'):
    assert not R.duplicated(['fold', 'seed', 'arm']).any(), 'duplicate (fold, seed, arm) rows'
    P = R.pivot_table(index=['fold', 'seed'], columns='arm', values=metric)
    if x not in P or y not in P:
        return dict(x=x, y=y, beats=False, missing=True)
    d = (P[x] - P[y]).dropna()
    fm = d.groupby(level='fold').mean()
    n = len(d)
    return dict(x=x, y=y, metric=metric, beats=bool(len(fm) == 2 and (fm > 0).all() and (d > 0).sum() >= int(np.ceil(.8 * n))),
                n_pos=int((d > 0).sum()), n=n, **{f'd_fold{k}_pp': round(float(v) * 100, 2) for k, v in fm.items()})


def decide(R, g1):
    T = []
    for x in NEW + ('noiseiso', 'sigrand', 'ls', 'ce'):
        for y in ('ccalign', 'ce', 'ls', CONTROL.get(x)):
            if y and y != x:
                for met in ('acc', 'noisy', 'clean', 'worst_rx'):
                    T.append(beats(R, x, y, met))
    T = pd.DataFrame(T)
    B = lambda x, y, m='acc': bool(T[(T.x == x) & (T.y == y) & (T.metric == m)].beats.any())
    go = [x for x in NEW if B(x, 'ccalign') and g1[x] and B(x, CONTROL[x])]
    gen = [x for x in NEW if B(x, 'ccalign') and x not in go]
    outcome = 'GO_METHOD' if go else 'GENERIC' if gen else 'STOP'
    best = None
    if go or gen:
        S = R[R.arm.isin(go or gen)].groupby('arm').acc.mean()
        best = str(S.idxmax())
    return outcome, best, T


def round_e6(a):
    jobs = [dict(stage='loro', held=h, fold=f, arm=arm, lam=lam, seed=SEARCH_SEED, epochs=20)
            for f in (0, 1) for h in range(5) for arm, gr in GRID_E6.items() for lam in gr]
    queue(jobs, OUT / 'loro', a.workers, a.dry_run)
    if a.dry_run:
        return
    T = status_table(OUT / 'loro')
    T.to_csv(REP / 'loro_runs.csv', index=False)
    sel, g1, G = select(T, {**GRID, **GRID_E6})
    G.to_csv(REP / 'loro_grid.csv', index=False)
    (REP / 'lambda_p2e_e6.json').write_text(json.dumps(dict(sel=sel, G1=g1), indent=1))
    print('[E6-loro] G1:', {k: g1[k] for k in GRID_E6}, flush=True)
    jobs = [dict(stage='formal', held=-1, fold=f, arm=arm, lam=sel[f'{f}_{arm}']['lam'], seed=s, epochs=20)
            for f in (0, 1) for arm in GRID_E6 for s in SEEDS]
    queue(jobs, OUT / 'formal', a.workers, False)
    subprocess.run([sys.executable, str(HERE / 'eval_p2e.py'), '--formal', str(OUT / 'formal'),
                    '--out', str(REP / 'eval')], check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
    R = pd.read_csv(REP / 'eval' / 'runs.csv')
    R = R[R.epochs == 20]
    PT = pd.DataFrame([beats(R, x, y, m) for x in NEW_E6 + ('nconsiso',)
                       for y in ('ccalign', 'ce', 'noise', 'ccanoise', CONTROL_E6.get(x)) if y and y != x
                       for m in ('acc', 'noisy', 'clean', 'worst_rx')])
    PT.to_csv(REP / 'eval' / 'pairwise_e6.csv', index=False)
    B = lambda x, y: bool(PT[(PT.x == x) & (PT.y == y) & (PT.metric == 'acc')].beats.any())
    P = R.pivot_table(index=['fold', 'seed'], columns='arm', values='clean')
    clean_ok = {x: bool(((P[x] - P['ccalign']).groupby(level='fold').mean() >= -0.005).all()) for x in NEW_E6}
    go = [x for x in NEW_E6 if B(x, 'ccalign') and B(x, CONTROL_E6[x]) and clean_ok[x]]
    summ = R.groupby(['fold', 'arm'])[['acc', 'noisy', 'clean', 'worst_rx', 'pluto', 'macro_f1']].mean()
    summ.round(4).to_csv(REP / 'eval' / 'summary.csv')
    dec = dict(outcome='GO_METHOD' if go else 'STOP_NOISE_LINE', go_arms=go, clean_ok=clean_ok,
               G1={k: g1[k] for k in GRID_E6})
    (REP / 'decision_e6.json').write_text(json.dumps(dec, indent=1))
    print((summ * 100).round(2).to_string(), flush=True)
    print('[E6]', dec, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--round', default='e1', choices=['e1', 'e6'])
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    REP.mkdir(parents=True, exist_ok=True)
    if a.round == 'e6':
        return round_e6(a)
    # E1
    jobs = [dict(stage='loro', held=h, fold=f, arm=arm, lam=lam, seed=SEARCH_SEED, epochs=20)
            for f in (0, 1) for h in range(5) for arm, gr in GRID.items() for lam in gr]
    queue(jobs, OUT / 'loro', a.workers, a.dry_run)
    if a.dry_run:
        return
    T = status_table(OUT / 'loro')
    T.to_csv(REP / 'loro_runs.csv', index=False)
    sel, g1, G = select(T)
    G.to_csv(REP / 'loro_grid.csv', index=False)
    (REP / 'lambda_p2e.json').write_text(json.dumps(dict(sel=sel, G1=g1), indent=1))
    print('[E1] G1 (beats ccalign on LORO held RX, both folds):', g1, flush=True)
    # E2
    jobs = [dict(stage='formal', held=-1, fold=f, arm=arm, lam=sel[f'{f}_{arm}']['lam'], seed=s, epochs=20)
            for f in (0, 1) for arm in GRID for s in SEEDS]
    queue(jobs, OUT / 'formal', a.workers, False)
    # E3
    subprocess.run([sys.executable, str(HERE / 'eval_p2e.py'), '--formal', str(OUT / 'formal'),
                    '--out', str(REP / 'eval')], check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
    R = pd.read_csv(REP / 'eval' / 'runs.csv')
    R = R[R.epochs == 20]
    outcome, best, PT = decide(R, g1)
    PT.to_csv(REP / 'eval' / 'pairwise.csv', index=False)
    summ = R.groupby(['fold', 'arm'])[['acc', 'noisy', 'clean', 'worst_rx', 'pluto', 'macro_f1']].mean()
    summ.round(4).to_csv(REP / 'eval' / 'summary.csv')
    dec = dict(outcome=outcome, best_arm=best, G1=g1)
    (REP / 'decision_e4.json').write_text(json.dumps(dec, indent=1))
    print((summ * 100).round(2).to_string(), flush=True)
    print('[E4]', dec, flush=True)
    if outcome == 'STOP':
        return
    # E5 horizon
    jobs = [dict(stage='formal', held=-1, fold=f, arm=arm, lam=sel[f'{f}_{arm}']['lam'], seed=s, epochs=40)
            for f in (0, 1) for arm in (best, 'ccalign', 'ce') for s in SEEDS]
    queue(jobs, OUT / 'formal', a.workers, False)
    subprocess.run([sys.executable, str(HERE / 'eval_p2e.py'), '--formal', str(OUT / 'formal'),
                    '--out', str(REP / 'eval')], check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
    R = pd.read_csv(REP / 'eval' / 'runs.csv')
    H = R[R.epochs == 40]
    ht = pd.DataFrame([beats(H, best, y, m) for y in ('ccalign', 'ce') for m in ('acc', 'noisy', 'clean')])
    ht.to_csv(REP / 'eval' / 'pairwise_ep40.csv', index=False)
    dec['horizon40'] = dict(beats_ccalign=bool(ht[(ht.y == 'ccalign') & (ht.metric == 'acc')].beats.iloc[0]),
                            summary=H.groupby(['fold', 'arm']).acc.mean().round(4).to_dict().__repr__())
    (REP / 'decision_e4.json').write_text(json.dumps(dec, indent=1))
    print('[E5]', dec, flush=True)


if __name__ == '__main__':
    main()
