#!/usr/bin/env python3
"""Source-only λ selection (PROTOCOL_P1 §2): per (fold, method) min held-RX CE, tie -> smaller λ.
Writes lambda_p1.json and the formal job list.

    python3 select_lambda_p1.py --search DIR --out DIR
"""
import argparse
import json
from pathlib import Path

import pandas as pd

SEEDS = (20261013, 20261014, 20261015)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--search', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = []
    for st in sorted(Path(a.search).glob('*/status.json')):
        s = json.loads(st.read_text())
        n = s['run_id'].split('_')   # shen rank16 CV{f} {method}{lam} Rq search seed
        rows.append(dict(run_id=s['run_id'], fold=int(n[2][2:]), arm=n[3], val_CE=s['best_val_CE'],
                         val_acc=s['best_val_acc'], epoch=s['best_epoch']))
    df = pd.DataFrame(rows)
    df['method'] = df.arm.str.extract(r'^([a-z_]+)')[0]
    df['lam'] = df.arm.str.extract(r'([0-9.]+)$')[0].astype(float)
    df = df.sort_values(['fold', 'method', 'lam'])
    sel = {}
    for fold in (0, 1):
        ce = df[(df.fold == fold) & (df.method == 'ce')].val_CE.iloc[0]
        for m in ('drift', 'var', 'margin', 'tail'):
            d = df[(df.fold == fold) & (df.method == m)].sort_values(['val_CE', 'lam'])
            best = d.iloc[0]
            sel[f'{fold}_{m}'] = dict(lam=float(best.lam), val_CE=float(best.val_CE), val_acc=float(best.val_acc),
                                      beats_ce_on_heldRX_CE=bool(best.val_CE < ce), ce_val_CE=float(ce))
    jobs = []
    for fold in (0, 1):
        for s in SEEDS:
            jobs.append(f'--stage formal --fold {fold} --method ls --seed {s}')
            for m in ('drift', 'var', 'margin', 'tail'):
                jobs.append(f'--stage formal --fold {fold} --method {m} --lam {sel[f"{fold}_{m}"]["lam"]:g} --seed {s}')
            jobs.append(f'--stage formal --fold {fold} --method tail_drift --lam {sel[f"{fold}_tail"]["lam"]:g} '
                        f'--lam2 {sel[f"{fold}_drift"]["lam"]:g} --seed {s}')
            for m in ('drift', 'tail'):
                jobs.append(f'--stage formal --fold {fold} --method {m} --lam {sel[f"{fold}_{m}"]["lam"]:g} --seed {s} --set rand')
    jobs.sort(key=lambda j: 0 if 'drift' in j else 1)   # slow jobs first
    out = Path(a.out)
    (out / 'lambda_p1.json').write_text(json.dumps(sel, indent=1))
    (out / 'jobs_formal.txt').write_text('\n'.join(jobs) + '\n')
    df.to_csv(out / 'search_table.csv', index=False)
    print(df[['fold', 'method', 'lam', 'val_CE', 'val_acc', 'epoch']].to_string(index=False))
    print(json.dumps(sel, indent=1))
    print(len(jobs), 'formal jobs')


if __name__ == '__main__':
    main()
