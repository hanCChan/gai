#!/usr/bin/env python3
"""Source-only λ selection (PROTOCOL_P2 §3): per (fold, method) min held-RX CE, tie -> smaller λ.
Grid {0.3, 1.0, 3.0}; ce/ls search values reused from P1 search. Writes lambda_p2.json,
search_table.csv and the formal job list (36 fits: rxnull, cmean, rxnull9, rxnull_ls, drift_ls,
rxnull_rand at λ*, 2 folds x 3 seeds; rxnull_rand uses λ*(rxnull)).

    python3 select_lambda_p2.py --search outputs/receiver_p2/search --p1search outputs/receiver_p1/search --out reports/receiver_p2
"""
import argparse
import json
from pathlib import Path

import pandas as pd

SEEDS = (20261013, 20261014, 20261015)
METHODS = ('rxnull', 'cmean', 'rxsel', 'rxnull_ls', 'drift_ls')


def rows_of(search_dir):
    rows = []
    for st in sorted(Path(search_dir).glob('*/status.json')):
        s = json.loads(st.read_text())
        n = s['run_id'].split('_')   # shen rank16 CV{f} {method_with_underscores}{lam} Rq {stage} seed
        rows.append(dict(run_id=s['run_id'], fold=int(n[2][2:]), arm='_'.join(n[3:-3]),
                         stage=n[-2], val_CE=s['best_val_CE'], val_acc=s['best_val_acc'],
                         epoch=s['best_epoch']))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--search', required=True)
    ap.add_argument('--p1search', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    df = rows_of(a.search)
    p1 = rows_of(a.p1search)
    df = pd.concat([df, p1], ignore_index=True)
    # arm tag = method + lam with :g formatting; match the method prefix explicitly (methods
    # contain underscores and can end in 's', so regex letters-only extraction is ambiguous)
    mtab = ('rxnull_ls', 'drift_ls', 'rxnull', 'cmean', 'rxsel', 'ce', 'ls')
    def parse(arm):
        for m in mtab:
            if arm.startswith(m):
                rest = arm[len(m):]
                try:
                    return m, float(rest)
                except ValueError:
                    return m, None
        return None, None
    df[['method', 'lam']] = df.arm.apply(lambda a: pd.Series(parse(a)))
    df = df.dropna(subset=['lam'])
    df = df.sort_values(['fold', 'method', 'lam'])
    sel = {}
    for fold in (0, 1):
        ce = df[(df.fold == fold) & (df.method == 'ce')].val_CE.iloc[0]
        for m in METHODS:
            d = df[(df.fold == fold) & (df.method == m)].sort_values(['val_CE', 'lam'])
            best = d.iloc[0]
            sel[f'{fold}_{m}'] = dict(lam=float(best.lam), val_CE=float(best.val_CE),
                                      val_acc=float(best.val_acc),
                                      beats_ce_on_heldRX_CE=bool(best.val_CE < ce), ce_val_CE=float(ce))
    jobs = []
    for fold in (0, 1):
        for s in SEEDS:
            for m in METHODS:
                jobs.append(f'--stage formal --fold {fold} --method {m} --lam {sel[f"{fold}_{m}"]["lam"]:g} --seed {s}')
            jobs.append(f'--stage formal --fold {fold} --method rxnull_rand --lam {sel[f"{fold}_rxnull"]["lam"]:g} --seed {s}')
    out = Path(a.out)
    (out / 'lambda_p2.json').write_text(json.dumps(sel, indent=1))
    (out / 'jobs_formal.txt').write_text('\n'.join(jobs) + '\n')
    df.to_csv(out / 'search_table.csv', index=False)
    print(df[['fold', 'method', 'lam', 'val_CE', 'val_acc', 'epoch']].to_string(index=False))
    print(json.dumps(sel, indent=1))
    print(len(jobs), 'formal jobs')


if __name__ == '__main__':
    main()
