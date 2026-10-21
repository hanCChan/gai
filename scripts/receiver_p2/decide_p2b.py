#!/usr/bin/env python3
"""Apply PROTOCOL_P2B §3 to eval runs.csv (ce, ls, rxnull from earlier phases + rxlogit, ccalign).
"X beats Y" = P2 rule: seed-mean Δacc > 0 in BOTH folds and >= 5/6 paired (fold, seed) Δ > 0.

    python3 decide_p2b.py --eval reports/receiver_p2b/eval
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True)
    a = ap.parse_args()
    ev = Path(a.eval)
    R = pd.read_csv(ev / 'runs.csv')
    metrics = ['acc', 'macro_f1', 'worst_rx_acc', 'pluto_acc', 'P_conf', 'nonP_conf', 'rho_row_end']
    piv = {m: R.pivot_table(index=['fold', 'seed'], columns='method', values=m) for m in metrics}

    def beats(x, y):
        d = piv['acc'][x] - piv['acc'][y]
        fm = d.groupby(level='fold').mean()
        return dict(x=x, y=y, beats=bool((fm > 0).all() and (d > 0).sum() >= 5),
                    delta_pp={f'fold{k}': round(float(v) * 100, 2) for k, v in fm.items()}, n_pos=int((d > 0).sum()))

    pairs = [('rxlogit', 'ce'), ('rxlogit', 'ls'), ('rxnull', 'rxlogit'), ('rxlogit', 'rxnull'),
             ('rxlogit', 'ccalign'), ('rxnull', 'ccalign'), ('ccalign', 'ce'), ('ccalign', 'ls'),
             ('rxnull', 'ce'), ('rxnull', 'ls')]
    T = pd.DataFrame([beats(x, y) for x, y in pairs])
    T.to_csv(ev / 'pairwise_beats.csv', index=False)
    B = lambda x, y: bool(T[(T.x == x) & (T.y == y)].beats.iloc[0])
    if B('rxlogit', 'ce') and B('rxlogit', 'ls') and not B('rxnull', 'rxlogit'):
        main_obj, rule = 'rxlogit', 'R1'
    elif B('rxnull', 'rxlogit'):
        main_obj, rule = 'rxnull', 'R2'
    else:
        main_obj, rule = 'rxnull+rxlogit', 'R3'
    mains = [m for m in ('rxlogit', 'rxnull') if m in main_obj]
    novelty = {m: B(m, 'ccalign') for m in mains}
    lam = json.loads((ev.parent / 'lambda_p2b.json').read_text())
    out = dict(rule=rule, main_objective=main_obj, beats_ccalign=novelty,
               boundary_limited_lam=[k for k, v in lam.items() if v['lam'] == 3.0])
    summary = R.groupby(['fold', 'method'])[metrics].mean().round(4)
    summary.to_csv(ev / 'summary_by_fold_arm.csv')
    (ev.parent / 'decision_p2b.json').write_text(json.dumps(out, indent=1))
    print(summary.to_string())
    print(T.to_string(index=False))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
