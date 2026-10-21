#!/usr/bin/env python3
"""Apply PROTOCOL_P2 §4 mechanically to eval_p2 runs.csv. Writes decision.json + paired tables.
"X beats Y": seed-mean Δacc > 0 in BOTH folds and >= 5/6 paired (fold, seed) Δ > 0.

    python3 decide_p2.py --eval reports/receiver_p2/eval
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
    metrics = ['acc', 'macro_f1', 'worst_rx_acc', 'pluto_acc', 'P_conf', 'nonP_conf',
               'P_nm_mean', 'P_nm_q10', 'rho_row_end']
    piv = {m: R.pivot_table(index=['fold', 'seed'], columns='method', values=m) for m in metrics}

    def beats(x, y, m='acc'):
        d = piv[m][x] - piv[m][y]
        fm = d.groupby(level='fold').mean()
        return bool((fm > 0).all() and (d > 0).sum() >= 5), {f'fold{k}': round(float(v) * 100, 2) for k, v in fm.items()}, int((d > 0).sum())

    arms = [c for c in piv['acc'].columns if c not in ('ce', 'ls')]
    table = []
    pairs = [(x, y) for x in arms for y in ('ce', 'ls')]
    pairs += [('rxnull', 'rxnull_rand'), ('rxnull', 'rxsel'), ('rxnull', 'cmean'),
              ('rxnull_ls', 'rxnull'), ('rxnull_ls', 'ls'), ('drift_ls', 'ls'),
              ('rxsel', 'rxnull_rand'), ('cmean', 'rxnull_rand')]
    for x, y in pairs:
        if x in piv['acc'] and y in piv['acc']:
            b, fm, n = beats(x, y)
            table.append(dict(x=x, y=y, beats=b, seed_mean_delta_pp=fm, n_pos_of_6=n))
    T = pd.DataFrame(table)
    T.to_csv(ev / 'pairwise_beats.csv', index=False)

    def B(x, y):
        r = T[(T.x == x) & (T.y == y)]
        return bool(r.beats.iloc[0]) if len(r) else False

    def pconf_down(x):
        d = (piv['P_conf'][x] - piv['P_conf']['ce']).groupby(level='fold').mean()
        return bool((d < 0).all())

    def nonp_ok(x):
        return float((piv['nonP_conf'][x] - piv['nonP_conf']['ce']).mean()) <= 0.005

    go = []
    if 'rxnull' in piv['acc'] and B('rxnull', 'ce') and B('rxnull', 'ls') and \
            B('rxnull', 'rxnull_rand') and B('rxnull', 'rxsel') and pconf_down('rxnull') and nonp_ok('rxnull'):
        go.append('rxnull')
    aux = [x for x in ('rxnull', 'cmean', 'rxsel', 'rxnull_ls', 'drift_ls', 'rxnull_rand')
           if x in piv['acc']]
    beat_ce_ls = [x for x in aux if B(x, 'ce') and B(x, 'ls')]
    beat_ce = [x for x in aux if B(x, 'ce')]
    decision = 'GO' if go else ('MODIFY' if beat_ce_ls else ('WAIT_GATE' if not beat_ce else 'STOP'))
    # boundary-limited λ note
    lam = json.loads((ev.parent / 'lambda_p2.json').read_text())
    boundary = [k for k, v in lam.items() if v['lam'] == 3.0]
    summary = R.groupby(['fold', 'method'])[metrics].mean().round(4)
    summary.to_csv(ev / 'summary_by_fold_arm.csv')
    rho_end = {m: float(R[R.method == m].rho_row_end.mean()) for m in piv['acc'].columns}
    out = dict(decision=decision, go_candidates=go, beat_ce=beat_ce, beat_ce_and_ls=beat_ce_ls,
               boundary_limited_lam=boundary, rho_row_end_by_arm=rho_end,
               pconf_down={x: pconf_down(x) for x in aux if x in piv['P_conf']},
               nonP_ok={x: nonp_ok(x) for x in aux if x in piv['nonP_conf']})
    (ev.parent / 'decision.json').write_text(json.dumps(out, indent=1))
    print(summary.to_string())
    print(T.to_string(index=False))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
