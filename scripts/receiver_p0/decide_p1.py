#!/usr/bin/env python3
"""Apply PROTOCOL_P1 §3–4 mechanically to eval_p1 runs.csv. Writes decision.json + paired tables.

"X beats Y": seed-mean Δacc > 0 in both folds and >= 5/6 paired (fold, seed) Δacc > 0.

    python3 decide_p1.py --eval DIR
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def arm(r):
    return r.method + ('_rand' if r.set == 'rand' else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True)
    a = ap.parse_args()
    ev = Path(a.eval)
    R = pd.read_csv(ev / 'runs.csv')
    R['arm'] = R.apply(arm, axis=1)
    metrics = ['acc', 'macro_f1', 'worst_rx_acc', 'pluto_acc', 'P_conf', 'nonP_conf', 'P_nm_mean', 'P_nm_q10']
    piv = {m: R.pivot_table(index=['fold', 'seed'], columns='arm', values=m) for m in metrics}

    def beats(x, y, m='acc'):
        d = piv[m][x] - piv[m][y]
        fm = d.groupby(level='fold').mean()
        return bool((fm > 0).all() and (d > 0).sum() >= 5), {f'fold{k}': round(float(v) * 100, 2) for k, v in fm.items()}, int((d > 0).sum())

    arms = [c for c in piv['acc'].columns if c != 'ce']
    table = []
    for x in arms:
        for y in ['ce', 'ls']:
            if x == y:
                continue
            b, fm, n = beats(x, y)
            table.append(dict(x=x, y=y, beats=b, seed_mean_delta_pp=fm, n_pos_of_6=n))
    for x, y in (('tail', 'margin'), ('drift', 'var'), ('tail', 'drift'), ('tail_drift', 'tail'), ('tail_drift', 'drift'),
                 ('drift', 'drift_rand'), ('tail', 'tail_rand'), ('var', 'drift'), ('margin', 'tail')):
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
    for cand, simpler, rand in (('tail', 'margin', 'tail_rand'), ('tail_drift', 'margin', 'tail_rand')):
        if cand in piv['acc'] and B(cand, 'ce') and B(cand, 'ls') and B(cand, simpler) and B(cand, rand) and pconf_down(cand) and nonp_ok(cand):
            go.append(cand)
    aux = ['drift', 'var', 'margin', 'tail', 'tail_drift']
    beat_ce_ls = [x for x in aux if x in piv['acc'] and B(x, 'ce') and B(x, 'ls')]
    beat_ce = [x for x in aux if x in piv['acc'] and B(x, 'ce')]
    dv = (piv['acc']['drift'] - piv['acc']['var']).groupby(level='fold').mean() * 100
    drift_equiv_var = bool((dv.abs() < 0.5).all() or ((dv > 0).any() and (dv < 0).any()))
    decision = 'GO' if go else ('MODIFY' if beat_ce else 'STOP')
    summary = R.groupby(['fold', 'arm'])[metrics].mean().round(4)
    summary.to_csv(ev / 'summary_by_fold_arm.csv')
    out = dict(decision=decision, go_candidates=go, beat_ce=beat_ce, beat_ce_and_ls=beat_ce_ls,
               drift_equiv_var=drift_equiv_var, drift_minus_var_pp=dv.round(2).to_dict(),
               pconf_down={x: pconf_down(x) for x in aux if x in piv['acc']},
               nonP_ok={x: nonp_ok(x) for x in aux if x in piv['acc']})
    (ev / 'decision.json').write_text(json.dumps(out, indent=1))
    print(summary.to_string())
    print(T.to_string(index=False))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
