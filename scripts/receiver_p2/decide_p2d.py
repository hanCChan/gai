#!/usr/bin/env python3
"""P2D decision rules (pre-declared, applied by run_p2d_chain.py after each formal round).

"X beats Y" = P2 rule: seed-mean Δacc > 0 in BOTH folds and >= 5/6 paired (fold, seed) Δ > 0.
Arms compared against ce, ls (anchors) and ccalign (the alignment control that tied the
method arms on Shen). rxnull is the P2 reference arm.

Outcomes (written to decision_p2d.json, consumed by the chain):
  GO_CROSS_DATASET   main arm beats ce, ls AND ccalign  -> WiSig transfer round
  ITERATE            main arm beats ce and ls but NOT ccalign -> spawn next candidate family
  STOP_TAIL          no arm beats ls                    -> diagnosis-only paper path
Also reports tail-audit stats (does the winner also cut mean_pflip / worst RX?).

    python3 decide_p2d.py --eval reports/receiver_p2d/eval --out reports/receiver_p2d/decision_p2d.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ev = Path(a.eval)
    R = pd.read_csv(ev / 'runs.csv')
    metrics = ['acc', 'macro_f1', 'worst_rx_acc', 'pluto_acc', 'P_conf', 'nonP_conf', 'rho_row_end', 'mean_pflip'] \
        if 'mean_pflip' in R else ['acc', 'macro_f1', 'worst_rx_acc', 'pluto_acc', 'P_conf', 'nonP_conf', 'rho_row_end']
    piv = {m: R.pivot_table(index=['fold', 'seed'], columns='method', values=m) for m in metrics}

    def beats(x, y):
        d = piv['acc'][x] - piv['acc'][y]
        fm = d.groupby(level='fold').mean()
        return dict(x=x, y=y, beats=bool((fm > 0).all() and (d > 0).sum() >= 5),
                    delta_pp={f'fold{k}': round(float(v) * 100, 2) for k, v in fm.items()},
                    n_pos=int((d > 0).sum()))

    arms = [m for m in R.method.unique() if m not in ('ce', 'ls', 'rxnull', 'ccalign', 'rxlogit')]
    pairs = []
    for m in arms:
        for anchor in ('ce', 'ls', 'ccalign', 'rxnull', 'rxlogit'):
            if anchor in R.method.unique():
                pairs.append((m, anchor))
    for anchor in ('rxnull', 'rxlogit', 'ccalign'):
        if anchor in R.method.unique():
            pairs.append((anchor, 'ce'))
            pairs.append((anchor, 'ls'))
    T = pd.DataFrame([beats(x, y) for x, y in pairs])
    T.to_csv(ev / 'pairwise_beats.csv', index=False)
    B = lambda x, y: (bool(T[(T.x == x) & (T.y == y)].beats.iloc[0])
                      if len(T[(T.x == x) & (T.y == y)]) else False)

    # rank arms by "beats ce+ls", prefer those that also beat ccalign
    winners = []
    for m in arms:
        bc, bls, bcc = B(m, 'ce'), B(m, 'ls'), B(m, 'ccalign')
        if bc and bls:
            winners.append(dict(arm=m, beats_ce=True, beats_ls=True, beats_ccalign=bcc))
    main_arm = None
    for w in winners:
        if w['beats_ccalign']:
            main_arm = w['arm']
            break
    if main_arm is None and winners:
        main_arm = max(winners, key=lambda w: T[(T.x == w['arm']) & (T.y == 'ccalign')]['delta_pp'].apply(
            lambda d: sum(d.values())).iloc[0] if not T[(T.x == w['arm']) & (T.y == 'ccalign')].empty else -1)['arm']
    any_beats_ls = any(B(m, 'ls') for m in arms) or B('rxnull', 'ls') or B('rxlogit', 'ls')
    outcome = ('GO_CROSS_DATASET' if (main_arm and B(main_arm, 'ccalign'))
               else 'ITERATE' if (main_arm or B('rxnull', 'ccalign') or B('rxlogit', 'ccalign'))
               else 'STOP_TAIL' if not any_beats_ls else 'STOP_TAIL')
    out = dict(outcome=outcome, main_arm=main_arm, winners=winners, arms=arms,
               summary_path=str(ev / 'summary_by_fold_arm.csv'))
    S = R.groupby(['fold', 'method'])[metrics].mean().round(4)
    S.to_csv(ev / 'summary_by_fold_arm.csv')
    (Path(a.out)).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(S.to_string())
    print(T.to_string(index=False))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
