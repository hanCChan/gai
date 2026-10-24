#!/usr/bin/env python3
"""Independent verification: saved scores -> metrics -> RX aggregate -> decision (plan §10 last row).

Written without importing rffi_calibration.evaluator or rffi_post_a.reporting aggregation, so it can
catch errors in them. Works on R0 evaluation directories (rows.csv + scores.npz) and on D1 task outputs
(<task>_rows.csv + <task>_rx<rx>_scores.npz).

    python scripts/rffi_post_a/verify_results.py --r0 reports/rffi_post_a/R0
    python scripts/rffi_post_a/verify_results.py --d1 reports/rffi_post_a/D1
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LOWSHOT = (1, 5)


def acc_from_scores(scores, y):
    return float((np.asarray(scores).argmax(1) == np.asarray(y)).mean())


def check_rows_against_scores(rows, scores, key_fn, y_fn, tol=1e-12):
    """rows: dicts with acc; key_fn(row) -> scores key or None; y_fn(row) -> query labels."""
    worst, n = 0., 0
    for r in rows:
        k = key_fn(r)
        if k is None or k not in scores:
            continue
        worst = max(worst, abs(acc_from_scores(scores[k], y_fn(r)) - float(r['acc'])))
        n += 1
    return dict(checked=n, max_abs=worst, ok=n > 0 and worst <= tol)


def rx_equal_means(rows, group_keys):
    """draw mean per (group..., rx, m), then RX-equal mean per (group..., m)."""
    per = defaultdict(list)
    for r in rows:
        per[tuple(r[k] for k in group_keys) + (r['rx'], int(r['m']))].append(float(r['acc']))
    rx_mean = {k: sum(v) / len(v) for k, v in per.items()}
    agg = defaultdict(list)
    for k, v in rx_mean.items():
        agg[k[:-2] + (k[-1],)].append(v)
    return rx_mean, {k: sum(v) / len(v) for k, v in agg.items()}


def lowshot(rx_mean, prefix, rx):
    v = [rx_mean.get(prefix + (rx, m)) for m in LOWSHOT]
    return None if None in v else sum(v) / 2


def decide_r0(rows, held):
    """Plan §7.7 rules on checkpoint 'last.pt'. held: {split: [rx, ...]}."""
    rows = [r for r in rows if r['checkpoint'] == 'last.pt']
    rx_mean, _ = rx_equal_means(rows, ('arm', 'head', 'split'))
    out, ok = {}, defaultdict(list)
    pos_rx = 0
    for split, rxs in held.items():
        f = lambda arm, head, m=None: (sum(lowshot(rx_mean, (arm, head, split), rx) for rx in rxs) / len(rxs) if m is None
                                       else sum(rx_mean[(arm, head, split, rx, m)] for rx in rxs) / len(rxs))
        g = f('P_RX', 'H1e') - f('C_MATCH', 'H1e')
        d1 = f('P_RX', 'H1e', 1) - f('C_MATCH', 'H1e', 1)
        d5 = f('P_RX', 'H1e', 5) - f('C_MATCH', 'H1e', 5)
        rxs_gain = f('P_RX', 'H1e') - f('P_MIX', 'H1e')
        per_rx = [lowshot(rx_mean, ('P_RX', 'H1e', split), rx) - lowshot(rx_mean, ('C_MATCH', 'H1e', split), rx) for rx in rxs]
        pos_rx += sum(x > 0 for x in per_rx)
        h0 = (sum(rx_mean[('P_RX', 'H0', split, rx, 0)] for rx in rxs) - sum(rx_mean[('C_MATCH', 'H0', split, rx, 0)] for rx in rxs)) / len(rxs)
        sec = [h for h in ('H1', 'H2L', 'H5a', 'H5b', 'H5c[k0=1]') if all(lowshot(rx_mean, ('C_MATCH', h, split), rx) is not None for rx in rxs)]
        best_sec = max(f('C_MATCH', h) for h in sec)
        out[split] = dict(gain_pp=100 * g, d1_pp=100 * d1, d5_pp=100 * d5, rx_structure_pp=100 * rxs_gain,
                          worst_rx_drop_pp=-100 * min(per_rx), h0_drop_pp=-100 * h0, p_rx_h1e=100 * f('P_RX', 'H1e'),
                          best_c_match_secondary=100 * best_sec, per_rx_pp=[100 * x for x in per_rx])
        ok['gain'].append(g >= .02)
        ok['per_budget'].append(d1 >= 0 and d5 >= 0)
        ok['rx'].append(rxs_gain >= .01)
        ok['risk'].append(-min(per_rx) <= .03 and -h0 <= .05)
        ok['competitive'].append(f('P_RX', 'H1e') >= best_sec)
    ok = {k: all(v) for k, v in ok.items()}
    ok['rx_consistency'] = pos_rx >= 6
    episodic = {s: out[s]['gain_pp'] for s in out}
    if all(ok.values()):
        decision = 'GO_DESIGN_SHEN'
    elif ok['gain'] and ok['per_budget'] and ok['risk'] and ok['competitive']:
        decision = 'EPISODIC_GAIN_RX_UNPROVEN'
    elif ok['gain'] and ok['per_budget']:
        decision = 'GAIN_NOT_COMPETITIVE'
    else:
        decision = 'NO_GAIN_IN_FIXED_PILOT'
    return dict(decision=decision, checks=ok, folds=out, episodic_gain_pp=episodic,
                note='INCONCLUSIVE_OPTIMIZATION must be set by a human reviewer from the training logs; not automatic')


def load_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def verify_r0(root):
    rows, report = [], {}
    for d in sorted((root / 'evaluation').glob('R0_*')):
        rs = load_csv(d / 'rows.csv')
        with np.load(d / 'scores.npz', allow_pickle=False) as a:
            sc = {k: a[k] for k in a.files}
        report[d.name] = check_rows_against_scores(
            rs, sc, lambda r: f'{r["rx"]}|{r["head"]}|{r["m"]}|{r["draw"]}', lambda r: sc[f'{r["rx"]}|y_query'])
        rows += rs
    held = defaultdict(set)
    for r in rows:
        held[r['split']].add(r['rx'])
    return dict(score_replay=report, all_ok=all(v['ok'] for v in report.values()),
                decision=decide_r0(rows, {k: sorted(v) for k, v in held.items()}) if rows else None)


def verify_d1(root):
    report = {}
    for p in sorted(root.glob('*_rows.csv')):
        task = p.name[:-len('_rows.csv')]
        rs = load_csv(p)
        for f in sorted(root.glob(f'{task}_rx*_scores.npz')):
            rx = f.name[len(task) + 3:-len('_scores.npz')]
            with np.load(f, allow_pickle=False) as a:
                sc = {k: a[k] for k in a.files}
            report[f'{task}|{rx}'] = check_rows_against_scores(
                [r for r in rs if r['rx'] == rx], sc, lambda r: f'{r["slot"]}|{r["m"]}|{r["draw"]}', lambda r: sc['y_query'])
    return dict(score_replay=report, all_ok=bool(report) and all(v['ok'] for v in report.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r0')
    ap.add_argument('--d1')
    a = ap.parse_args()
    res = {}
    if a.r0:
        res['R0'] = verify_r0(Path(a.r0))
    if a.d1:
        res['D1'] = verify_d1(Path(a.d1))
    print(json.dumps(res, indent=1, default=str))
    sys.exit(0 if all(v.get('all_ok') for v in res.values()) else 1)


if __name__ == '__main__':
    main()
