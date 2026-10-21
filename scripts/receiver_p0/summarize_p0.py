#!/usr/bin/env python3
"""Summarize P0 + PoC into the decision table that fixes the next algorithm phase.

Inputs: p0a/ (pairs, geometry, refit), p0c/ (risk), p0_wisig/, poc eval runs.
Outputs: REPORT.md skeleton with the six pre-registered questions answered from CSVs,
plus the PoC comparison table (CE anchor vs sub/drift/pair x lambda x seed).

    python3 summarize_p0.py --p0a DIR --p0c DIR [--wisig DIR] [--poc-eval DIR] [--poc-eval2 DIR] --out DIR
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


def load(d, pat):
    return pd.concat([pd.read_csv(f) for f in sorted(Path(d).glob(pat))], ignore_index=True)


def fmt(x, nd=3):
    return 'nan' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p0a', required=True)
    ap.add_argument('--p0c', required=True)
    ap.add_argument('--wisig', default='')
    ap.add_argument('--poc-eval', default='')
    ap.add_argument('--poc-eval2', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = []
    L.append('# P0 report (auto-generated): pairwise decision-failure localization\n')
    L.append('Rules: docs/receiver_p0/PROTOCOL_P0.md (frozen before any P0 output).\n')

    # Q1: which pairs are hard
    m = pd.read_csv(Path(a.p0c) / 'top_pairs.csv')
    L.append('## Q1 Which pairs fail (source ranking, frozen rule R_q)\n')
    for fold in (0, 1):
        d = m[m.fold == fold].head(10)
        L.append(f'\nfold {fold} top-10 by source risk R_q (of 90 ordered pairs):\n')
        L.append(d[['t', 'u', 'R_q', 'R_conf', 'outer_conf', 'src_dclass']].to_markdown(index=False))

    # Q2: predictability
    corr = pd.read_csv(Path(a.p0c) / 'risk_correlation.csv')
    L.append('\n## Q2 Can source risk predict unseen-RX failure?\n')
    L.append(corr.to_markdown(index=False))
    ans = json.loads((Path(a.p0c) / 'answers.json').read_text())
    L.append('\nSWAD vs CE at pair level: ' + json.dumps(ans['swad_vs_ce_pair_level']))
    L.append('\nSeed stability of R_q ranking: ' + json.dumps(ans['seed_stability']))

    # Q3: decomposition
    refit = load(a.p0a, 'refit_shard*.csv')
    pairs = load(a.p0a, 'pairs_shard*.csv')
    frozen = pairs[pairs.pool == 'outer'].groupby(['run_id', 'fold', 'method', 'seed', 't', 'u']).conf.mean().reset_index(name='frozen_conf')
    rf = refit[refit.t >= 0].groupby(['run_id', 'fold', 'method', 'seed', 't', 'u']).conf.mean().reset_index(name='refit_conf')
    mm = frozen.merge(rf, on=['run_id', 'fold', 'method', 'seed', 't', 'u'])
    hard = mm[mm.frozen_conf > 0.2]
    L.append('\n## Q3 Encoder collapse or head boundary?\n')
    L.append(f'Hard pairs (frozen conf>0.2): n={len(hard)}; refit-head reproduces collapse (>0.2) in '
             f'{(hard.refit_conf > 0.2).mean():.1%}; mean conf frozen {hard.frozen_conf.mean():.3f} -> refit {hard.refit_conf.mean():.3f}.')
    L.append('\nRefit-vs-frozen by method (outer accuracy / hard-pair conf):\n')
    rec = load(a.p0a, 'recall_shard*.csv')
    fr = rec[rec.pool == 'outer'].groupby('method').accuracy.mean()
    rfall = refit[refit.rx_key == 'OVERALL'].groupby('method').conf.mean()
    hc = hard.groupby('method')[['frozen_conf', 'refit_conf']].mean()
    tab = pd.DataFrame({'outer_acc_frozen': fr, 'outer_acc_refit': rfall}).join(hc)
    L.append(tab.round(3).to_markdown())

    # Q4 capacity
    geo = load(a.p0a, 'geometry_shard*.csv')
    er = geo.dropna(subset=['eff_rank_512']).groupby('method')[['eff_rank_512', 'eff_rank_128']].mean()
    L.append('\n## Q4 Capacity control (rank-16 bottleneck)\n')
    L.append(er.round(2).to_markdown())
    L.append('\nNote: eff_rank ~3.4-3.8 of 512 is the ENCODER rank, not the intrinsic dimension of RF identity.\n')

    # Q5 WiSig
    if a.wisig:
        wr = pd.read_csv(Path(a.wisig) / 'wisig_risk.csv')
        L.append('\n## Q5 WiSig mirror (source -> internal RX)\n')
        L.append(wr.to_markdown(index=False))

    # PoC
    for tag, pe in (('poc', a.poc_eval), ('poc2', a.poc_eval2)):
        if not pe:
            continue
        runs = pd.read_csv(Path(pe) / 'runs.csv')
        pr = runs[runs.rx_key == 'OVERALL']
        agg = pr.groupby(['fold', 'branch', 'lam']).agg(
            acc=('accuracy', 'mean'), acc_sd=('accuracy', 'std'),
            conf_P=('mean_conf_P', 'mean'), collapsed=('n_collapsed_P', 'mean')).reset_index()
        L.append(f'\n## {tag}: PoC arms (seed-mean; CE anchor = frozen recipe rerun)\n')
        L.append(agg.round(4).to_markdown(index=False))
        # Pluto-fold focus
        pl0 = pr[pr.fold == 0]
        L.append('\nfold-0 (Pluto is outer RX) detail:\n')
        L.append(pl0.groupby(['branch', 'lam']).agg(acc=('accuracy', 'mean'), conf_P=('mean_conf_P', 'mean')).round(4).to_markdown(index=False))

    (out / 'REPORT_SKELETON.md').write_text('\n'.join(L) + '\n')
    print('wrote', out / 'REPORT_SKELETON.md')


if __name__ == '__main__':
    main()
