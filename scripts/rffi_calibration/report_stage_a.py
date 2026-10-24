#!/usr/bin/env python3
"""Produce a fixed Stage A decision and evidence index from independently verified outputs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from rffi_calibration.registry import Registry, file_sha

HEADS = ('H0', 'H0S', 'H1', 'H1e', 'H2fix', 'H2L', 'H2*', 'H3', 'H4',
         'H4-AK', 'H4-pooled', 'H5a', 'H5b', 'H5c*')
BUDGETS = (0, 1, 2, 5, 10)


def crossfit_verify(rows):
    checked = 0
    for family, prefix in (('H2', 'H2['), ('H5c', 'H5c[')):
        raw = rows[rows['head'].str.startswith(prefix)]
        for rx in raw.rx.unique():
            others = raw[raw.rx != rx]
            for m in raw.m.unique():
                means = others[others.m == m].groupby(['head', 'rx']).acc.mean().groupby('head').mean()
                selected = rows[(rows.rx == rx) & (rows.m == m) & (rows['head'] == family + '*')]
                if means.empty:
                    assert selected.empty
                    continue
                best = sorted(means.index, key=lambda h: (-means[h], float(h.split('=')[1].rstrip(']'))))[0]
                assert selected.selected.eq(best).all() and len(selected) > 0
                original = raw[(raw.rx == rx) & (raw.m == m) & (raw['head'] == best)].sort_values('draw')
                assert np.array_equal(original.acc.to_numpy(), selected.sort_values('draw').acc.to_numpy())
                checked += len(selected)
    return checked


def pair(df, a, b, m):
    per = df[df.m == m].groupby(['head', 'rx']).acc.mean().unstack('head')
    diff = (per[a] - per[b]).dropna()
    n = len(diff)
    interval = student_t.ppf(.975, n - 1) * diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(a=a, b=b, m=m, n_rx=n, mean_pp=100 * diff.mean(),
                lo_pp=100 * (diff.mean() - interval), hi_pp=100 * (diff.mean() + interval),
                n_improved=int((diff > 0).sum()), per_rx_pp={str(k): float(100 * v) for k, v in diff.items()})


def main():
    reg = Registry()
    out = ROOT / 'reports/rffi_calibration/stage_A'
    out.mkdir(parents=True, exist_ok=True)
    matrix = json.loads((ROOT / 'configs/rffi_calibration/formal_matrix.json').read_text())
    datasets, gates, sources, crossfit_checked = {}, [], [], 0
    for job in matrix['jobs']:
        path = ROOT / 'reports/rffi_calibration/E4_frozen_heads' / job['job_id'] / 'gpu_head_fix_v2'
        verification = json.loads((path / 'independent_verification.json').read_text())
        assert verification['verified']
        assert verification['rows_sha256'] == file_sha(path / 'rows.csv')
        assert verification['summary_sha256'] == file_sha(path / 'summary.json')
        rows = pd.read_csv(path / 'rows.csv')
        crossfit_checked += crossfit_verify(rows)
        summary = json.loads((path / 'summary.json').read_text())
        sources.append(dict(job=job, result=str(path.relative_to(ROOT)), verification=verification))
        rows['job_id'] = job['job_id']
        rows['split'] = job['split']
        datasets.setdefault(job['dataset'], []).append(rows)
        if job['dataset'] == 'wisig':
            means = rows[rows.m == 1].groupby(['head', 'rx']).acc.mean().groupby('head').mean()
            simple = max(('H1', 'H2*', 'H2L', 'H3'), key=lambda h: means[h])
            delta = float(100 * (means['H4'] - means[simple]))
            gates.append(dict(split=job['split'], head=simple, h4_acc=float(100 * means['H4']),
                              simple_acc=float(100 * means[simple]), delta_pp=delta, passed=bool(delta >= 2.),
                              paired=pair(rows, 'H4', simple, 1)))
        assert summary['mode'] == 'E4_REAL'
    aggregates, comparisons = {}, {}
    for dataset, chunks in datasets.items():
        df = pd.concat(chunks, ignore_index=True)
        assert not df.duplicated(['job_id', 'rx', 'head', 'm', 'draw']).any()
        df.to_csv(out / f'{dataset}_rows.csv', index=False)
        per = df.groupby(['head', 'm', 'rx']).acc.mean()
        table = []
        for head in HEADS:
            for m in BUDGETS:
                if (head, m) not in per.index.droplevel('rx'):
                    continue
                values = per.loc[(head, m)]
                row = dict(head=head, m=m, accuracy_pct=100 * values.mean(),
                           worst_rx_pct=100 * values.min(), n_rx=len(values))
                selected = df[(df['head'] == head) & (df.m == m)]
                for key in ('macro_f1', 'nll', 'brier', 'ece', 'adapt_s', 'predict_s'):
                    row[key] = float(selected.groupby('rx')[key].mean().mean())
                table.append(row)
        aggregates[dataset] = table
        pd.DataFrame(table).to_csv(out / f'{dataset}_aggregate.csv', index=False)
        comparisons[dataset] = [pair(df, 'H4', head, m) for head in
                                ('H1', 'H2L', 'H3', 'H4-AK', 'H4-pooled') for m in (1, 5)]
        if dataset == 'wisig':
            comparisons[dataset] += [pair(df, 'H4', 'H2*', m) for m in (1, 5)]
    all_paths = sorted((ROOT / 'reports/rffi_calibration/E4_A0').glob('*/gpu_head_fix_v2/summary.json'))
    assert len(all_paths) == 6
    a0 = []
    for path in all_paths:
        directory = path.parent
        v = json.loads((directory / 'independent_verification.json').read_text())
        assert v['verified'] and v['summary_sha256'] == file_sha(path)
        crossfit_checked += crossfit_verify(pd.read_csv(directory / 'rows.csv'))
        summary = json.loads(path.read_text())
        assert summary['mode'] == 'E4_A0_WISIG8_HISTORICAL'
        a0.append(dict(name=directory.parent.name, aggregate=summary['aggregate'], verification=v))
    execution = json.loads((ROOT / 'reports/rffi_calibration/stage_A_execution/gpu_head_fix_v2/evaluation_execution.json').read_text())
    assert len(execution['records']) == 12 and all(r['verified'] for r in execution['records'])
    # Reserve a conservative 30 GPU seconds for the additional GPU regression test.
    gpu_hours = execution['total_reserved_gpu_hours'] + 30 / 3600.
    assert gpu_hours < 1.5
    decision = dict(decision='STOP_DECOMPOSITION', scope='this fixed T0 frozen-head Stage A only',
                    stage_B_active=False, optional_T1_fits=0, fits=6, fit_failures=0, fit_retries=0,
                    initial_evaluation_failures=12, corrected_evaluations_verified=12,
                    max_gpu_hours=1.5, total_conservative_gpu_hours=gpu_hours,
                    additional_gpu_regression_allowance_seconds=30, gates=gates,
                    shen_tuned_family_gate='unavailable: only one encoder-held RX per checkpoint',
                    shen_fixed_head_h4_vs_h2l_improved_rx=0, crossfit_rows_independently_checked=crossfit_checked,
                    third_multirx_validation='not satisfied', sealed_numeric_access=False,
                    temperature_fitting='not run; probability metrics are unscaled',
                    fitting_registry_fingerprint=json.loads((ROOT / 'reports/rffi_calibration/stage_A_execution/execution.json').read_text())['registry_fingerprint'],
                    evaluation_ledger_recovery=execution.get('ledger_recovered', False))
    (out / 'decision.json').write_text(json.dumps(decision, indent=2))
    (out / 'paired.json').write_text(json.dumps(comparisons, indent=2))
    (out / 'A0_diagnostics.json').write_text(json.dumps(a0, indent=1))
    (out / 'evidence_index.json').write_text(json.dumps(dict(
        main_sources=sources, a0_directories=[str(p.parent.relative_to(ROOT)) for p in all_paths],
        historical_base='b6ce3c25648b958424cd4818067f65e800985f1d',
        input_integrity='reports/rffi_calibration/stage_A_execution/input_integrity.json'), indent=2))
    lines = [
        '# Stage A: STOP_DECOMPOSITION',
        '',
        'Date: 2026-10-24. Fixed T0 encoder-held source-selection screening; not a final-data test.',
        '',
        'Six T0 fits completed without failed fits or retries. All twelve corrected A0/A3 evaluations',
        'passed independent score-to-metric and aggregate verification. Twelve initial evaluation',
        'failures (CPU/CUDA native-head mismatch, before rows were written) are retained separately.',
        f'Total conservative A100 reservation: **{gpu_hours:.6f} GPUh**, below 1.5 GPUh.',
        '',
        '## Decision',
        '',
        'H4 failed the preregistered +2 pp WiSig 1-shot gate in both folds. On Shen, H4 was below',
        'fixed H2L on all four RX at both 1 and 5 shots. Stop claiming the current decomposition',
        'is necessary; preserve simple calibration and negative evidence. No Stage B/T1/final test.',
        '',
        '| WiSig split | H4 1-shot | strongest simple head | simple 1-shot | delta pp | gate |',
        '|---|---:|---|---:|---:|---|',
    ]
    for gate in gates:
        lines.append(f'| {gate["split"]} | {gate["h4_acc"]:.3f} | {gate["head"]} | '
                     f'{gate["simple_acc"]:.3f} | {gate["delta_pp"]:+.3f} | FAIL |')
    for dataset, items in aggregates.items():
        lines += ['', f'## {dataset.title()}', '',
                  '| Head | 0-shot | 1-shot | 2-shot | 5-shot | 10-shot |', '|---|---:|---:|---:|---:|---:|']
        lookup = {(i['head'], i['m']): i for i in items}
        for head in HEADS:
            if not any((head, m) in lookup for m in BUDGETS):
                continue
            cells = [f'{lookup[(head, m)]["accuracy_pct"]:.2f}' if (head, m) in lookup else 'N/A' for m in BUDGETS]
            lines.append('| ' + ' | '.join([head, *cells]) + ' |')
    lines += [
        '', '## Interpretation Limits', '',
        '- WiSig folds have disjoint selection RX but overlapping fitting RX; do not claim independent datasets/seeds.',
        '- Shen cross-fitted H2*/H5c* are unavailable. The other LORO encoder fitted this RX, so pooling cannot fix exclusion.',
        '- Shen fixed-head results average four RX, not support draws as independent samples; raw per-RX rows are retained.',
        '- Both WiSig and both Shen-B jobs selected the final step. New episode-schedule encoders may be underconverged.',
        '- These are new T0 encoders and different query/capture protocols, not reproductions of historical P2E accuracies.',
        '- H4 underperformance does not prove all Bayesian heads or all learned representations fail.',
        '- H4-pooled beats H4 in the aggregate; finite-center U brings only small aggregate changes. No mechanism is inferred.',
        '- Probability metrics use temperature 1; source-only temperature fitting was not executed.',
        '- Shen row order is not verified temporal separation. OSU numeric evaluation was not run.',
        '- Third independent multi-RX validation remains unresolved; the program is not a completed three-dataset study.',
        '', '## Evidence', '',
        '- `decision.json`, `paired.json`, `wisig_aggregate.csv`, `shen_aggregate.csv`, and `evidence_index.json`.',
        '- Full local prediction arrays, manifests, confusion counts and per-task verification are in the indexed directories.',
        '- Initial failure logs remain in each task parent; successful results are only under `gpu_head_fix_v2`.',
        '- Source integrity: 68 source assets matched historical SHA256 registration; no sealed/development numeric access.',
        '- Test counts and command timings are recorded in the final execution verification record.',
        '- Weights, signal arrays, feature vectors and class-score arrays are local only; no push performed.',
    ]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
