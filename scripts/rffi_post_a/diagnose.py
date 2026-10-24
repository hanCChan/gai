#!/usr/bin/env python3
"""D1 fixed diagnostic matrix on cached frozen features (plan §6), CPU only.

    python scripts/rffi_post_a/diagnose.py --out reports/rffi_post_a/D1          # needs active D1 + caches
    python scripts/rffi_post_a/diagnose.py --synthetic --out /tmp/rffi_post_a_D1  # no data access

Outputs per task: slot rows (all slots x budgets x draws), N/A list, anchor check, H4 parameter panel,
stability block, metric-layer centre errors and H4/H1 flips. At the end of a real run the D1
reservation is closed (active=false) regardless of the outcome.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from rffi_calibration.registry import Registry  # noqa: E402
from rffi_calibration.sampling import RecordTable, build_manifest, manifest_sha  # noqa: E402
from rffi_post_a import diagnostic_stats as DS  # noqa: E402
from rffi_post_a.diagnostic_matrix import (anchor_check, fit_replays, metric_layer_center_errors,  # noqa: E402
                                           metric_layer_flip_analysis, run_task)
from rffi_post_a.reporting import write_csv, write_json  # noqa: E402

CONFIG = ROOT / 'configs/rffi_post_a'
RULES = json.loads((ROOT / 'configs/rffi_calibration/registration.json').read_text())['support_query']


def synthetic_tasks(seed=0):
    """Two toy tasks shaped like WiSig (dates 0/1) and Shen (one date); class-specific RX shifts."""
    rng = np.random.default_rng(seed)
    out = []
    for name, ds, R_fit, held, T, d, n, dates in (('SYN_wisig', 'wisig', 6, [6, 7], 10, 12, 60, (0, 1)),
                                                   ('SYN_shen', 'shen', 4, [4], 10, 16, 120, (0,))):
        proto = 2.5 * rng.standard_normal((T, d))
        E = rng.standard_normal((T, d, 3))
        rows = []
        for r in range(R_fit + len(held)):
            shift = np.einsum('tdk,tk->td', E, .8 * rng.standard_normal((T, 3)))
            for dd in dates:
                for t in range(T):
                    z = proto[t] + shift[t] + .2 * dd + rng.standard_normal((n, d)) * rng.uniform(.6, 1.6, d)
                    rows.append((t, r, dd, z))
        tx = np.concatenate([[t] * len(z) for t, _, _, z in rows])
        rx = np.concatenate([[r] * len(z) for _, r, _, z in rows])
        date = np.concatenate([[dd] * len(z) for _, _, dd, z in rows])
        pos = np.concatenate([np.arange(len(z)) for *_, z in rows])
        z = np.concatenate([z for *_, z in rows])
        src = ~np.isin(rx, held)
        task = dict(task=name, dataset=ds, fit_rx=list(range(R_fit)), held_rx=held)
        out.append((task, dict(z=z[src], tx=tx[src], rx=rx[src].astype(str), date=date[src], position=pos[src]),
                    dict(z=z[~src], tx=tx[~src], rx=rx[~src].astype(str), date=date[~src], position=pos[~src])))
    return out


def target_table(ds, tgt):
    rx = tgt['rx']
    date = np.asarray(tgt['date'])
    cap = np.array([f'rx{int(r):02d}_day{int(d)}' if ds == 'wisig' else str(r) for r, d in zip(rx, date)])
    unit = 'cache_position' if ds == 'wisig' else 'packet'
    return RecordTable(ds, tgt['tx'], rx, date if ds == 'wisig' else np.full(len(rx), -1), tgt['position'], unit,
                       capture=cap if ds == 'wisig' else None)


def diagnose_task(task, src, tgt, matrix, T, out, n_draws, expected_manifests=None):
    t0 = time.time()
    b = DS.source_diagnostics(src['z'], src['tx'], src['rx'], dict(held_rx=task['held_rx'], num_classes=T))
    rx_codes = np.unique(src['rx'], return_inverse=True)[1]
    replays = fit_replays(matrix['replays'], src['z'], src['tx'], rx_codes, T)
    res = dict(task=task['task'], panel=DS.panel_summary(b), targets={})
    tab = target_table(task['dataset'], tgt)
    rows_all = []
    sq = RULES.get(task['dataset'], RULES['wisig'])
    for rx in task['held_rx']:
        man = build_manifest(tab, str(rx), T, sq['rule'], q_per_class=sq['q_per_class'], n_draws=n_draws)
        sha = manifest_sha(man)
        if expected_manifests is not None and expected_manifests.get(str(rx)) != sha:
            raise ValueError(f'{task["task"]} RX {rx}: manifest differs from the cached Stage A manifest')
        rows, na, scores = run_task(b, matrix, replays, tgt['z'], tgt['tx'], tab.ids, man, T, keep_scores=True)
        lut = {s: i for i, s in enumerate(tab.ids)}
        yq = np.asarray(tgt['tx'])[[lut[i] for i in man['query_ids']]]
        np.savez(out / f'{task["task"]}_rx{rx}_scores.npz', y_query=yq,
                 **{f'{n}|{m}|{k}': v for (n, m, k), v in scores.items()})
        res['targets'][str(rx)] = dict(manifest_sha256=sha, not_applicable=na,
                                       anchors=anchor_check(scores, matrix['anchor_pairs']),
                                       flips=metric_layer_flip_analysis(scores, yq, *matrix['stability']['flip_analysis']))
        write_csv(out / f'{task["task"]}_rx{rx}_center_errors.csv',
                  metric_layer_center_errors(b, tgt['z'], tgt['tx'], tab.ids, man, matrix['stability']['center_errors_budgets']))
        rows_all += [dict(r, task=task['task']) for r in rows]
    st = dict(loro=DS.loro_stability(src['z'], src['tx'], src['rx'], T),
              half_split=DS.half_split_stability(src['z'], src['tx'], src['rx'], src['position'], T),
              lags=DS.lag_correlation(src['z'], src['tx'], src['rx'], src['position'], matrix['stability']['lags']))
    if task['dataset'] == 'wisig':
        st['date_decomposition'] = DS.date_decomposition(src['z'], src['tx'], src['rx'], src['date'])
    res['stability'] = st
    res['seconds'] = time.time() - t0
    write_csv(out / f'{task["task"]}_rows.csv', rows_all)
    write_json(out / f'{task["task"]}_diagnostics.json', res)
    return rows_all, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--draws', type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out) if a.synthetic else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    matrix = json.loads((CONFIG / 'diagnostic_matrix.json').read_text())
    if a.synthetic:
        for task, src, tgt in synthetic_tasks():
            rows, res = diagnose_task(task, src, tgt, matrix, 10, out, a.draws)
            bad = {rx: {k: v for k, v in t['anchors'].items() if not v['ok']} for rx, t in res['targets'].items()}
            print(task['task'], len(rows), 'rows; anchor failures:', {k: v for k, v in bad.items() if v})
        write_json(out / 'mode.json', dict(mode='SYNTHETIC_DRY_RUN (no real data, not evidence)'))
        return
    reg = Registry(CONFIG)
    res = reg.reservation()
    if not (res.get('active') and res.get('stage') == 'D1'):
        sys.exit('D1 reservation is not active; nothing was read')
    from rffi_post_a.artifact_guard import ArtifactGuard
    cache = ArtifactGuard(ROOT / 'outputs/rffi_post_a/D1_cache')
    try:
        for task in matrix['tasks']:
            T = reg.num_classes(task['dataset'])
            mans = cache.load_meta(f'{task["task"]}_source')['manifest_sha256']
            _, src = cache.open_features(f'{task["task"]}_source', task, res, mans, reg.roles)
            _, tgt = cache.open_features(f'{task["task"]}_target', task, res, mans, reg.roles)
            diagnose_task(task, src, tgt, matrix, T, out, a.draws, expected_manifests=mans)
            print(task['task'], 'done', flush=True)
    finally:
        reg_path = CONFIG / 'registration.json'
        r = json.loads(reg_path.read_text())
        r['reservation'] = dict(r['reservation'], active=False, closed_after='D1')
        reg_path.write_text(json.dumps(r, indent=2))
        write_json(out / 'reservation_closed.json', dict(stage='D1', closed=True, time=time.time()))


if __name__ == '__main__':
    main()
