#!/usr/bin/env python3
"""D0: observation audit of EXISTING Stage A evidence (plan §5). Reads only the published summary
CSV/JSON files and the six training curve.json/status.json files. Never opens selected.pt, signal
arrays, features or the per-query predictions_*.npz score files (score replay awaits approval).

    python scripts/rffi_post_a/audit_existing.py --out reports/rffi_post_a/D0
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from rffi_post_a import reporting as R  # noqa: E402

STAGE_A = ROOT / 'reports/rffi_calibration/stage_A'
DEFAULT_OUTPUTS = Path('/workspace/gai/branches/rffi-budget-calibration-v1/outputs/rffi_calibration')
FORBIDDEN_SUFFIXES = ('.pt', '.npz', '.npy', '.h5', '.zip', '.dat')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_checked(p, opened):
    p = Path(p)
    if p.suffix in FORBIDDEN_SUFFIXES:
        raise PermissionError(f'D0 may not open {p}')
    opened.append(dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='reports/rffi_post_a/D0')
    ap.add_argument('--outputs-root', default=str(DEFAULT_OUTPUTS))
    a = ap.parse_args()
    out = ROOT / a.out
    t0 = time.time()
    opened = []
    tables = {}
    for ds in ('wisig', 'shen'):
        rows = R.read_rows(read_checked(STAGE_A / f'{ds}_rows.csv', opened))
        tables[ds] = R.per_rx_table(rows, ds)
        replay = R.replay_aggregate(rows, read_checked(STAGE_A / f'{ds}_aggregate.csv', opened))
        R.write_json(out / f'{ds}_aggregate_replay.json', replay)
        R.write_csv(out / f'{ds}_per_rx.csv', tables[ds])
        R.write_json(out / f'{ds}_fixed_differences.json', R.fixed_differences(tables[ds]))
        R.write_json(out / f'{ds}_budget_gains.json', R.budget_gains(tables[ds]))
        if ds == 'wisig':
            R.write_json(out / 'wisig_h5c_members.json', R.h5c_members(tables[ds]))
        for split, rx in sorted({(r['split'], r['rx']) for r in tables[ds]}):
            series = {h: [(r['m'], 100 * r['acc_mean']) for r in tables[ds]
                          if r['rx'] == rx and r['split'] == split and r['head'] == h and r['m'] > 0]
                      for h in R.CURVE_HEADS}
            R.svg_lines({k: v for k, v in series.items() if v},
                        f'{ds} {split} RX {rx}: 20 draws/budget, development data, T0 seed 20261013',
                        out / 'figures' / f'{ds}_{split.replace("/", "-")}_rx{rx}.svg')
    curves = []
    root = Path(a.outputs_root)
    for d in sorted(root.glob('A_*')):
        curve = json.loads(read_checked(d / 'curve.json', opened).read_text())
        status = json.loads(read_checked(d / 'status.json', opened).read_text())
        curves.append(R.curve_audit(curve, status))
        R.svg_lines({'val CE': [(c['step'], c['val_ce']) for c in curve]}, f'{d.name}: fit-RX held-position native CE',
                    out / 'figures' / f'curve_{d.name}.svg', xlabel='step', ylabel='val CE')
    R.write_json(out / 'curve_audit.json', curves)
    decision = json.loads(read_checked(STAGE_A / 'decision.json', opened).read_text())
    R.write_json(out / 'inputs_opened.json', opened)
    R.write_json(out / 'commands_and_environment.json', dict(
        command=' '.join(sys.argv), python=sys.version, platform=platform.platform(), seconds=time.time() - t0,
        stage_a_decision=decision['decision']))
    R.write_json(out / 'access_audit.json', dict(
        signal_arrays='not_opened', features='not_opened', weights='not_opened',
        per_query_score_files='not_opened (score replay awaits approval)',
        opened_file_types=sorted({Path(o['path']).suffix for o in opened}), n_files=len(opened)))
    print(json.dumps(dict(files=len(opened), curves=len(curves), out=str(out)), indent=1))


if __name__ == '__main__':
    main()
