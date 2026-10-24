#!/usr/bin/env python3
"""Metadata-only preflight (E0/E3). Never reads a signal value.

Checks: config fingerprints and role consistency; presence/size of every registered data asset
(os.stat only for sealed RX); WiSig/Shen label/position coverage of SOURCE RX (labels only);
OSU sigmf sidecars and window plans; inventory of checkpoints usable for Stage A; reservation state.

    python scripts/rffi_calibration/preflight.py --out reports/rffi_calibration/E0_preflight
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src')]
from rffi_calibration.access import AccessGuard  # noqa: E402
from rffi_calibration.registry import Registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    reg = Registry()
    guard = AccessGuard(reg, log_path=Path(a.out) / 'access_log.jsonl', stage='E0')
    rep = dict(registry_fingerprint=reg.fingerprint(), reservation=reg.reservation(), problems=[])
    P = rep['problems']

    # --- WiSig: file presence for every non-final RX x date; final RX must have no cache file
    w = reg.roles['wisig']
    from rffi_calibration.datasets import wisig
    present = {}
    for r in w['source'] + w['development'] + w['final_reserved']:
        for d in (0, 1, 2, 3):
            p = wisig.cache_path(reg, r, d)
            present[f'{r}/{d}'] = p.exists()
    rep['wisig_cache_files'] = sum(present.values())
    leaked = [k for k, v in present.items() if v and (int(k.split('/')[0]) in w['final_reserved'] or k.endswith('/3'))]
    if leaked:
        P.append(f'WiSig cache contains reserved RX/date files: {leaked}')
    cov = wisig.coverage(reg, guard, w['source'], [0, 1])
    rep['wisig_source_min_cell'] = int(min(min(v) for v in cov.values()))
    rep['wisig_source_cells_lt_200'] = [k for k, v in cov.items() if min(v) < 200]
    archive = reg.path('wisig', 'archive')
    rep['wisig_archive_bytes'] = archive.stat().st_size if archive.exists() else None

    # --- Shen: source caches
    from rffi_calibration.datasets import shen
    s = reg.roles['shen']
    rep['shen_cache'] = {}
    for r in s['source']:
        d = shen.cache_dir(reg, r)
        ok = (d / 'manifest.json').exists() and (d / 'strict_cis.npy').exists()
        n = None
        if ok:
            t, _ = shen.load(reg, guard, [r], 'all')
            n = np.bincount(t.tx, minlength=10).tolist()
        rep['shen_cache'][r] = dict(present=ok, per_tx=n)
        if not ok:
            P.append(f'Shen cache missing for source RX {r}')
    root = reg.resolve(reg.dataset('shen')['root'])
    rep['shen_root_h5'] = sorted(p.name for p in root.glob('*.h5'))
    for r in s['development'] + s['final_reserved']:
        rep.setdefault('shen_uncached_present_h5', {})[r] = sorted(p.name for p in root.glob(f'{r}_*.h5'))

    # --- OSU: sidecars + block plan
    from rffi_calibration.datasets import osu
    rep['osu'] = {}
    for rx in ('RX1', 'RX2'):
        try:
            t = osu.planned_table(reg, guard, rx)
            rep['osu'][rx] = {role: int((t.split_role == role).sum()) for role in ('fit', 'support', 'query')}
            rep['osu'][rx]['per_tx_windows'] = int(len(t) // 25)
        except Exception as e:                                    # noqa: BLE001 - report, do not hide
            P.append(f'OSU {rx}: {e}')

    # --- checkpoints usable for Stage A
    rr = reg.vars['RETRAIN_ROOT']
    search = sorted(Path(rr, 'outputs/receiver_p2_wisig/search').glob('*/best.pt'))
    rep['legacy_wisig8_checkpoints'] = [str(p.parent.name) for p in search]
    rep['new_checkpoints'] = sorted(str(p.parent.name) for p in (ROOT / 'outputs/rffi_calibration').glob('*/status.json'))
    rep['shen_checkpoints_found'] = sorted(str(p) for p in Path(rr, 'outputs').glob('**/selected.pt'))[:20]
    matrix = json.loads((ROOT / 'configs/rffi_calibration/formal_matrix.json').read_text())
    rep['stage_A_jobs_registered'] = [j['job_id'] for j in matrix['jobs']]
    rep['stage_A_predicted_gpu_hours'] = round(sum(j['predicted_gpu_seconds'] for j in matrix['jobs']) / 3600, 3)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'preflight.json').write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ('wisig_cache_files', 'wisig_source_min_cell', 'osu', 'legacy_wisig8_checkpoints',
                                          'new_checkpoints', 'stage_A_predicted_gpu_hours', 'problems')}, indent=1))


if __name__ == '__main__':
    main()
