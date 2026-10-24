#!/usr/bin/env python3
"""D1 feature extraction (plan §6.1): six Stage A T0 checkpoints, source fit records and the original
held-RX targets, cached once under outputs/rffi_post_a/D1_cache with lineage metadata.

Fail closed: the D1 reservation (configs/rffi_post_a/registration.json) must be active and list the
task; checkpoint SHA (status.json + file hash), manifest SHAs (rebuilt from metadata and matched to the
registered Stage A prefixes) and roles are verified BEFORE the checkpoint is deserialised or any
signal member is opened.

    python scripts/rffi_post_a/cache_features.py --device cpu
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from rffi_calibration.access import AccessGuard  # noqa: E402
from rffi_calibration.registry import Registry, file_sha  # noqa: E402
from rffi_calibration.sampling import build_manifest, manifest_sha  # noqa: E402
from rffi_post_a.artifact_guard import ArtifactGuard, verify_lineage  # noqa: E402

CONFIG = ROOT / 'configs/rffi_post_a'
STAGE_A_OUT = Path('/workspace/gai/branches/rffi-budget-calibration-v1/outputs/rffi_calibration')
LAYER = {'wisig': 'OfficialCNN.encode 80-D pre-dropout', 'shen': 'FeatureCNN classifier input 128-D'}


def code_sha():
    h = hashlib.sha256()
    for p in sorted((ROOT / 'src').glob('rffi_*/**/*.py')):
        h.update(p.read_bytes())
    return h.hexdigest()


def manifests_for(reg, guard, task):
    """Rebuild the Stage A target manifests from METADATA (labels/positions) only."""
    ds = task['dataset']
    sq = json.loads((ROOT / 'configs/rffi_calibration/registration.json').read_text())['support_query'][ds]
    T = reg.num_classes(ds)
    out = {}
    for rx in task['held_rx']:
        if ds == 'wisig':
            from rffi_calibration.datasets import wisig
            tab, _ = wisig.load(reg, guard, [rx], [0, 1], (0, 200), numeric=False, purpose='D1 manifest rebuild')
        else:
            from rffi_calibration.datasets import shen
            tab, _ = shen.load(reg, guard, [rx], 'all', numeric=False, purpose='D1 manifest rebuild')
        man = build_manifest(tab, rx, T, sq['rule'], q_per_class=sq['q_per_class'], n_draws=20)
        out[str(rx)] = (manifest_sha(man), man)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--stage-a-outputs', default=str(STAGE_A_OUT))
    a = ap.parse_args()
    reg = Registry(CONFIG)
    res = reg.reservation()
    matrix = json.loads((CONFIG / 'diagnostic_matrix.json').read_text())
    if not (res.get('active') and res.get('stage') == 'D1' and res.get('allow_feature_extraction')):
        sys.exit('D1 reservation is not active; nothing was read')
    guard = AccessGuard(reg, res, log_path=ROOT / 'reports/rffi_post_a/D1/access_log.jsonl', stage='D1')
    cache = ArtifactGuard(ROOT / 'outputs/rffi_post_a/D1_cache')
    import torch
    from rffi_calibration.encoders import build_shen_featurecnn, build_wisig_official, extract
    ledger = []
    for task in matrix['tasks']:
        t0 = time.time()
        ck = Path(a.stage_a_outputs) / task['task']
        status = json.loads((ck / 'status.json').read_text())
        mans = manifests_for(reg, guard, task)
        verify_lineage(dict(checkpoint_sha256=status['checkpoint_sha256']), None,
                       {k: v[0] for k, v in mans.items()}, res, reg.roles, task)
        if file_sha(ck / 'selected.pt') != task['checkpoint_sha256']:
            sys.exit(f'{task["task"]}: checkpoint bytes differ from the registered SHA; stopped')
        obj = torch.load(ck / 'selected.pt', map_location='cpu', weights_only=False)
        ds = task['dataset']
        if ds == 'wisig':
            from rffi_calibration.datasets import wisig
            w = build_wisig_official(obj['model'], classes=10)
            st, xs = wisig.load(reg, guard, task['fit_rx'], task['fit_dates'], tuple(task['fit_positions']),
                                numeric=True, purpose='D1 source')
            tt, xt = wisig.load(reg, guard, task['held_rx'], [0, 1], (0, 200), numeric=True, purpose='D1 target')
            scaler_sha = 'none'
        else:
            from rffi_calibration.datasets import shen
            w = build_shen_featurecnn(obj['model'])
            st, xs = shen.load(reg, guard, task['fit_rx'], 'fit', numeric=True, scaler=obj['scaler'], purpose='D1 source')
            tt, xt = shen.load(reg, guard, task['held_rx'], 'all', numeric=True, scaler=obj['scaler'], purpose='D1 target')
            scaler_sha = hashlib.sha256(json.dumps(obj['scaler'], sort_keys=True, default=str).encode()).hexdigest()
        for role, tab, x in (('source', st, xs), ('target', tt, xt)):
            z = extract(w, x, device=a.device).astype(np.float64)
            meta = dict(checkpoint_sha256=task['checkpoint_sha256'], encoder_layer=LAYER[ds], scaler_sha256=scaler_sha,
                        source_asset_sha256='see reports/rffi_calibration/stage_A_execution/input_integrity.json',
                        dataset=ds, rx=sorted(map(str, set(tab.rx))), dates=sorted(set(map(int, tab.date))),
                        positions=task.get('fit_positions', 'fit partition') if role == 'source' else 'all',
                        manifest_sha256={k: v[0] for k, v in mans.items()}, code_sha256=code_sha(), dtype='float64',
                        role=role, task=task['task'])
            cache.write(f'{task["task"]}_{role}', meta, z=z, tx=tab.tx, rx=np.asarray(tab.rx).astype(str),
                        date=tab.date, position=tab.position, ids=tab.ids)
        ledger.append(dict(task=task['task'], seconds=time.time() - t0, device=a.device))
        print(json.dumps(ledger[-1]), flush=True)
    (ROOT / 'reports/rffi_post_a/D1').mkdir(parents=True, exist_ok=True)
    (ROOT / 'reports/rffi_post_a/D1/extraction_ledger.json').write_text(json.dumps(ledger, indent=1))


if __name__ == '__main__':
    main()
