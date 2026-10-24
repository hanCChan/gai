#!/usr/bin/env python3
"""R0 runner: exactly the six registered fits, then the fixed evaluation (plan §7.4-7.6).

Refuses unless configs/rffi_post_a/registration.json has an ACTIVE R0 reservation whose allowlist is
exactly the six pilot jobs, max_fits 6, cap <= 1.5 A100 GPUh, no retries and no final/development
access. GPU time = sum over occupied cards of wall time (CUDA synchronised in the fits). Exceeding the
cap stops everything; failed fits are never retried; no seventh fit, no extra epochs.

    python scripts/rffi_post_a/run_pilot.py --gpus 0 1 2 3
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from rffi_calibration.queue import JobQueue  # noqa: E402
from rffi_calibration.registry import Registry, file_sha  # noqa: E402

CONFIG = ROOT / 'configs/rffi_post_a'
OUT = ROOT / 'reports/rffi_post_a/R0'
FITS = ROOT / 'outputs/rffi_post_a'


def check_reservation(res, matrix):
    jobs = {j['job_id'] for j in matrix['jobs']}
    problems = []
    if not res.get('active') or res.get('stage') != 'R0':
        problems.append('R0 reservation inactive')
    if set(res.get('allowed_job_ids', [])) != jobs or len(jobs) != 6:
        problems.append('allowlist is not exactly the six registered R0 jobs')
    if res.get('max_fits') != 6:
        problems.append('max_fits != 6')
    if not 0 < float(res.get('max_reserved_gpu_hours', 0)) <= 1.5:
        problems.append('GPU cap missing or > 1.5 h')
    if res.get('allow_failed_fit_retry') or res.get('allow_final_numeric_access') or res.get('allow_development_numeric_access'):
        problems.append('retry/final/development access must be false')
    if problems:
        raise SystemExit('; '.join(problems) + '; nothing was run')


def verify_fit(path, job):
    import torch
    st = json.loads((path / 'status.json').read_text())
    curve = json.loads((path / 'curve.json').read_text())
    assert st['state'] == 'completed' and st['job']['job_id'] == job['job_id']
    assert file_sha(path / 'last.pt') == st['last_sha256'] and file_sha(path / 'native_best.pt') == st['native_best_sha256']
    assert curve[-1]['step'] == st['job']['steps'] and all(np.isfinite(c['val_ce']) for c in curve)
    for name in ('last.pt', 'native_best.pt'):
        obj = torch.load(path / name, map_location='cpu', weights_only=False)
        assert all(bool(torch.isfinite(v).all()) for v in obj['model'].values())
    return st


def cross_arm_checks(statuses, matrix):
    """Same fold: identical initial state SHA and identical all_ids chain across C_MATCH/P_MIX/P_RX."""
    out = {}
    for split in sorted({j['split'] for j in matrix['jobs']}):
        ss = [statuses[j['job_id']] for j in matrix['jobs'] if j['split'] == split and j['job_id'] in statuses]
        out[split] = dict(n=len(ss), same_init=len({s['init_state_sha256'] for s in ss}) == 1,
                          same_all_ids=len({s['all_ids_chain_sha256'] for s in ss}) == 1)
    return out


def evaluate_fit(reg, res, job, ckpt_name, device, matrix, *, config_dir=None, fit_root=None, report_root=None):
    """Fixed heads on the frozen encoder; Stage A manifests rebuilt from metadata and matched."""
    import torch
    from rffi_calibration.access import AccessGuard
    from rffi_calibration.datasets import wisig
    from rffi_calibration.encoders import build_wisig_official, extract
    from rffi_calibration.evaluator import FeatureSet, evaluate_target, fit_heads
    from rffi_calibration.heads import (LegacyP2EProto, NativeHead, NIWPredictive, SimpleCNAPSFrozen, SimpleShotCL2N,
                                        TargetPrototype)
    from rffi_calibration.sampling import RecordTable, build_manifest, manifest_sha
    config_dir, fit_root, report_root = config_dir or CONFIG, fit_root or FITS, report_root or OUT
    diag = json.loads((config_dir / 'diagnostic_matrix.json').read_text())
    prefixes = next(t for t in diag['tasks'] if t['dataset'] == 'wisig' and
                    t['task'].startswith(f'A_wisig_{job["split"]}'))['manifest_sha256_prefix']
    guard = AccessGuard(reg, res, log_path=report_root / 'access_log.jsonl', stage=res['stage'] + '-eval')
    fold = reg.roles['wisig']['folds'][job['split']]
    obj = torch.load(fit_root / job['job_id'] / ckpt_name, map_location='cpu', weights_only=False)
    w = build_wisig_official(obj['model'], classes=10)
    st, xs = wisig.load(reg, guard, fold['fit'], [0, 1], (0, 120), numeric=True, purpose='R0 eval source')
    tt, xt = wisig.load(reg, guard, fold['source_selection'], [0, 1], (0, 200), numeric=True, purpose='R0 eval target')
    heads = [NativeHead(*w.linear_head()), TargetPrototype('euclid'), TargetPrototype('pooled'), LegacyP2EProto(),
             SimpleShotCL2N(), SimpleCNAPSFrozen(), NIWPredictive(1.)]
    fitted, skipped = fit_heads(heads, FeatureSet(extract(w, xs, device=device), st, {}), 10)
    zt = extract(w, xt, device=device)
    rows, logits, mans = [], {}, {}
    sq = json.loads((ROOT / 'configs/rffi_calibration/registration.json').read_text())['support_query']['wisig']
    for rx in fold['source_selection']:
        m = tt.rx == rx
        sub = RecordTable('wisig', tt.tx[m], tt.rx[m], tt.date[m], tt.position[m], 'cache_position', capture=tt.capture[m])
        man = build_manifest(sub, rx, 10, sq['rule'], q_per_class=sq['q_per_class'], n_draws=20)
        sha = manifest_sha(man)
        if not sha.startswith(prefixes[str(rx)]):
            raise ValueError(f'RX {rx}: manifest differs from Stage A')
        mans[str(rx)] = sha
        r, lg = evaluate_target(fitted, FeatureSet(zt[m], sub, {}), man, 10, keep_logits=True)
        rows += [dict(x, job_id=job['job_id'], arm=job['arm'], split=job['split'], checkpoint=ckpt_name,
                      seed=job['seed']) for x in r]
        q = sub.index_of(man['query_ids'])
        logits.update({f'{rx}|{h}|{mm}|{k}': v for (h, mm, k), v in lg.items()})
        logits[f'{rx}|y_query'] = sub.tx[q]
    return rows, logits, dict(skipped=skipped, manifests=mans, checkpoint_sha256=file_sha(fit_root / job['job_id'] / ckpt_name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    a = ap.parse_args()
    reg = Registry(CONFIG)
    res = reg.reservation()
    matrix = json.loads((CONFIG / 'pilot_matrix.json').read_text())
    check_reservation(res, matrix)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'queue.sqlite').exists():
        raise SystemExit('existing R0 queue: no automatic rerun/resume')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    code = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT / 'src').glob('rffi_*/**/*.py')))).hexdigest()
    queue = JobQueue(OUT / 'queue.sqlite')
    for j in matrix['jobs']:
        queue.add(job_id=j['job_id'], stage='R0', method_id=j['arm'], dataset_id='wisig', source_split=j['split'],
                  seed=j['seed'], config_sha=reg.fingerprint(), code_commit=commit, data_manifest_sha=code,
                  predicted_gpu_seconds=j['predicted_gpu_seconds'])
    cap = float(res['max_reserved_gpu_hours']) * 3600.
    spent, running, tasks, statuses, stopped = 0., {}, [], {}, False
    checks, evals = None, []

    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = dict(os.environ, RFFI_POST_QUEUE=str(OUT / 'queue.sqlite'), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
    try:
        while True:
            live = sum(time.monotonic() - r['start'] for r in running.values())
            if spent + live >= cap - 10:
                stopped = True
            for gpu, r in list(running.items()):
                if stopped and r['proc'].poll() is None:
                    r['proc'].terminate()
                rc = r['proc'].poll()
                if rc is None:
                    continue
                sec = time.monotonic() - r['start']
                spent += sec
                r['log'].close()
                path = FITS / r['job']['job_id']
                ok, err = False, ''
                if rc == 0:
                    try:
                        statuses[r['job']['job_id']] = verify_fit(path, r['job'])
                        ok = True
                    except Exception as e:                      # noqa: BLE001 - recorded, never retried
                        err = repr(e)
                if stopped and rc != 0:
                    queue.cancel(r['job']['job_id'], reason='budget/interrupt', gpu_reserved_seconds=sec, exit_code=rc)
                    state = 'cancelled'
                else:
                    state = queue.complete(r['job']['job_id'], rc, (path / 'last.pt').exists(), ok,
                                           gpu_reserved_seconds=sec, wall_seconds=sec, artifact_path=str(path))
                tasks.append(dict(job_id=r['job']['job_id'], state=state, exit_code=rc, error=err, gpu=gpu, seconds=sec))
                del running[gpu]
            if not stopped:
                for gpu in a.gpus:
                    if gpu in running:
                        continue
                    claim = queue.claim(gpu, res)
                    if claim is None:
                        continue
                    j = next(x for x in matrix['jobs'] if x['job_id'] == claim['job_id'])
                    log = (OUT / f'{j["job_id"]}.log').open('x')
                    proc = subprocess.Popen([sys.executable, 'scripts/rffi_post_a/train_pilot.py', '--job', j['job_id']],
                                            cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log,
                                            stderr=subprocess.STDOUT)
                    running[gpu] = dict(proc=proc, start=time.monotonic(), job=j, log=log)
            if not running:
                break
            time.sleep(1)
        checks = cross_arm_checks(statuses, matrix)
        if not stopped and len(statuses) == 6 and all(v['same_init'] and v['same_all_ids'] for v in checks.values()):
            import torch
            for j in matrix['jobs']:
                for ck in ('last.pt', 'native_best.pt'):
                    t0 = time.monotonic()
                    if spent + 120 > cap:
                        stopped = True
                        break
                    rows, logits, meta = evaluate_fit(reg, res, j, ck, 'cuda:0' if torch.cuda.is_available() else 'cpu', matrix)
                    d = OUT / 'evaluation' / f'{j["job_id"]}_{ck[:-3]}'
                    d.mkdir(parents=True, exist_ok=False)
                    from rffi_post_a.reporting import write_csv, write_json
                    write_csv(d / 'rows.csv', rows)
                    np.savez(d / 'scores.npz', **logits)
                    write_json(d / 'meta.json', meta)
                    sec = time.monotonic() - t0
                    spent += sec
                    evals.append(dict(job_id=j['job_id'], checkpoint=ck, seconds=sec))
    finally:
        for r in running.values():
            if r['proc'].poll() is None:
                r['proc'].terminate()
                r['proc'].wait()
            sec = time.monotonic() - r['start']
            queue.cancel(r['job']['job_id'], gpu_reserved_seconds=sec, exit_code=r['proc'].returncode)
            spent += sec
        queue.export_jsonl(OUT / 'jobs.jsonl')
        r = json.loads((CONFIG / 'registration.json').read_text())
        r['reservation'] = dict(r['reservation'], active=False, closed_after='R0')
        (CONFIG / 'registration.json').write_text(json.dumps(r, indent=2))
        (OUT / 'execution_ledger.json').write_text(json.dumps(dict(
            fits=tasks, evaluations=evals, cross_arm=checks, reserved_gpu_hours=spent / 3600.,
            cap_gpu_hours=cap / 3600., stopped=stopped, reservation_closed=True, commit=commit), indent=1, default=str))


if __name__ == '__main__':
    main()
