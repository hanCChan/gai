#!/usr/bin/env python3
"""Run exactly the activated six-fit Stage A, with a wall-clock GPU budget and no retries."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from rffi_calibration.queue import JobQueue
from rffi_calibration.registry import Registry, file_sha


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def verify_fit(path, job, fingerprint):
    status = json.loads((path / 'status.json').read_text())
    curve = json.loads((path / 'curve.json').read_text())
    import numpy as np
    import torch
    obj = torch.load(path / 'selected.pt', map_location='cpu', weights_only=False)
    assert status['state'] == 'completed' and status['job'] == job and obj['job'] == job
    assert status['registry_fingerprint'] == fingerprint
    assert file_sha(path / 'selected.pt') == status['checkpoint_sha256']
    assert curve[-1]['step'] == job['steps'] and all(np.isfinite(r['val_ce']) for r in curve)
    best = min(curve, key=lambda r: r['val_ce'])
    assert best['step'] == obj['step'] == status['best']['step']
    assert best['val_ce'] == obj['val_ce'] == status['best']['ce']
    assert all(bool(torch.isfinite(v).all()) for v in obj['model'].values())
    return status


def verify_inputs(reg, out):
    """Hash only the registered source assets, before any fitting claim."""
    from rffi_calibration.access import AccessGuard
    from rffi_calibration.datasets import shen, wisig
    guard = AccessGuard(reg, log_path=out / 'input_access.jsonl', stage='A-input-integrity')
    paths = []
    for rx in reg.roles['wisig']['source']:
        for date in (0, 1):
            p = wisig.cache_path(reg, rx, date)
            guard.check('wisig', p, rx=rx, date=date, purpose='source integrity hash')
            paths.append(p)
    for rx in reg.roles['shen']['source']:
        d = shen.cache_dir(reg, rx)
        for name in ('strict_cis.npy', 'labels.npy', 'rows.npy', 'manifest.json'):
            p = d / name
            guard.check('shen', p, rx=rx, purpose='source integrity hash')
            paths.append(p)
    historical = json.loads(reg.path('wisig', 'registration').read_text())
    expected = {}

    def visit(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                expected[str(Path(value['path']).resolve())] = value['sha256']
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(historical)
    records = []
    for p in paths:
        exp = expected.get(str(p.resolve()))
        actual = file_sha(p)
        records.append(dict(path=str(p), bytes=p.stat().st_size, expected=exp, actual=actual,
                            matched=exp is not None and exp == actual))
    write_json(out / 'input_integrity.json', records)
    if not all(r['matched'] for r in records):
        raise ValueError('source input hash mismatch or missing historical hash; no fits started')
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='reports/rffi_calibration/stage_A_execution')
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    reg = Registry()
    res = reg.reservation()
    matrix = json.loads((ROOT / 'configs/rffi_calibration/formal_matrix.json').read_text())
    jobs = matrix['jobs']
    assert res['active'] and res['stage'] == 'A'
    assert set(res['allowed_job_ids']) == {j['job_id'] for j in jobs}
    assert len(jobs) == res['max_fits'] == 6 and all(j['arm'] == 'T0' for j in jobs)
    assert res['max_reserved_gpu_hours'] <= 1.5 and not res['allow_failed_fit_retry']
    assert not res['allow_final_numeric_access']
    assert not (out / 'queue.sqlite').exists(), 'existing queue: no automatic rerun/resume'
    data_sha = verify_inputs(reg, out)
    queue = JobQueue(out / 'queue.sqlite')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for j in jobs:
        queue.add(job_id=j['job_id'], stage='A', method_id=j['arm'], dataset_id=j['dataset'],
                  source_split=j['split'], seed=j['seed'], config_sha=reg.fingerprint(), code_commit=commit,
                  data_manifest_sha=data_sha, predicted_gpu_seconds=j['predicted_gpu_seconds'])
    queue.export_jsonl(out / 'jobs.jsonl')
    running, tasks, spent = {}, [], 0.
    cap = res['max_reserved_gpu_hours'] * 3600.
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env_base = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    RFFI_QUEUE=str(out / 'queue.sqlite'), PYTHONUNBUFFERED='1')
    try:
        while True:
            elapsed = sum(time.monotonic() - r['start'] for r in running.values())
            if spent + elapsed >= cap - 10:
                stopped = True
            for gpu, r in list(running.items()):
                if stopped and r['proc'].poll() is None:
                    r['proc'].terminate()
                rc = r['proc'].poll()
                if rc is None:
                    continue
                seconds = time.monotonic() - r['start']
                spent += seconds
                r['log'].close()
                job = r['job']
                path = ROOT / 'outputs/rffi_calibration' / job['job_id']
                verified, error, status = False, '', {}
                if rc == 0:
                    try:
                        status = verify_fit(path, job, reg.fingerprint())
                        verified = True
                    except Exception as e:
                        error = repr(e)
                if stopped and rc != 0:
                    queue.cancel(job['job_id'], reason='interrupt or budget stop',
                                 gpu_reserved_seconds=seconds, wall_seconds=seconds,
                                 exit_code=rc, artifact_path=str(path))
                    state = 'cancelled'
                else:
                    state = queue.complete(job['job_id'], rc, (path / 'selected.pt').exists(), verified,
                                           gpu_reserved_seconds=seconds, wall_seconds=seconds,
                                           peak_memory_mb=status.get('peak_memory_mb'), artifact_path=str(path))
                tasks.append(dict(job_id=job['job_id'], state=state, exit_code=rc,
                                  verifier_error=error, gpu=gpu, gpu_reserved_seconds=seconds))
                print(json.dumps(tasks[-1]), flush=True)
                del running[gpu]
                queue.export_jsonl(out / 'jobs.jsonl')
            if not stopped:
                for gpu in a.gpus:
                    if gpu in running:
                        continue
                    elapsed = sum(time.monotonic() - r['start'] for r in running.values())
                    if spent + elapsed >= cap - 10:
                        stopped = True
                        break
                    pending_wisig = [j['job_id'] for j in jobs if j['dataset'] == 'wisig'
                                     and queue.get(j['job_id'])['status'] == 'pending']
                    claim_res = dict(res, allowed_job_ids=pending_wisig) if gpu < 2 and pending_wisig else res
                    claim = queue.claim(gpu, claim_res)
                    if claim is None:
                        continue
                    j = next(j for j in jobs if j['job_id'] == claim['job_id'])
                    log = (out / f'{j["job_id"]}.log').open('x')
                    start = time.monotonic()
                    proc = subprocess.Popen([sys.executable, 'scripts/rffi_calibration/train.py', '--job', j['job_id']],
                                            cwd=ROOT, env=dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu)),
                                            stdout=log, stderr=subprocess.STDOUT)
                    running[gpu] = dict(proc=proc, start=start, job=j, log=log)
                    print(f'claimed GPU {gpu}: {j["job_id"]}', flush=True)
                queue.export_jsonl(out / 'jobs.jsonl')
            if not running:
                break
            write_json(out / 'progress.json', dict(completed=tasks, spent_gpu_seconds=spent,
                                                  running=list(running), stopped=stopped))
            time.sleep(1)
    finally:
        for r in running.values():
            if r['proc'].poll() is None:
                r['proc'].terminate()
                try:
                    r['proc'].wait(timeout=20)
                except subprocess.TimeoutExpired:
                    r['proc'].kill()
                    r['proc'].wait()
            seconds = time.monotonic() - r['start']
            queue.cancel(r['job']['job_id'], gpu_reserved_seconds=seconds,
                         wall_seconds=seconds, exit_code=r['proc'].returncode)
            spent += seconds
            tasks.append(dict(job_id=r['job']['job_id'], state='cancelled',
                              exit_code=r['proc'].returncode, gpu_reserved_seconds=seconds))
            r['log'].close()
        queue.export_jsonl(out / 'jobs.jsonl')
        write_json(out / 'execution.json', dict(tasks=tasks, reserved_gpu_hours=spent / 3600.,
                                               max_gpu_hours=cap / 3600., stopped=stopped,
                                               registry_fingerprint=reg.fingerprint(), data_manifest_sha=data_sha))
    if len(tasks) != 6 or any(t['state'] != 'completed' for t in tasks):
        sys.exit('Stage A incomplete; failures retained, no retries')


if __name__ == '__main__':
    main()
