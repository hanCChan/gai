#!/usr/bin/env python3
"""RV1 write-once twelve-fit runner, with bounded fixed evaluations and reservation closure."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/rffi_post_a')]
from rffi_calibration.queue import JobQueue
from rffi_calibration.registry import Registry, file_sha
from rffi_post_a.reporting import write_csv, write_json
from rffi_post_a.variance import (CONFIG, FITS, OUT, check_reservation, cross_pair_checks,
                                 verify_preregistration)
from run_pilot import evaluate_fit, verify_fit


def preflight(reg, res):
    from rffi_calibration.access import AccessGuard
    from rffi_calibration.datasets import wisig
    from rffi_calibration.sampling import RecordTable, build_manifest, manifest_sha
    from rffi_post_a.episode_controls import ControlSchedule

    guard = AccessGuard(reg, res, OUT / 'access_log.jsonl', stage='RV1-metadata')
    diag = json.loads((CONFIG / 'diagnostic_matrix.json').read_text())
    sq = json.loads((ROOT / 'configs/rffi_calibration/registration.json').read_text())['support_query']['wisig']
    out = {}
    for split, fold in reg.roles['wisig']['folds'].items():
        if set(fold['fit']) & set(fold['source_selection']):
            raise ValueError('fit/held overlap')
        tr, _ = wisig.load(reg, guard, fold['fit'], [0, 1], (0, 120), purpose='RV1 source coverage')
        ControlSchedule(tr.tx, tr.rx, tr.date, 10, seed=0)
        tt, _ = wisig.load(reg, guard, fold['source_selection'], [0, 1], (0, 200), purpose='RV1 manifest')
        prefixes = next(t['manifest_sha256_prefix'] for t in diag['tasks']
                        if t['task'].startswith(f'A_wisig_{split}'))
        old_meta = json.loads((ROOT / 'reports/rffi_post_a/R0/evaluation' /
                              f'R0_wisig_{split}_C_MATCH_s20261024_last/meta.json').read_text())
        manifests = {}
        for rx in fold['source_selection']:
            m = tt.rx == rx
            sub = RecordTable('wisig', tt.tx[m], tt.rx[m], tt.date[m], tt.position[m],
                              'cache_position', capture=tt.capture[m])
            man = build_manifest(sub, rx, 10, sq['rule'], q_per_class=sq['q_per_class'], n_draws=20)
            sha = manifest_sha(man)
            if not sha.startswith(prefixes[str(rx)]) or sha != old_meta['manifests'][str(rx)]:
                raise ValueError(f'{split}/{rx}: Stage A manifest mismatch')
            manifests[str(rx)] = sha
        out[split] = dict(source_records=len(tr.tx), held_records=len(tt.tx), manifests=manifests)
    assets = {str(wisig.cache_path(reg, rx, day)): file_sha(wisig.cache_path(reg, rx, day))
              for rx in reg.roles['wisig']['source'] for day in (0, 1)}
    write_json(OUT / 'preflight.json', dict(ok=True, folds=out, asset_sha256=assets))
    print('RV1 metadata preflight passed: 28 cache hashes, eight unchanged held-RX manifests', flush=True)


def evaluation_worker(job_id, checkpoint):
    reg = Registry(CONFIG)
    matrix = json.loads((CONFIG / 'pilot_matrix.json').read_text())
    res = reg.reservation()
    check_reservation(res, matrix)
    verify_preregistration()
    jobs = {j['job_id']: j for j in matrix['jobs']}
    queue = JobQueue(OUT / 'queue.sqlite')
    if job_id not in jobs or not all(queue.get(j)['status'] == 'completed' for j in jobs):
        raise SystemExit('evaluation requires the entire completed RV1 fitting matrix')
    if checkpoint not in ('last.pt', 'native_best.pt'):
        raise SystemExit('unregistered checkpoint')
    path = OUT / 'evaluation' / f'{job_id}_{checkpoint[:-3]}'
    if path.exists():
        raise SystemExit('existing evaluation: no retry or overwrite')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('RV1 evaluation requires A100')
    st = json.loads((FITS / job_id / 'status.json').read_text())
    expected = st['last_sha256' if checkpoint == 'last.pt' else 'native_best_sha256']
    if file_sha(FITS / job_id / checkpoint) != expected:
        raise ValueError('checkpoint hash mismatch')
    rows, scores, meta = evaluate_fit(reg, res, jobs[job_id], checkpoint, 'cuda:0', matrix,
                                     config_dir=CONFIG, fit_root=FITS, report_root=OUT)
    path.mkdir(parents=True)
    write_csv(path / 'rows.csv', rows)
    np.savez(path / 'scores.npz', **scores)
    write_json(path / 'meta.json', meta)
    torch.cuda.synchronize()
    print(json.dumps(dict(job=job_id, checkpoint=checkpoint, rows=len(rows))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    ap.add_argument('--eval-job')
    ap.add_argument('--checkpoint')
    a = ap.parse_args()
    if a.eval_job:
        evaluation_worker(a.eval_job, a.checkpoint)
        return
    reg = Registry(CONFIG)
    res = reg.reservation()
    matrix = json.loads((CONFIG / 'pilot_matrix.json').read_text())
    check_reservation(res, matrix)
    verify_preregistration()
    if not a.gpus or len(set(a.gpus)) != len(a.gpus):
        raise SystemExit('unique GPU indices required')
    if OUT.exists() or FITS.exists():
        raise SystemExit('existing RV1 outputs: no automatic rerun/resume')
    gpu_info = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True)
    for gpu in a.gpus:
        row = next((x for x in gpu_info.splitlines() if int(x.split(',')[0]) == gpu), None)
        if row is None or 'A100' not in row or int(row.split(',')[-2]) > 0 or int(row.split(',')[-1]) > 0:
            raise SystemExit(f'GPU {gpu} is not an idle A100; nothing was read')
    OUT.mkdir(parents=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    spent, running, tasks, evals, statuses = 0., {}, [], [], {}
    stopped, error, checks = False, None, {}
    queue, worker = None, None
    cap = res['max_reserved_gpu_hours'] * 3600.
    process_start = time.monotonic()
    env = dict(os.environ, RFFI_POST_QUEUE=str(OUT / 'queue.sqlite'), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        preflight(reg, res)
        queue = JobQueue(OUT / 'queue.sqlite')
        for j in matrix['jobs']:
            queue.add(job_id=j['job_id'], stage='RV1', method_id=j['arm'], dataset_id='wisig',
                      source_split=j['split'], seed=j['seed'], config_sha=reg.fingerprint(),
                      code_commit=commit, data_manifest_sha=file_sha(OUT / 'preflight.json'),
                      predicted_gpu_seconds=j['predicted_gpu_seconds'])
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
                        st = verify_fit(path, r['job'])
                        if st['job']['steps'] != 4800:
                            raise ValueError('fixed horizon changed')
                        for k in ('split', 'arm', 'seed'):
                            if st['job'][k] != r['job'][k]:
                                raise ValueError(f'job {k} mismatch')
                        statuses[r['job']['job_id']] = st
                        ok = True
                    except Exception as exc:
                        err = repr(exc)
                state = queue.complete(r['job']['job_id'], rc, (path / 'last.pt').exists(), ok,
                                       gpu_reserved_seconds=sec, wall_seconds=sec, artifact_path=str(path))
                tasks.append(dict(job_id=r['job']['job_id'], state=state, exit_code=rc,
                                  error=err, gpu=gpu, seconds=sec))
                print(json.dumps(tasks[-1]), flush=True)
                del running[gpu]
                if not ok:
                    stopped = True
            if not stopped:
                for gpu in a.gpus:
                    if gpu in running:
                        continue
                    claim_res = dict(res, max_reserved_gpu_hours=(cap - 610) / 3600.)
                    claim = queue.claim(gpu, claim_res)
                    if claim is None:
                        continue
                    j = next(j for j in matrix['jobs'] if j['job_id'] == claim['job_id'])
                    log = (OUT / f'{j["job_id"]}.log').open('x')
                    start = time.monotonic()
                    proc = subprocess.Popen([sys.executable, 'scripts/rffi_post_a/train_pilot.py',
                                             '--stage', 'RV1', '--job', j['job_id']],
                                            cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                            stdout=log, stderr=subprocess.STDOUT)
                    running[gpu] = dict(proc=proc, start=start, job=j, log=log)
            if not running:
                break
            time.sleep(.5)
        checks = cross_pair_checks(statuses, matrix)
        if len(statuses) != 12 or not all(c['same_init'] and c['same_all_ids'] for c in checks.values()):
            stopped = True
        if not stopped:
            print('12/12 fits passed; six paired initialization/sample-chain checks passed', flush=True)
            for j in matrix['jobs']:
                for ck in ('last.pt', 'native_best.pt'):
                    if stopped or spent >= cap - 10:
                        raise RuntimeError('budget/interruption before fixed evaluation')
                    log_path = OUT / f'eval_{j["job_id"]}_{ck[:-3]}.log'
                    with log_path.open('x') as log:
                        start = time.monotonic()
                        worker = subprocess.Popen(
                            [sys.executable, 'scripts/rffi_post_a/run_variance.py', '--eval-job', j['job_id'],
                             '--checkpoint', ck], cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(a.gpus[0])),
                            stdout=log, stderr=subprocess.STDOUT)
                        timeout = cap - spent - 10
                        while worker.poll() is None:
                            if stopped or time.monotonic() - start >= timeout:
                                worker.terminate()
                                worker.wait(timeout=10)
                                break
                            time.sleep(.2)
                        sec = time.monotonic() - start
                        spent += sec
                        rc = worker.returncode
                        worker = None
                    evals.append(dict(job_id=j['job_id'], checkpoint=ck, seconds=sec, exit_code=rc))
                    print(json.dumps(evals[-1]), flush=True)
                    if rc != 0:
                        raise RuntimeError(f'fixed evaluation failed; retained log {log_path}')
    except BaseException as exc:
        stopped, error = True, repr(exc)
        print('RV1 stopped: ' + error, flush=True)
    finally:
        if worker is not None:
            worker.terminate()
            worker.wait()
        for r in running.values():
            if r['proc'].poll() is None:
                r['proc'].terminate()
            r['proc'].wait()
            sec = time.monotonic() - r['start']
            spent += sec
            r['log'].close()
            queue.cancel(r['job']['job_id'], reason='RV1 budget/interruption', gpu_reserved_seconds=sec,
                         exit_code=r['proc'].returncode)
            tasks.append(dict(job_id=r['job']['job_id'], state='cancelled', seconds=sec))
        if queue is not None:
            queue.export_jsonl(OUT / 'jobs.jsonl')
        reg_json = json.loads((CONFIG / 'registration.json').read_text())
        reg_json['reservation'].update(active=False, closed_after='RV1')
        write_json(CONFIG / 'registration.json', reg_json)
        write_json(OUT / 'execution_ledger.json', dict(
            fits=tasks, evaluations=evals, cross_pair=checks, reserved_gpu_hours=spent / 3600.,
            cap_gpu_hours=cap / 3600., stopped=stopped, error=error, reservation_closed=True,
            commit=commit, gpu_info=gpu_info, wall_seconds=time.monotonic() - process_start))
    if stopped:
        raise SystemExit(1)
    print(f'RV1 completed; reservation closed; {spent / 3600.:.6f} A100 GPUh', flush=True)


if __name__ == '__main__':
    main()
