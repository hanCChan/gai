#!/usr/bin/env python3
"""Bounded A0/A3 evaluation; verify every score-derived metric without reading signals again."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from rffi_calibration.registry import Registry, file_sha


def independent_metrics(scores, labels):
    p = softmax(scores, axis=1)
    pred = p.argmax(1)
    t = p.shape[1]
    confusion = np.zeros((t, t), dtype=np.int64)
    np.add.at(confusion, (labels, pred), 1)
    tp = confusion.diagonal()
    true_n, pred_n = confusion.sum(1), confusion.sum(0)
    denom = true_n + pred_n
    f1 = np.divide(2 * tp, denom, out=np.zeros(t, float), where=denom > 0)
    recall = np.divide(tp, true_n, out=np.zeros(t, float), where=true_n > 0)
    brier = ((p - np.eye(t)[labels])**2).sum(1).mean()
    conf = p.max(1)
    correct = pred == labels
    ece = 0.
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (conf > low) & (conf <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return dict(acc=correct.mean(), macro_f1=f1.mean(), brier=brier, ece=ece,
                nll=-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-300, None)).mean(),
                **{f'recall_{i}': v for i, v in enumerate(recall)}), confusion


def verify_scores(out):
    rows = pd.read_csv(out / 'rows.csv')
    checked, confusions = 0, []
    for path in sorted(out.glob('predictions_rx*.npz')):
        rx = path.stem.removeprefix('predictions_rx')
        with np.load(path, allow_pickle=False) as a:
            assert len(set(a['query_ids'])) == len(a['query_ids'])
            for head, m, draw, scores in zip(a['heads'], a['budgets'], a['draws'], a['scores']):
                matching = rows[(rows.rx.astype(str) == rx) & (rows['head'] == head)
                                & (rows.m == m) & (rows.draw == draw)]
                assert len(matching) == 1
                assert np.isfinite(scores).all()
                metrics, cm = independent_metrics(scores, a['labels'])
                for key, value in metrics.items():
                    assert abs(float(matching.iloc[0][key]) - value) < 1e-10, (rx, head, m, draw, key)
                checked += 1
                confusions.append(dict(rx=rx, head=str(head), m=int(m), draw=int(draw), confusion=cm.tolist()))
    assert checked == int((~rows['head'].str.endswith('*')).sum())
    summary = json.loads((out / 'summary.json').read_text())
    for item in summary['aggregate']:
        group = rows[(rows['head'] == item['head']) & (rows.m == item['m'])]
        rx_mean = group.groupby('rx').acc.mean()
        assert len(rx_mean) == item['n_rx']
        assert abs(rx_mean.mean() - item['mean']) < 1e-12
        assert abs(rx_mean.min() - item['worst']) < 1e-12
    result = dict(verified=True, raw_rows_checked=checked, total_rows=len(rows),
                  score_files={p.name: file_sha(p) for p in out.glob('predictions_rx*.npz')},
                  rows_sha256=file_sha(out / 'rows.csv'), summary_sha256=file_sha(out / 'summary.json'))
    (out / 'independent_verification.json').write_text(json.dumps(result, indent=2))
    (out / 'confusions.json').write_text(json.dumps(confusions))
    return result


def recover_ledger(execution_dir, attempt, max_gpu_hours):
    target = execution_dir / 'evaluation_execution.json'
    assert not target.exists(), 'do not overwrite an existing ledger'
    progress = json.loads((execution_dir / 'evaluation_progress.json').read_text())
    assert not progress['pending'] and not progress['running']
    assert len(progress['records']) == 12 and all(r['verified'] for r in progress['records'])
    ledger = dict(records=progress['records'], total_reserved_gpu_hours=progress['reserved_gpu_seconds'] / 3600.,
                  max_gpu_hours=max_gpu_hours, pending=[], stopping=False, attempt=attempt,
                  ledger_recovered=True, recovery_reason='argument-name shadowing after all task verification')
    target.write_text(json.dumps(ledger, indent=2))
    return ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', default='initial')
    ap.add_argument('--recover-ledger-only', action='store_true')
    cli_args = ap.parse_args()
    assert cli_args.attempt.replace('_', '').isalnum()
    reg = Registry()
    training_dir = ROOT / 'reports/rffi_calibration/stage_A_execution'
    execution_dir = training_dir if cli_args.attempt == 'initial' else training_dir / cli_args.attempt
    execution_dir.mkdir(parents=True, exist_ok=True)
    if cli_args.recover_ledger_only:
        print(json.dumps(recover_ledger(execution_dir, cli_args.attempt,
                                       reg.reservation()['max_reserved_gpu_hours']), indent=2))
        return
    assert reg.reservation()['active'] and reg.reservation()['allow_feature_extraction']
    assert not (execution_dir / 'evaluation_execution.json').exists(), 'existing attempt, do not overwrite'
    train = json.loads((training_dir / 'execution.json').read_text())
    assert len(train['tasks']) == 6 and all(t['state'] == 'completed' for t in train['tasks'])
    matrix = json.loads((ROOT / 'configs/rffi_calibration/formal_matrix.json').read_text())
    tasks = []
    for j in matrix['jobs']:
        out = ROOT / 'reports/rffi_calibration/E4_frozen_heads' / j['job_id']
        if cli_args.attempt != 'initial':
            out = out / cli_args.attempt
        tasks.append((j['job_id'], out, ['--ckpt', str(ROOT / 'outputs/rffi_calibration' / j['job_id'])]))
    for fold in ('w0', 'w1'):
        for lam in ('0.3', '1', '3'):
            name = f'{fold}_ccalign_lam{lam}'
            ck = Path(reg.vars['RETRAIN_ROOT']) / 'outputs/receiver_p2_wisig/search' / name
            out = ROOT / 'reports/rffi_calibration/E4_A0' / name
            if cli_args.attempt != 'initial':
                out = out / cli_args.attempt
            tasks.append((name, out, ['--legacy-wisig8', str(ck), '--fold', fold]))
    spent = train['reserved_gpu_hours'] * 3600.
    prior = list(training_dir.glob('**/evaluation_execution.json'))
    if prior:
        spent = max(json.loads(p.read_text())['total_reserved_gpu_hours'] for p in prior) * 3600.
    cap = reg.reservation()['max_reserved_gpu_hours'] * 3600.
    running, records = {}, []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while tasks or running:
            if spent + sum(time.monotonic() - v['start'] for v in running.values()) >= cap - 10:
                stopping = True
            for gpu, v in list(running.items()):
                if stopping and v['proc'].poll() is None:
                    v['proc'].terminate()
                rc = v['proc'].poll()
                if rc is None:
                    continue
                seconds = time.monotonic() - v['start']
                spent += seconds
                v['log'].close()
                verified, error = False, ''
                if rc == 0:
                    try:
                        verify_scores(v['out'])
                        verified = True
                    except Exception as e:
                        error = repr(e)
                row = dict(task=v['name'], exit_code=rc, verified=verified, error=error,
                           reserved_gpu_seconds=seconds, gpu=gpu)
                records.append(row)
                print(json.dumps(row), flush=True)
                del running[gpu]
            if not stopping:
                for gpu in range(4):
                    if gpu in running or not tasks:
                        continue
                    name, out, command_args = tasks.pop(0)
                    out.mkdir(parents=True, exist_ok=False)
                    log = (out / 'execution.log').open('x')
                    proc = subprocess.Popen([sys.executable, 'scripts/rffi_calibration/frozen_heads.py',
                                             *command_args, '--out', str(out)],
                                            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                                              OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                                              OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1'),
                                            stdout=log, stderr=subprocess.STDOUT)
                    running[gpu] = dict(proc=proc, log=log, out=out, name=name, start=time.monotonic())
                    print(f'evaluation GPU {gpu}: {name}', flush=True)
            if stopping and not running:
                break
            (execution_dir / 'evaluation_progress.json').write_text(json.dumps(dict(
                records=records, pending=[t[0] for t in tasks], running=[v['name'] for v in running.values()],
                reserved_gpu_seconds=spent), indent=2))
            time.sleep(1)
    finally:
        for v in running.values():
            if v['proc'].poll() is None:
                v['proc'].terminate()
                try:
                    v['proc'].wait(timeout=20)
                except subprocess.TimeoutExpired:
                    v['proc'].kill()
                    v['proc'].wait()
            spent += time.monotonic() - v['start']
            v['log'].close()
            records.append(dict(task=v['name'], exit_code=v['proc'].returncode, verified=False, cancelled=True))
        (execution_dir / 'evaluation_execution.json').write_text(json.dumps(dict(
            records=records, total_reserved_gpu_hours=spent / 3600., max_gpu_hours=cap / 3600.,
            pending=[t[0] for t in tasks], stopping=stopping, attempt=cli_args.attempt), indent=2))
    if len(records) != 12 or not all(r['verified'] for r in records):
        sys.exit('A0/A3 incomplete; no automatic reruns')


if __name__ == '__main__':
    main()
