#!/usr/bin/env python3
"""Queue P2 jobs (search or formal) on 4 GPUs. Same convention as P1 queue_p1.py.
Skips jobs whose output dir already has status.json.

    python3 queue_p2.py --jobs jobs.txt --out OUT --sets sets_p1.json [--workers 8]
"""
import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sets', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    lines = [l.split() for l in Path(a.jobs).read_text().splitlines() if l.strip() and not l.startswith('#')]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failed = []

    def worker(k):
        while True:
            with lock:
                if not lines:
                    return
                args = lines.pop(0)
            run_id = f"shen_rank16_CV{args[3]}_{args[5]}{args[7]}_Rq_{args[1]}_seed{args[9]}"
            if (Path(a.out) / run_id / 'status.json').exists():
                with lock:
                    print('SKIP ' + ' '.join(args), flush=True)
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % 4), CUBLAS_WORKSPACE_CONFIG=':4096:8')
            cmd = [sys.executable, str(HERE / 'train_p2.py'), *args, '--sets', a.sets, '--out', a.out]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            msg = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:]
            with lock:
                print(('OK   ' if r.returncode == 0 else 'FAIL ') + ' '.join(args) + ' :: ' + msg, flush=True)
                if r.returncode:
                    failed.append(args)

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(a.workers)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    print(f'done, failed={len(failed)}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
