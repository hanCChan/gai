#!/usr/bin/env python3
"""Run a sharded inference script on the 4 GPUs in parallel and wait (P2E helper).

    python3 run_sharded.py p2e_audit.py --out reports/receiver_p2e/audit
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    script, args = sys.argv[1], sys.argv[2:]
    procs = []
    for i in range(4):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), CUBLAS_WORKSPACE_CONFIG=':4096:8')
        log = open(f'/tmp/{Path(script).stem}_shard{i}.log', 'w')
        procs.append(subprocess.Popen([sys.executable, str(HERE / script), *args, '--shard', f'{i}/4'],
                                      env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    print('exit codes', codes)
    sys.exit(max(codes))


if __name__ == '__main__':
    main()
