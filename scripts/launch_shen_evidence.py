#!/usr/bin/env python3
"""Four independent GPU queues; P1 requires every actual-data smoke to pass."""
import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['smoke','p1'],required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--smoke-dir',type=Path)
    p.add_argument('--wait-for',type=Path,help='Wait for the original P1 queue on each GPU before reusing it')
    p.add_argument('--matrix',type=Path,default=ROOT/'configs/shen_evidence/matrix.json')
    a=p.parse_args(); matrix=json.loads(a.matrix.read_text())
    if not isinstance(matrix,list) or not all(isinstance(job,dict) for job in matrix):
        p.error('Expected a list of P1 jobs, not a P2/P3 plan')
    for job in matrix:
        if job.get('stage')!='P1' or not job.get('automatic_start'):
            p.error('This launcher only executes authorized P1 jobs')
    if a.stage=='p1':
        if a.smoke_dir is None: p.error('P1 requires --smoke-dir')
        for gpu in range(4):
            f=a.smoke_dir/f'gpu{gpu}'/'passed.json'
            if not f.exists() or json.loads(f.read_text()).get('passed') is not True:
                p.error(f'GPU {gpu} smoke has not passed')
    a.output.mkdir(parents=True,exist_ok=False)
    def queue(gpu):
        if a.wait_for is not None:
            preceding=json.loads((ROOT/'configs/shen_evidence/matrix.json').read_text())
            for previous in (r for r in preceding if r['gpu']==gpu):
                state_path=a.wait_for/previous['run_id']/'status.json'
                deadline=time.monotonic()+7200
                while True:
                    try:
                        state=json.loads(state_path.read_text())
                    except (FileNotFoundError,json.JSONDecodeError):
                        state={}
                    if state.get('status')=='completed': break
                    if state.get('status')=='failed' or time.monotonic()>deadline:
                        raise RuntimeError(f'Preceding GPU {gpu} run failed or timed out')
                    time.sleep(5)
        results=[]
        jobs=[r for r in matrix if r['gpu']==gpu]
        if a.stage=='smoke': jobs=jobs[:1]
        for r in jobs:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            run_name=f'gpu{gpu}' if a.stage=='smoke' else r['run_id']
            run_dir=a.output/run_name
            cmd=[sys.executable,str(ROOT/'scripts'/('smoke_shen_evidence.py' if a.stage=='smoke' else 'train_shen_evidence.py')),
                 '--config',str(ROOT/r['config']),'--output',str(run_dir)]
            with (a.output/f'{run_name}.log').open('w') as log:
                code=subprocess.call(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            results.append({'run_id':run_name,'gpu':gpu,'returncode':code,'command':cmd})
            # Keep failure artifacts; do not hide a failed run or retry silently.
            if code!=0: break
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        outcomes=list(ex.map(queue,range(4)))
    (a.output/'launcher_results.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    print(json.dumps(outcomes,indent=2),flush=True)
    if any(r['returncode']!=0 for group in outcomes for r in group): sys.exit(1)

if __name__=='__main__': main()
