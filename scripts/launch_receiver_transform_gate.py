#!/usr/bin/env python3
"""Finite queue, 20 encoders, four GPUs, no fitting retries or replacement."""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from receiver_transform_common import dump
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'outputs/receiver_transform';logs=out/'logs';logs.mkdir(exist_ok=False)
    (out/'runs').mkdir(exist_ok=False)
    results=[];lock=threading.Lock();stop=threading.Event()
    def queue(gpu):
        for index in range(gpu,20,4):
            if stop.is_set():break
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            begin=time.perf_counter()
            with (logs/f'job{index:02d}.log').open('x') as stream:
                process=subprocess.run([sys.executable,'scripts/train_receiver_transform_gate.py','--index',str(index)],
                    cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            record=dict(index=index,gpu=gpu,exit_code=process.returncode,reserved_seconds=time.perf_counter()-begin)
            if process.returncode:stop.set()
            with lock:
                results.append(record)
                dump(out/'execution.json',dict(state='STOP_FAILURE_NO_RETRY' if stop.is_set() else 'running',
                    jobs=sorted(results,key=lambda x:x['index']),completed_jobs=len(results),planned_jobs=20,
                    reserved_gpu_hours=sum(x['reserved_seconds'] for x in results)/3600,
                    retries=0,optimizer_jobs_are_sharedness_gate_only=True,formal_poc_jobs=0))
            print(json.dumps(record),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(queue,range(4)))
    dump(out/'execution.json',dict(state='completed' if len(results)==20 and not stop.is_set() else 'STOP_FAILURE_NO_RETRY',
        jobs=sorted(results,key=lambda x:x['index']),completed_jobs=len(results),planned_jobs=20,
        reserved_gpu_hours=sum(x['reserved_seconds'] for x in results)/3600,retries=0,formal_poc_jobs=0))

if __name__=='__main__':main()
