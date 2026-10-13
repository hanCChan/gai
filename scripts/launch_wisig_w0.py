#!/usr/bin/env python3
"""Sequential two-pilot window; timing after first; no formal expansion."""
import os,json,subprocess,sys,time,signal
from pathlib import Path
from wisig_w01_common import dump
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'outputs/wisig_w01';(out/'runs').mkdir();(out/'logs').mkdir();jobs=[]
 for i in [0,1]:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
  started=time.perf_counter()
  with (out/'logs'/f'pilot{i}.log').open('x') as f:
   p=subprocess.Popen([sys.executable,'scripts/train_wisig_w0.py','--index',str(i)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=1800)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);code=p.wait()
  jobs.append(dict(index=i,exit_code=code,reserved_seconds=time.perf_counter()-started))
  dump(out/'execution.json',dict(state='STOP_FAILURE_NO_RETRY' if code else 'first_pilot_timed' if i==0 else 'completed',jobs=jobs,reserved_gpu_hours=sum(x['reserved_seconds'] for x in jobs)/3600,retries=0,formal_models=0,
   estimated_12_model_gpu_hours=jobs[0]['reserved_seconds']*12*1.3/3600))
  print(json.dumps(jobs[-1]),flush=True)
  if code:break
if __name__=='__main__':main()
