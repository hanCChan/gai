#!/usr/bin/env python3
"""Measure the five registered Shen first jobs while WiSig capacity timing completes."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from wisig_w01_common import dump


def main():
    reg=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text());ledger=[]
    dest=ROOT/'outputs/receiver_r01/shen_calibration_execution.json';assert not dest.exists()
    seen=set()
    for index,job in enumerate(reg['search_jobs']):
        if job['dataset']!='shen' or job['method'] in seen:continue
        seen.add(job['method']);path=ROOT/'outputs/receiver_r01/logs'/(job['run_id']+'.log');assert not path.exists()
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with path.open('w') as f:
            tick=time.perf_counter();p=subprocess.Popen([sys.executable,str(ROOT/'scripts/train_receiver_r01.py'),'--stage','search','--index',str(index)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            code=p.wait(timeout=1800);seconds=time.perf_counter()-tick
        row=dict(stage='search',index=index,run_id=job['run_id'],gpu=1,reserved_seconds=seconds,exit_code=code)
        ledger.append(row);dump(dest,dict(jobs=ledger,reserved_gpu_hours=sum(a['reserved_seconds'] for a in ledger)/3600));print(json.dumps(row),flush=True)
        if code:raise RuntimeError('Failed registered calibration, no fitting retry')
    dump(dest,dict(state='completed',jobs=ledger,reserved_gpu_hours=sum(a['reserved_seconds'] for a in ledger)/3600))

if __name__=='__main__':main()
