#!/usr/bin/env python3
"""Eight approved fitting jobs, no fitting retry; pre-fit failures retained."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    r=json.loads((ROOT/'configs/d05_b0/registration.json').read_text())
    out=ROOT/'outputs/d05_b0/b0';out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();(out/'runs').mkdir()
    assert len(r['jobs'])==r['b0']['max_new_training_jobs']==8
    startup_seconds=r['b0'].get('startup_failure_reserved_seconds',0)
    timeout=r['b0'].get('per_job_timeout_seconds',450)
    assert startup_seconds+8*timeout<=3600
    def queue(gpu):
        records=[]
        for index,job in enumerate(r['jobs']):
            if index%4!=gpu:continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            log=out/'logs'/(job['run_id']+'.log');start=time.perf_counter();timed_out=False
            with log.open('x') as f:
                try:
                    result=subprocess.run([sys.executable,'scripts/train_b0_replication.py','--job-index',str(index)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,check=False);code=result.returncode
                except subprocess.TimeoutExpired:timed_out=True;code=-9
            seconds=time.perf_counter()-start
            run=out/'runs'/job['run_id']
            if timed_out:
                run.mkdir(exist_ok=True);(run/'status.json').write_text(json.dumps(dict(status='failed',error=f'{timeout}-second per-job reserved-GPU budget timeout; no retry',seconds=seconds),indent=2)+'\n')
            records.append(dict(run_id=job['run_id'],gpu=gpu,exit_code=code,timed_out=timed_out,seconds=seconds,log_sha256=__import__('hashlib').sha256(log.read_bytes()).hexdigest()))
            print(json.dumps(records[-1]),flush=True)
        return records
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for values in pool.map(queue,range(4)):records.extend(values)
    (out/'execution.json').write_text(json.dumps(dict(jobs=sorted(records,key=lambda x:x['run_id']),fitting_jobs_launched=len(records),startup_pre_fit_attempts_retained=8,reserved_gpu_hours=(sum(x['seconds'] for x in records)+startup_seconds)/3600,fitting_retries=0,seed_replacements=0,b1_started=False),indent=2)+'\n')
    print('B0 completed without retries',flush=True)
if __name__=='__main__':main()
