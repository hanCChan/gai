#!/usr/bin/env python3
"""Capped32 fitting jobs, shared schedules, inner-only frozen LR selection."""
import concurrent.futures,datetime,json,os,signal,subprocess,sys,time
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def main():
    reg=json.loads((ROOT/'configs/b1_baselines/registration.json').read_text());out=ROOT/'outputs/b1_baselines'
    assert (out/'preflight.json').exists() and reg['user_approved'].startswith('B1:32jobs')
    for stage in ('search','formal'):(out/stage/'logs').mkdir(parents=True,exist_ok=False);(out/stage/'runs').mkdir()
    records=[]
    def run(stage,index,gpu):
        jobs=reg['matrix']['search_jobs' if stage=='search' else 'formal_jobs'];job=jobs[index]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');log=out/stage/'logs'/(job['run_id']+'.log');timeout=reg['per_job_timeout_seconds'][stage];start=time.perf_counter();timed_out=False
        with log.open('x') as f:
            process=subprocess.Popen([sys.executable,'scripts/train_b1_baseline.py','--stage',stage,'--index',str(index)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(process.pid,signal.SIGTERM)
                try:code=process.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
        seconds=time.perf_counter()-start
        if timed_out:
            dst=out/stage/'runs'/job['run_id'];dst.mkdir(exist_ok=True);dump(dst/'status.json',dict(status='failed',error='reserved-GPU timeout; no retry',seconds=seconds))
        result=dict(stage=stage,index=index,run_id=job['run_id'],gpu=gpu,exit_code=code,timed_out=timed_out,reserved_seconds=seconds);print(json.dumps(result),flush=True);return result
    def save(state,**extra):dump(out/'execution.json',dict(state=state,jobs=records,fitting_jobs_launched=len(records),reserved_gpu_hours=sum(r['reserved_seconds'] for r in records)/3600,retries=0,budget_cap_gpu_hours=2,**extra))
    records.append(run('search',0,0));first=records[0]
    projection=first['reserved_seconds']*(1+1.8*(15+16*2*1.25))
    save('calibrated',projected_gpu_hours=projection/3600)
    if first['exit_code']!=0 or projection>7200:
        save('STOP_FIRST_JOB_FAILED_OR_PROJECTED_BUDGET_EXCEEDED',projected_gpu_hours=projection/3600);return
    def remaining_search(gpu):return [run('search',i,gpu) for i in range(1,16) if i%4==gpu]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for completed in pool.map(remaining_search,range(4)):records.extend(completed)
    save('source_search_completed',projected_gpu_hours=projection/3600)
    if any(r['exit_code']!=0 for r in records):save('STOP_SEARCH_FAILURE_NO_RETRY');return
    choices={}
    for fold in range(2):
        for method in reg['methods']:
            candidates=[]
            for job in reg['matrix']['search_jobs']:
                if job['fold']!=fold or job['method']!=method:continue
                path=out/'search/runs'/job['run_id'];metrics=json.loads((path/'metrics.json').read_text());part=metrics['macro_f1_fraction'];exact=Fraction(part['numerator'],part['denominator'])
                candidates.append((exact,-job['lr'],job,metrics))
            assert len(candidates)==2;winner=max(candidates,key=lambda x:(x[0],x[1]))
            choices[f'CV{fold}_{method}']=dict(lr=winner[2]['lr'],winner=winner[2]['run_id'],macro_f1=float(winner[0]),macro_f1_fraction=dict(numerator=winner[0].numerator,denominator=winner[0].denominator),candidates=[dict(run=x[2]['run_id'],lr=x[2]['lr'],macro_f1=float(x[0]),macro_f1_fraction=dict(numerator=x[0].numerator,denominator=x[0].denominator)) for x in candidates])
    dump(out/'selection.frozen.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),choices=choices,outer_evaluation_used=False,metric='fixed10 source-internal macro-F1; exact tie->lower LR'))
    def formal_queue(gpu):return [run('formal',i,gpu) for i in range(16) if i%4==gpu]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for completed in pool.map(formal_queue,range(4)):records.extend(completed)
    assert len(records)==32 and sum(r['reserved_seconds'] for r in records)<7200
    save('completed' if all(r['exit_code']==0 for r in records) else 'completed_with_retained_failures',selection_frozen_before_formal=True)
    print('B1 stopped after approved32jobs',flush=True)
if __name__=='__main__':main()
