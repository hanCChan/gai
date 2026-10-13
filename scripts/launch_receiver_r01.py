#!/usr/bin/env python3
"""Budget-aware finite matrix. Calibrations are already registered search jobs."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from wisig_w01_common import sha,dump


def launch(stage,index,gpu,reg,ledger):
    job=reg[stage+'_jobs'][index];folder=ROOT/'outputs/receiver_r01/logs';folder.mkdir(exist_ok=True)
    path=folder/(job['run_id']+'.log');assert not path.exists()
    f=path.open('w');env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    t=time.perf_counter();proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/train_receiver_r01.py'),'--stage',stage,'--index',str(index)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    return dict(proc=proc,file=f,tick=t,index=index,gpu=gpu,job=job)


def close(active,ledger):
    code=active['proc'].poll();assert code is not None;active['file'].close()
    row=dict(stage=active['job']['stage'],index=active['index'],run_id=active['job']['run_id'],gpu=active['gpu'],
             reserved_seconds=time.perf_counter()-active['tick'],exit_code=code)
    ledger.append(row);dump(ROOT/'outputs/receiver_r01/execution.json',dict(jobs=ledger,reserved_gpu_hours=sum(r['reserved_seconds'] for r in ledger)/3600))
    print(json.dumps(row),flush=True)
    if code:raise RuntimeError('Failed fit; no retries: '+row['run_id'])


def select(reg):
    choices={};all_results=[]
    for job in reg['search_jobs']:
        run=ROOT/'outputs/receiver_r01/search'/job['run_id'];status=json.loads((run/'status.json').read_text());assert status['state']=='completed'
        sel=json.loads((run/'selection.json').read_text());key=f"{job['dataset']}_{job['backbone']}_CV{job['fold']}_{job['method']}"
        row=dict(key=key,lr=job['lr'],run_id=job['run_id'],validation_CE=sel['source_validation']['CE'],validation_macro_f1=sel['source_validation']['macro_f1'])
        all_results.append(row)
        if key not in choices or (row['validation_CE'],row['lr'])<(choices[key]['validation_CE'],choices[key]['lr']):choices[key]=row
    dest=ROOT/'outputs/receiver_r01/selection.frozen.json';assert not dest.exists()
    result=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),choices=choices,all_candidates=all_results,
                source_only=True,internal_RX_used=False,outer_RX_used=False,formal_fits_started=False)
    dump(dest,result);dump(ROOT/'reports/receiver_r01/2026-10-13/selection.frozen.json',result)


def main():
    reg=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text());dest=ROOT/'outputs/receiver_r01/execution.json';assert not dest.exists()
    ledger=[];cap=reg['reserved_gpu_hours_cap']*3600
    # First each dataset/backbone/method: sequential actual calibration before scheduling the rest.
    calibrations=[];seen=set()
    for i,j in enumerate(reg['search_jobs']):
        key=(j['dataset'],j['backbone'],j['method'])
        if key not in seen:seen.add(key);calibrations.append(i)
    timings={};estimates=[]
    try:
        for i in calibrations:
            active=launch('search',i,0,reg,ledger);active['proc'].wait(timeout=1800);close(active,ledger)
            job=reg['search_jobs'][i];timings[(job['dataset'],job['backbone'],job['method'])]=ledger[-1]['reserved_seconds']/job['epochs']
            total=sum(row['reserved_seconds'] for row in ledger)
            todo=[j for k,j in enumerate(reg['search_jobs']) if k not in [r['index'] for r in ledger]]+reg['formal_jobs']
            pred=sum(timings.get((j['dataset'],j['backbone'],j['method']),max(timings.values()))*j['epochs']+5 for j in todo)*1.5
            estimates.append(dict(calibration=job['run_id'],used_GPUh=total/3600,remaining_estimate_with_50pct_margin_GPUh=pred/3600))
            dump(ROOT/'outputs/receiver_r01/budget_calibration.json',dict(estimates=estimates,cap_GPUh=6.))
            if total+pred>cap:raise RuntimeError('Full matrix estimate exceeds cap; request budget without truncation')
        # Independent model processes; no multi-agent work and no shared weights.
        for stage in ('search','formal'):
            if stage=='formal':select(reg)
            queue=[i for i in range(len(reg[stage+'_jobs'])) if stage!='search' or i not in calibrations];running=[]
            while queue or running:
                busy={a['gpu'] for a in running}
                for gpu in range(4):
                    if queue and gpu not in busy:
                        i=queue.pop(0);running.append(launch(stage,i,gpu,reg,ledger))
                time.sleep(.25)
                for a in running[:]:
                    if a['proc'].poll() is not None:close(a,ledger);running.remove(a)
                    elif time.perf_counter()-a['tick']>1800:raise RuntimeError('Registered job timeout, no retry')
                accrued=sum(r['reserved_seconds'] for r in ledger)+sum(time.perf_counter()-a['tick'] for a in running)
                if accrued>cap:raise RuntimeError('Reserved GPU cap reached')
        dump(dest,dict(state='completed',jobs=ledger,reserved_gpu_hours=sum(r['reserved_seconds'] for r in ledger)/3600,
                       fitting_jobs=len(ledger),failures=0,retries=0,seed_replacements=0))
    except BaseException as e:
        for a in locals().get('running',[]):
            if a['proc'].poll() is None:a['proc'].terminate();a['proc'].wait()
            if not a['file'].closed:close(a,ledger)
        dump(dest,dict(state='halted',error=repr(e),jobs=ledger,reserved_gpu_hours=sum(r['reserved_seconds'] for r in ledger)/3600));raise

if __name__=='__main__':main()
