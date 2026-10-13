#!/usr/bin/env python3
"""Fixed twelve-job dispatcher; measured first jobs are part of the matrix."""
import json,os,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/keyan/venvs/shen-stage1/bin/python'


def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    reg=json.loads((ROOT/'configs/receiver_r2/registration.json').read_text());jobs=reg['jobs'];cap=reg['approved_new_cap_reserved_V100_GPUh']*3600
    assert cap==10800 and len(jobs)==12 and not reg['R2_C_authorized']
    out=ROOT/'outputs/receiver_r2';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'execution.json').exists(),'No dispatcher or fit restart'
    ledger=[];running=[];measured={};deadline={};reserve=360.
    def key(j):return (j['stage'],j['dataset'],j['method'])
    def historical_estimate(j):
        h=reg['reservation_estimate']['historical_method_timings'][j['dataset']+'_'+j['method']]
        return (j['epochs']*h['epoch_max_seconds']+25)*(4 if j['stage']=='diagnostic' else 1.5)
    def estimate(j):
        k=key(j)
        return measured[k]*1.5 if k in measured else historical_estimate(j)
    first=[];seen=set()
    for i,j in enumerate(jobs):
        if key(j) not in seen:first.append(i);seen.add(key(j))
    def receipt(state,**extra):
        dump(out/'execution.json',dict(state=state,jobs=ledger,reserved_gpu_hours=sum(r['reserved_seconds'] for r in ledger)/3600,
            fitting_jobs=len(ledger),approved_cap_GPUh=cap/3600,failures=sum(r['exit_code']!=0 for r in ledger),retries=0,seed_replacements=0,**extra))
    try:
        for phase,queue in [('first_registered_jobs',first),('remaining_fixed_matrix',[i for i in range(len(jobs)) if i not in first])]:
            queue=list(queue)
            if phase=='remaining_fixed_matrix':
                used=sum(r['reserved_seconds'] for r in ledger);forecast=sum(estimate(jobs[i]) for i in queue)+reserve
                assert used+forecast<cap,('Complete matrix exceeds fresh cap',used,forecast,cap)
                for k,secs in measured.items():deadline[k]=max(120,secs*2+60)
                dump(out/'measured_budget.json',dict(used_GPUh=used/3600,remaining_forecast_with_margin_GPUh=forecast/3600,
                    cap_GPUh=cap/3600,types={str(k):dict(measured_reserved_seconds=s,timeout_seconds=deadline[k],peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in ledger if key(jobs[r['index']])==k)) for k,s in measured.items()},
                    first_jobs_included_in_matrix=True,remaining_jobs=len(queue)))
            while queue or running:
                busy={a['gpu'] for a in running}
                for gpu in range(4):
                    if queue and gpu not in busy:
                        i=queue.pop(0);j=jobs[i];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                        log=(out/(j['run_id']+'.log')).open('w');tick=time.perf_counter()
                        p=subprocess.Popen([PYTHON,str(ROOT/'scripts/train_receiver_r2.py'),'--index',str(i)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                        running.append(dict(proc=p,file=log,tick=tick,index=i,gpu=gpu));print('LAUNCH',i,j['run_id'],'gpu',gpu,flush=True)
                time.sleep(.25)
                for a in running[:]:
                    p=a['proc'];j=jobs[a['index']];elapsed=time.perf_counter()-a['tick']
                    if p.poll() is not None:
                        a['file'].close();s=json.loads((out/j['run_id']/'status.json').read_text())
                        row=dict(index=a['index'],run_id=j['run_id'],gpu=a['gpu'],exit_code=p.returncode,reserved_seconds=elapsed,peak_allocated_bytes=s.get('peak_allocated_bytes',0))
                        ledger.append(row);running.remove(a);receipt('running',phase=phase)
                        assert p.returncode==0 and s['state']=='completed',('Registered fit failed; no retry',j['run_id'],s)
                        measured[key(j)]=max(measured.get(key(j),0),elapsed);print('DONE',j['run_id'],elapsed,flush=True)
                    elif elapsed>deadline.get(key(j),max(300,historical_estimate(j)*3)):
                        raise RuntimeError('Job reservation timeout: '+j['run_id'])
                accrued=sum(r['reserved_seconds'] for r in ledger)+sum(time.perf_counter()-a['tick'] for a in running)
                if accrued+reserve>=cap:raise RuntimeError('New R2-A reservation cap reached')
        assert len(ledger)==12 and len({a['run_id'] for a in ledger})==12
        receipt('completed')
    except BaseException as error:
        for a in running:
            p=a['proc']
            if p.poll() is None:p.terminate();p.wait()
            a['file'].close();ledger.append(dict(index=a['index'],run_id=jobs[a['index']]['run_id'],gpu=a['gpu'],exit_code=p.returncode,reserved_seconds=time.perf_counter()-a['tick'],aborted=True))
        receipt('halted',error=repr(error));raise

if __name__=='__main__':main()
