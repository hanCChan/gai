#!/usr/bin/env python3
"""Finish the original matrix using method-specific measured reservation costs.

No fitting restart: existing completed jobs are skipped and all run paths are exclusive.
The initial dispatcher's surrogate estimate is preserved, not overwritten in history.
"""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from wisig_w01_common import dump
from launch_receiver_r01 import launch,close,select


def main():
    out=ROOT/'outputs/receiver_r01';reg=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text())
    old=json.loads((out/'execution.json').read_text());extra=json.loads((out/'shen_calibration_execution.json').read_text())
    assert old['state']=='halted' and 'Full matrix estimate exceeds cap' in old['error'] and extra['state']=='completed'
    dump(out/'initial_dispatcher_estimate_halt.json',old)
    ledger=old['jobs']+extra['jobs'];assert len({r['run_id'] for r in ledger})==len(ledger) and all(r['exit_code']==0 for r in ledger)
    amendment=json.loads((ROOT/'configs/receiver_r01/budget_amendment.json').read_text())
    cap=amendment['new_cap_reserved_V100_GPUh']*3600;assert cap==25200
    done={r['run_id'] for r in ledger};timings={}
    for row in ledger:
        job=reg[row['stage']+'_jobs'][row['index']];key=(job['dataset'],job['backbone'],job['method'])
        status=json.loads((out/row['stage']/row['run_id']/'status.json').read_text())
        timings[key]=(status['fitting_seconds']/job['epochs'],max(10,row['reserved_seconds']-status['fitting_seconds']))
    remaining=[j for j in reg['search_jobs']+reg['formal_jobs'] if j['run_id'] not in done]
    # Formal WiSig has 14 fitting RX versus search10; account for that before the margin.
    def estimate(job):
        epoch_seconds,setup_seconds=timings[(job['dataset'],job['backbone'],job['method'])]
        return epoch_seconds*job['epochs']*(1.4 if job['dataset']=='wisig' and job['stage']=='formal' else 1.25 if job['dataset']=='shen' and job['stage']=='formal' else 1)+setup_seconds
    predicted=sum(estimate(j) for j in remaining)*1.5
    used=sum(r['reserved_seconds'] for r in ledger);assert used+predicted+300<cap
    dump(out/'method_specific_budget.json',dict(existing_jobs=len(ledger),remaining_jobs=len(remaining),used_GPUh=used/3600,
        remaining_estimate_with_RX_scaling_and_50pct_margin_GPUh=predicted/3600,analysis_GPUh_allowance=300/3600,cap_GPUh=cap/3600,
        no_fitting_definition_change=True,initial_surrogate_estimate_preserved=True))
    running=[]
    try:
        for stage in ['search','formal']:
            if stage=='formal':select(reg)
            queue=[i for i,j in enumerate(reg[stage+'_jobs']) if j['run_id'] not in done]
            queue.sort(key=lambda i:estimate(reg[stage+'_jobs'][i]),reverse=True)
            while queue or running:
                busy={a['gpu'] for a in running}
                for gpu in range(4):
                    if queue and gpu not in busy:running.append(launch(stage,queue.pop(0),gpu,reg,ledger))
                time.sleep(.25)
                for active in running[:]:
                    if active['proc'].poll() is not None:
                        close(active,ledger);done.add(active['job']['run_id']);running.remove(active)
                    elif time.perf_counter()-active['tick']>max(1800,estimate(active['job'])*2+60):
                        raise RuntimeError('Job reservation timeout; no fitting retry')
                accrued=sum(r['reserved_seconds'] for r in ledger)+sum(time.perf_counter()-a['tick'] for a in running)
                if accrued+300>cap:raise RuntimeError('Reserved GPU cap with analysis allowance reached')
        assert len(ledger)==85 and len({r['run_id'] for r in ledger})==85
        dump(out/'execution.json',dict(state='completed',jobs=ledger,reserved_gpu_hours=sum(r['reserved_seconds'] for r in ledger)/3600,
            fitting_jobs=85,failures=0,retries=0,seed_replacements=0,dispatcher_estimate_correction=True))
    except BaseException as error:
        for a in running:
            if a['proc'].poll() is None:a['proc'].terminate();a['proc'].wait()
            a['file'].close()
        dump(out/'execution.json',dict(state='halted',error=repr(error),jobs=ledger));raise

if __name__=='__main__':main()
