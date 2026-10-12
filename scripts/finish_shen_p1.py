#!/usr/bin/env python3
"""Finish authorized P1: fair two-LR screening, frozen recipes, three seeds.

Never runs P2/P3 or final tests. Uses existing queues, then sequentially owns
all four GPUs for confirmation. Reports and failures are retained.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n'); tmp.replace(path)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    status_path=a.output/'pipeline_status.json'
    first=ROOT/'outputs/shen_evidence/p1_seed20261012'; search=ROOT/'outputs/shen_evidence/p1_lr0003'
    def status(stage,**extra): save(status_path,{'stage':stage,'timestamp_utc':datetime.now(timezone.utc).isoformat(),**extra})
    try:
        status('waiting_for_screening')
        for folder in (first,search):
            deadline=time.monotonic()+10800
            while not (folder/'launcher_results.json').exists():
                if time.monotonic()>deadline: raise TimeoutError(f'Training queue timeout: {folder}')
                time.sleep(5)
            results=json.loads((folder/'launcher_results.json').read_text())
            if any(r['returncode']!=0 for group in results for r in group): raise RuntimeError(f'Failed screening queue: {folder}')
        choices=[]
        for kind in ('full_iq','strict_ib','oob_only','cis'):
            for receiver in ('n210_1','rtl_6'):
                candidates=[]
                for folder in (first,search):
                    for run in folder.iterdir():
                        if not (run/'metrics.json').exists(): continue
                        m=json.loads((run/'metrics.json').read_text())
                        if m['kind']==kind and m['receiver']==receiver:
                            candidates.append((m,json.loads((run/'config.json').read_text()),run))
                if len(candidates)!=2 or {m['lr'] for m,_,_ in candidates}!={.001,.0003}: raise AssertionError('Equal two-LR budgets required')
                m,cfg,run=max(candidates,key=lambda c:c[0]['macro_f1'])
                choices.append({'kind':kind,'receiver':receiver,'lr':m['lr'],'config':cfg,'screening_run':str(run),
                                'screening_macro_f1':m['macro_f1'],'screening_accuracy':m['packet_accuracy'],
                                'expected_seconds':m.get('total_seconds',m.get('training_seconds',1.0)),
                                'candidate_metrics':[{'run':str(r),**c} for c,_,r in candidates]})
        selection=a.output/'selection.json'
        save(selection,{'selection_seed':20261012,'criterion':'whole Day2 macro_f1; equal two-LR budgets; tie prefers original .001',
                        'frozen_utc':datetime.now(timezone.utc).isoformat(),'choices':choices})
        matrix=[]
        for seed in (20261013,20261014):
            for c in choices:
                cfg=dict(c['config']); cfg['seed']=seed
                # Runtime provenance must be recorded afresh by the runner.
                for key in ('source_commit','protocol_sha256','cuda_visible_devices','torch_version','cuda_runtime'):
                    cfg.pop(key,None)
                run_name=f'{cfg["experiment_id"]}_{c["receiver"]}_seed{seed}'
                config_path=a.output/f'{run_name}.json'; save(config_path,cfg)
                gpu=('full_iq','strict_ib','oob_only','cis').index(c['kind'])
                matrix.append({'run_id':run_name,'gpu':gpu,'config':str(config_path.resolve()),'receiver':c['receiver'],
                               'kind':c['kind'],'stage':'P1','automatic_start':True})
        # Identical V100 cards: balance repeated runs by measured screening cost.
        # Input/recipe/seed stay frozen; only physical GPU placement changes.
        costs={(c['receiver'],c['kind']):c['expected_seconds'] for c in choices}
        matrix.sort(key=lambda j:costs[(j['receiver'],j['kind'])],reverse=True)
        loads=[0.]*4
        for job in matrix:
            gpu=min(range(4),key=lambda i:loads[i])
            job['gpu']=gpu
            loads[gpu]+=costs[(job['receiver'],job['kind'])]
        save(a.output/'gpu_cost_plan.json',{'estimated_seconds_per_gpu':loads,'basis':'seed20261012 measured total seconds'})
        matrix_path=a.output/'confirmation_matrix.json'; save(matrix_path,matrix)
        oob=[c for c in choices if c['kind']=='oob_only']
        go=all(c['screening_accuracy']>.1 for c in oob)
        save(a.output/'decision.json',{'confirmation_go':go,'reason':'selected OOB-only exceeds 10% in both Day2 receivers',
                                     'reliability_fusion_supported':False,'P2_P3_started':False})
        runs=[first,search]
        if go:
            status('three_seed_confirmation_running')
            confirmed=a.output/'confirmation_runs'
            subprocess.run([sys.executable,str(ROOT/'scripts/launch_shen_evidence.py'),'--stage','p1','--matrix',str(matrix_path.resolve()),
                            '--smoke-dir',str(ROOT/'outputs/shen_evidence/p0_20261012/four_gpu_smoke_v2'),
                            '--output',str(confirmed.resolve())],cwd=ROOT,check=True)
            runs.append(confirmed)
        status('generating_report')
        subprocess.run([sys.executable,str(ROOT/'scripts/report_shen_evidence.py'),'--runs',*[str(r) for r in runs],
                        '--selection',str(selection.resolve()),'--output',str((a.output/'report').resolve())],cwd=ROOT,check=True)
        status('completed',report=str((a.output/'report/REPORT.md').resolve()),three_seed_confirmation=go)
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise

if __name__=='__main__': main()
