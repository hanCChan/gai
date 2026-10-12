#!/usr/bin/env python3
"""Complete all preregistered P2-A screening and confirmation; never launch P2-B/P3."""
import argparse,datetime,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GROUPS={'A1_matched_representation','A2_capacity_raw_cis','A4_frequency_rings','A5_information_complementarity','A6_stability'}
def run(args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    status=out/'pipeline_status.json'
    def state(s,**kw):status.write_text(json.dumps({'status':s,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw},indent=2)+'\n')
    try:
        state('screening_running');run(['scripts/launch_shen_p2a.py','--stage','train','--matrix','configs/shen_p2a/screening_matrix.json','--output',out/'screening','--smoke-dir',ROOT/'outputs/shen_p2a/four_gpu_smoke_v2'])
        jobs=json.loads((ROOT/'configs/shen_p2a/screening_matrix.json').read_text());groups={}
        for job in jobs:
            cfg=json.loads((ROOT/job['config']).read_text());m=json.loads((out/'screening'/job['run_id']/'metrics.json').read_text())
            if cfg['comparison_group'] in GROUPS:groups.setdefault((cfg['receiver'],cfg['comparison_group'],cfg['method']),[]).append((m,cfg,job))
        choices=[];confirm=[];(out/'configs').mkdir()
        for key,candidates in sorted(groups.items()):
            m,cfg,job=max(candidates,key=lambda r:(r[0]['macro_f1'],-r[1]['weight_decay'],r[1]['lr']==.001))
            choices.append({'receiver':key[0],'comparison_group':key[1],'method':key[2],'screening_run':str(out/'screening'/job['run_id']),'config':cfg,'candidate_metrics':[r[0] for r in candidates]})
            for seed in (20261013,20261014):
                c=dict(cfg,seed=seed);name=f'{job["run_id"]}_seed{seed}';path=out/'configs'/f'{name}.json';path.write_text(json.dumps(c,indent=2)+'\n')
                confirm.append(dict(job,run_id=name,config=str(path)))
        assert len(choices)==28 and len(confirm)==56
        (out/'selection.frozen.json').write_text(json.dumps({'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'criterion':'Day2 fixed10-class Macro-F1; ties lower WD then original LR .001','choices':choices,'P2_B_or_P3_automatic_start':False},indent=2)+'\n')
        (out/'confirmation_matrix.json').write_text(json.dumps(confirm,indent=2)+'\n');state('confirmation_running',groups=28,runs=56)
        run(['scripts/launch_shen_p2a.py','--stage','train','--matrix',out/'confirmation_matrix.json','--output',out/'confirmation','--smoke-dir',ROOT/'outputs/shen_p2a/four_gpu_smoke_v2'])
        state('report_running');run(['scripts/report_shen_p2a.py','--pipeline',out,'--output',out/'report'])
        state('completed',screening_runs=52,confirmation_runs=56,report=str(out/'report'),P2_B_or_P3_started=False)
    except BaseException as exc:state('failed',error=repr(exc));raise
