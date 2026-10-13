#!/usr/bin/env python3
"""One evaluation pass for all six frozen 200-epoch checkpoints; no target selection."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from receiver_r01.data import wisig
from receiver_r01.core import predict,metrics
from train_receiver_r01 import build,tensors,table
from wisig_w01_common import sha,dump


def main():
    tick=time.perf_counter();out=ROOT/'outputs/receiver_r2';public=ROOT/'reports/receiver_r2/r2_a';public.mkdir(parents=True,exist_ok=True)
    r=json.loads((ROOT/'configs/receiver_r2/registration.json').read_text());old=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text())
    execution=json.loads((out/'execution.json').read_text());assert execution['state']=='completed' and execution['fitting_jobs']==12
    jobs=[j for j in r['jobs'] if j['stage']=='horizon'];frozen=[]
    for j in r['jobs']:
        p=out/j['run_id'];s=json.loads((p/'status.json').read_text());assert s['state']=='completed' and sha(p/'selected.pt')==s['checkpoint_sha256']
        frozen.append(dict(run_id=j['run_id'],checkpoint_sha256=s['checkpoint_sha256'],selection_sha256=sha(p/'selection.json')))
    dump(public/'before_evaluation_freeze.json',dict(jobs=frozen,all12_jobs_frozen=True,source_only_selection=True,final_RX_used=False))
    ev=out/'evaluation';ev.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device('cuda:0');rows=[];counts=[];txrows=[];summary=[]
    for j in jobs:
        cp=torch.load(out/j['run_id']/'selected.pt',map_location='cpu',weights_only=False);model=build(j).to(device);model.load_state_dict(cp['model'])
        source=old['wisig']['roles']['source'];internal=old['wisig']['roles']['internal_rx'];common=dict(run_id=j['run_id'],dataset='wisig',backbone='official',fold=0,method=j['method'],seed=j['seed'],horizon=200)
        for role,rxs,days,positions in [('source_seen_date',source,[0,1],[160,200]),('source_unseen_date',source,[2],[0,200]),('internal_seen_date',internal,[0,1],[0,200]),('internal_unseen_date',internal,[2],[0,200])]:
            data=wisig(old,rxs,days,positions);gd=tensors(data,device);z=predict(model,gd['x']).cpu().numpy();pred=z.argmax(1)
            np.savez(ev/(j['run_id']+'_'+role+'.npz'),prediction=pred,y=data['y'],rx=data['rx'],day=data['day'],position=data['position'])
            cells=[];tcells=[]
            for rx in np.unique(data['rx']):
                for day in np.unique(data['day'][data['rx']==rx]):
                    mask=(data['rx']==rx)&(data['day']==day);m=metrics(z[mask],data['y'][mask]);cm=np.array(m['confusion'])
                    row=dict(**common,role=role,rx=int(rx),date=int(day),**{k:v for k,v in m.items() if k not in ['confusion','recall']});rows.append(row);cells.append(row)
                    for t in range(10):
                        tr=dict(**common,role=role,rx=int(rx),date=int(day),tx=t,support=int(cm[t].sum()),correct=int(cm[t,t]),recall=m['recall'][t]);txrows.append(tr);tcells.append(tr)
                        for p in range(10):counts.append(dict(**common,role=role,rx=int(rx),date=int(day),tx=t,predicted_tx=p,count=int(cm[t,p])))
            acc=[];f1=[];tail=[]
            for rx in rxs:
                sub=[c for c in cells if c['rx']==rx];acc.append(np.mean([c['accuracy'] for c in sub]));f1.append(np.mean([c['macro_f1'] for c in sub]))
                recalls=[[c['recall'] for c in tcells if c['rx']==rx and c['tx']==t and c['recall'] is not None] for t in range(10)];tail.append(min(np.mean(a) for a in recalls if a))
            summary.append(dict(**common,role=role,RX_count=len(acc),accuracy=float(np.mean(acc)),macro_f1=float(np.mean(f1)),RX_accuracy_q10=float(np.quantile(acc,.1)),RX_accuracy_std=float(np.std(acc,ddof=1)),mean_RX_worst_TX_recall=float(np.mean(tail)),RX_worst_TX_recall_q10=float(np.quantile(tail,.1))))
        print('evaluated',j['run_id'],flush=True)
    for name,data in [('per_RX_date.csv',rows),('per_RX_TX_date.csv',txrows),('confusion_counts.csv',counts),('per_seed200.csv',summary)]:table(public/name,data)
    torch.cuda.synchronize();seconds=time.perf_counter()-tick
    assert execution['reserved_gpu_hours']+seconds/3600+.01<3
    dump(public/'evaluation_cost.json',dict(seconds=seconds,reserved_GPUh=seconds/3600,frozen_horizon_jobs=6,frozen_total_jobs=12,final_RX_read=False,last_date_read=False,equalized_read=False,diagnostic_outer_evaluations=0))

if __name__=='__main__':main()
