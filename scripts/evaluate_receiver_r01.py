#!/usr/bin/env python3
"""One frozen evaluation pass after every formal checkpoint is fixed."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from wisig_w01_common import sha,dump
from receiver_r01.data import wisig,shen
from receiver_r01.core import predict,metrics
from train_receiver_r01 import build,tensors,table


def main():
    tick=time.perf_counter();reg=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text());out=ROOT/'outputs/receiver_r01';public=ROOT/'reports/receiver_r01/2026-10-13'
    execution=json.loads((out/'execution.json').read_text());assert execution['state']=='completed' and execution['fitting_jobs']==85
    # All checkpoints fixed and all LR choices source-only before ANY internal evaluation load.
    frozen=[]
    for job in reg['formal_jobs']:
        run=out/'formal'/job['run_id'];status=json.loads((run/'status.json').read_text())
        assert status['state']=='completed' and sha(run/'selected.pt')==status['checkpoint_sha256']
        frozen.append(dict(run_id=job['run_id'],checkpoint_sha256=status['checkpoint_sha256'],selection_sha256=sha(run/'selection.json')))
    sel=json.loads((out/'selection.frozen.json').read_text());assert sel['source_only'] and not sel['internal_RX_used'] and not sel['outer_RX_used']
    assert not (out/'evaluation').exists();(out/'evaluation').mkdir()
    dump(public/'before_evaluation_freeze.json',dict(all_formal_checkpoints=frozen,selection_sha256=sha(out/'selection.frozen.json'),
        formal_jobs=51,no_internal_RX_selection=True,no_final_data_access=True))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device('cuda:0')
    per_rx=[];per_tx=[];confusions=[];inference=[];summaries=[]
    for job in reg['formal_jobs']:
        run=out/'formal'/job['run_id'];cp=torch.load(run/'selected.pt',map_location='cpu',weights_only=False)
        model=build(job).to(device);model.load_state_dict(cp['model']);model.eval();datasets=[]
        if job['dataset']=='wisig':
            source=reg['wisig']['roles']['source'];internal=reg['wisig']['roles']['internal_rx']
            datasets=[('source_seen_date',wisig(reg,source,[0,1],[160,200])),
                      ('source_unseen_date',wisig(reg,source,[2],[0,200])),
                      ('internal_seen_date',wisig(reg,internal,[0,1],[0,200])),
                      ('internal_unseen_date',wisig(reg,internal,[2],[0,200]))]
        else:
            keys=reg['shen_folds'][job['fold']]['outer_evaluation'];data,_=shen(reg,keys,cp['scaler'],'all');datasets=[('source_pool_outer_RX',data)]
        common={k:job[k] for k in ['dataset','backbone','fold','method','seed','run_id']}
        for role,data in datasets:
            gd=tensors(data,device);torch.cuda.synchronize();itick=time.perf_counter()
            z=predict(model,gd['x']).cpu().numpy();torch.cuda.synchronize();seconds=time.perf_counter()-itick
            inference.append(dict(**common,role=role,packets=len(data['y']),seconds=seconds,seconds_per_packet=seconds/len(data['y'])))
            pred=z.argmax(1);path=out/'evaluation'/(job['run_id']+'_'+role+'.npz')
            np.savez(path,prediction=pred,y=data['y'],rx=data['rx'],day=data['day'],position=data['position'])
            cells=[]
            for rx in np.unique(data['rx']):
                for d in np.unique(data['day'][data['rx']==rx]):
                    mask=(data['rx']==rx)&(data['day']==d);m=metrics(z[mask],data['y'][mask]);cm=np.array(m['confusion'])
                    name=reg['wisig']['rx_names'][rx] if job['dataset']=='wisig' else reg['shen_allowed'][rx]
                    row=dict(**common,role=role,rx=int(rx),receiver=name,date=int(d),**{k:v for k,v in m.items() if k not in ['recall','confusion']})
                    per_rx.append(row);cells.append(row)
                    for tx in range(10):
                        per_tx.append(dict(**common,role=role,rx=int(rx),receiver=name,date=int(d),tx=tx,
                            support=int(cm[tx].sum()),correct=int(cm[tx,tx]),recall=m['recall'][tx]))
                        for guessed in range(10):
                            confusions.append(dict(**common,role=role,rx=int(rx),date=int(d),tx=tx,predicted_tx=guessed,count=int(cm[tx,guessed])))
            # Average dates equally within RX, then receivers equally: no pooled weighting.
            acc=[];f1=[];worst=[]
            for rx in sorted({c['rx'] for c in cells}):
                sub=[c for c in cells if c['rx']==rx];acc.append(np.mean([c['accuracy'] for c in sub]));f1.append(np.mean([c['macro_f1'] for c in sub]));worst.append(np.mean([c['worst_tx_recall'] for c in sub]))
            summaries.append(dict(**common,role=role,RX_count=len(acc),accuracy=float(np.mean(acc)),macro_f1=float(np.mean(f1)),
                RX_accuracy_q10=float(np.quantile(acc,.1)),RX_accuracy_std=float(np.std(acc,ddof=1)),
                mean_RX_worst_TX_recall=float(np.mean(worst)),RX_worst_TX_recall_q10=float(np.quantile(worst,.1)),
                minimum_RX_accuracy=float(np.min(acc)),complete_RX_date_cells=sum(c['complete_coverage'] for c in cells),RX_date_cells=len(cells)))
        del model
        print('evaluated',job['run_id'],flush=True)
    for name,rows in [('per_RX_date.csv',per_rx),('per_RX_TX_date.csv',per_tx),('confusion_counts.csv',confusions),('per_seed.csv',summaries),('inference_cost.csv',inference)]:table(public/name,rows)
    total=time.perf_counter()-tick
    cap=json.loads((ROOT/'configs/receiver_r01/budget_amendment.json').read_text())['new_cap_reserved_V100_GPUh']
    assert execution['reserved_gpu_hours']+total/3600+.01<cap
    dump(public/'evaluation_cost.json',dict(seconds=total,reserved_gpu_hours=total/3600,frozen_jobs=51,
        final_RX_read=False,last_date_read=False,equalized_read=False,source_pool_is_development_not_independent_blind=True))

if __name__=='__main__':main()
