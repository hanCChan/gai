#!/usr/bin/env python3
"""Read-only actual-Adam risks on its own fitting batch from complete new states.

This supplements C1/C2 without another fit or changing any saved trajectory.
The common train-mode masks are diagnostic masks, not a reconstruction of the
original stochastic training-objective draw. Eval-mode results are separate.
"""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from receiver_r2.data import diagnostic_data
from receiver_r2.measurement import risk_measurement
from train_receiver_r01 import build,tensors,table


def main():
    tick=time.perf_counter();reg=json.loads((ROOT/'configs/receiver_r2/registration.json').read_text());old=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text());out=ROOT/'outputs/receiver_r2';pub=ROOT/'reports/receiver_r2/r2_a';rows=[]
    pub.mkdir(parents=True,exist_ok=True)
    diagnostics=[j for j in reg['jobs'] if j['stage']=='diagnostic']
    assert len(diagnostics)==6 and all(json.loads((out/j['run_id']/'status.json').read_text())['state']=='completed' for j in diagnostics)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);device=torch.device('cuda:0')
    for j in reg['jobs']:
        if j['stage']!='diagnostic':continue
        run=out/j['run_id'];pools,_=diagnostic_data(old,j);fit=tensors(pools['fit'],device);model=build(j).to(device);schedule=np.load(run/'schedule.npy');parts=np.load(run/'RX_partitions.npy');nC=reg['rx_partitions'][j['dataset']][2]
        for path in sorted(run.glob('actual_adam_epoch*.pt')):
            s=torch.load(path,map_location=device,weights_only=False);model.load_state_dict(s['model']);step=(s['epoch']-1)*schedule.shape[1];ids=torch.tensor(schedule[s['epoch']-1,0],device=device);C=parts[step,-nC:]
            batch={k:v[ids] for k,v in fit.items()};mask=~torch.isin(batch['rx'],torch.tensor(C,device=device));batch={k:v[mask] for k,v in batch.items()}
            r,_=risk_measurement(model,s['pre_parameters'],{'fitting_AB':batch},s['epoch'],step+1,j['seed'])
            for a in r:a.update({k:j[k] for k in ['run_id','dataset','method','fold']})
            rows+=r
        print('remeasured source fitting batch',j['run_id'],flush=True)
    table(pub/'actual_fitting_batch_risks.csv',rows);torch.cuda.synchronize();seconds=time.perf_counter()-tick
    (pub/'fitting_remeasurement_cost.json').write_text(json.dumps(dict(seconds=seconds,reserved_GPUh=seconds/3600,additional_fitting_jobs=0,source_training_only=True,original_objective_dropout_draw_reconstructed=False),indent=2)+'\n')

if __name__=='__main__':main()
