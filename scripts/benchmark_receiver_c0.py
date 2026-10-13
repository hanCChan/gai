#!/usr/bin/env python3
"""Finite reset single-step timing; no fitted or evolving model is produced."""
import argparse,copy,json,os,sys,time,csv
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from receiver_r2.data import diagnostic_data
from receiver_r2.controller import controlled_adam_step
from receiver_r2.measurement import clone_parameters,state_hash,commit_scaled
from receiver_c0.optimized import optimized_adam_step
from receiver_r01.core import mldg_loss
from train_receiver_r01 import build,tensors


def main(card):
    tick=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    reg=json.loads((ROOT/'configs/receiver_c0/registration.json').read_text());old=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text());device=torch.device('cuda:0');rows=[];checks=[];gradrows=[]
    for ds in ['wisig','shen']:
        entry=next(e for e in reg['diagnostic_runs'] if e['job']['dataset']==ds and e['job']['fold']==0 and e['job']['method'].startswith('mldg'));job=entry['job'];path=Path(entry['run_path']);pools,_=diagnostic_data(old,job);data={n:tensors(p,device) for n,p in pools.items()};schedule=np.load(path/'schedule.npy');parts=np.load(path/'RX_partitions.npy');model=build(job).to(device);opt=torch.optim.Adam(model.parameters(),lr=job['lr'],eps=1e-7 if ds=='wisig' else 1e-8)
        for epoch in reg['timing_only']['snapshot_epochs'][ds]:
            saved=torch.load(path/f'actual_adam_epoch{epoch:03d}.pt',map_location='cpu',weights_only=False);before={n:p.to(device) for n,p in saved['pre_parameters'].items()};sindex=(epoch-1)*schedule.shape[1];partition=parts[sindex];nA,nB,nC=(5,5,4) if ds=='wisig' else (2,1,2);A=partition[:nA];B=partition[nA:nA+nB];C=partition[-nC:]
            risk={n:{k:v[torch.isin(p['rx'],torch.tensor(C,device=device))] for k,v in p.items()} for n,p in data.items() if n in ['C1','C2']}
            def reset():
                model.load_state_dict(before);opt.load_state_dict(copy.deepcopy(saved['pre_optimizer']));opt.zero_grad(set_to_none=True)
                for name,p in model.named_parameters():p.grad=saved['clipped_gradient'][name].to(device).clone()
                model.train()
            reset();ids=torch.tensor(schedule[epoch-1,0],device=device);batch={k:v[ids] for k,v in data['fit'].items()};tr=torch.isin(batch['rx'],torch.tensor(A,device=device));held=torch.isin(batch['rx'],torch.tensor(B,device=device))
            # Full training objective timing uses a fresh isolated documented mask;
            # original pre-gradient RNG was not saved, so this is not a claimed replay.
            model.load_state_dict(before);opt.zero_grad(set_to_none=True);torch.manual_seed(job['seed']+epoch);torch.cuda.synchronize();start=time.perf_counter()
            if ds=='wisig':
                loss=(mldg_loss(model,batch['x'],batch['y'],tr&(batch['day']==0),held&(batch['day']==1))+
                      mldg_loss(model,batch['x'],batch['y'],held&(batch['day']==0),tr&(batch['day']==1)))/2
            else:loss=mldg_loss(model,batch['x'],batch['y'],tr,held)
            loss.backward();pre_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True));torch.cuda.synchronize();gradient_seconds=time.perf_counter()-start
            gradrows.append(dict(gpu=card,dataset=ds,epoch=epoch,gradient_seconds=gradient_seconds,preclip_gradient_norm=pre_norm,A_RX=list(map(int,A)),B_RX=list(map(int,B)),C_RX=list(map(int,C)),original_Dropout_gradient_replayed=False))
            # Saved clipped gradients reproduce exactly the original actual proposal.
            reset();opt.step();assert state_hash(model.state_dict())==state_hash(saved['model'])
            outcomes={}
            for name in reg['timing_only']['mechanisms']:
                reset();forward=[0,0]
                handle=model.register_forward_hook(lambda m,args,out:(forward.__setitem__(0,forward[0]+1),forward.__setitem__(1,forward[1]+len(args[0]))) and None)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                if name=='standard':opt.step();result={'scale':1.}
                elif name.startswith('legacy'):result=controlled_adam_step(model,opt,risk['C1'],risk['C2'],epoch,sindex+1,job['seed'],name.endswith('multi'))
                else:result=optimized_adam_step(model,opt,risk['C1'],None,name.endswith('multi'),False)
                torch.cuda.synchronize();seconds=time.perf_counter()-start;handle.remove()
                outcomes[name]=dict(scale=result['scale'],parameters=state_hash(model.state_dict()),optimizer=state_hash(opt.state_dict()))
                rows.append(dict(gpu=card,dataset=ds,epoch=epoch,mechanism=name,decision_seconds=seconds,gradient_seconds=gradient_seconds,complete_opportunity_seconds=gradient_seconds+seconds,
                    C1_packets=len(risk['C1']['y']),C2_packets=len(risk['C2']['y']),forward_calls=forward[0],packets_forwarded=forward[1],scale=result['scale'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),trajectory_updates=0))
            for suffix in ['mean','multi']:assert outcomes['legacy_'+suffix]==outcomes['optimized_'+suffix]
            # One fixed sparse diagnosis cost; its chosen-scale decision is identical.
            reset();torch.cuda.synchronize();start=time.perf_counter();diag=optimized_adam_step(model,opt,risk['C1'],risk['C2'],True,True);torch.cuda.synchronize();diag_seconds=time.perf_counter()-start
            assert diag['scale']==outcomes['optimized_multi']['scale'] and state_hash(model.state_dict())==outcomes['optimized_multi']['parameters'] and state_hash(opt.state_dict())==outcomes['optimized_multi']['optimizer']
            checks.append(dict(gpu=card,dataset=ds,epoch=epoch,scale=diag['scale'],legacy_optimized_parameters_moments_equal=True,sparse_C2_does_not_change_decision=True,sparse_diagnostic_seconds=diag_seconds,forward_calls=diag['forward_calls'],packets_forwarded=diag['packets_forwarded']))
        del model,opt,data;torch.cuda.empty_cache()
    result=dict(gpu=card,state='completed',seconds=time.perf_counter()-tick,timing_cases=rows,gradient_cases=gradrows,equivalence_cases=checks,formal_fits=0,continuous_model_trajectory_updates=0)
    (ROOT/f'outputs/receiver_c0/timing_gpu{card}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(gpu=card,seconds=result['seconds'],cases=len(rows),formal_fits=0)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--card',type=int,required=True);a=p.parse_args();main(a.card)
