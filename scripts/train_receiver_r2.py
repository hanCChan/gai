#!/usr/bin/env python3
"""One preregistered R2-A fit. Horizon-only replay or isolated Adam diagnostics."""
import argparse,copy,csv,json,os,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from train_receiver_r01 import build,tensors,validate,fitting_data,table
from receiver_r01.core import payload,regularization,mldg_loss,balanced_schedule,meta_masks
from receiver_r01.data import groups
from receiver_r2.data import diagnostic_data,hashes,subset
from receiver_r2.measurement import clone_parameters,state_hash,squared_norm,risk_measurement
from wisig_w01_common import sha,dump


def save_state(path,model,opt,job,epoch,rng,drng,**extra):
    torch.save(dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),job=job,epoch=epoch,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),meta_rng=copy.deepcopy(rng.bit_generator.state),
        date_rng=copy.deepcopy(drng.bit_generator.state),**extra),path)


def run(index):
    tick=time.perf_counter();registration=json.loads((ROOT/'configs/receiver_r2/registration.json').read_text());job=registration['jobs'][index]
    out=ROOT/'outputs/receiver_r2'/job['run_id'];out.mkdir(parents=True,exist_ok=False);steps=0
    dump(out/'status.json',dict(state='running',pid=os.getpid(),steps=0))
    try:
        seal=json.loads((ROOT/'configs/receiver_r2/pre_execution_seal.json').read_text())
        for path,h in seal['files'].items():assert sha(ROOT/path)==h,path
        reg=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text());historical=Path(registration['historical_worktree'])
        assert sha(ROOT/'configs/receiver_r01/registration.json')==registration['historical_registration_sha256']
        torch.set_num_threads(2);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed']);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        device=torch.device('cuda:0');model=build(job).to(device);initial=payload(model)
        opt=torch.optim.Adam(model.parameters(),lr=job['lr'],eps=1e-7 if job['dataset']=='wisig' else 1e-8)
        if job['stage']=='horizon':
            oldjob=dict(job,stage='formal');train,val,stats=fitting_data(reg,oldjob);pools={'fit':train,'validation':val}
        else:pools,stats=diagnostic_data(reg,job);train=pools['fit'];val=pools['validation']
        schedule=balanced_schedule(groups(train),job['epochs'],job['seed']);np.save(out/'schedule.npy',schedule)
        data=tensors(train,device);vdata=tensors(val,device)
        rng=np.random.default_rng(job['seed']+997);drng=np.random.default_rng(job['seed']+1997)
        domains=np.unique(train['rx']);parts=[]
        if job['stage']=='diagnostic':
            prng=np.random.default_rng(job['seed']+997)
            nA,nB,nC=registration['rx_partitions'][job['dataset']]
            for _ in range(job['epochs']*len(schedule[0])):
                d=prng.permutation(domains);parts.append([d[:nA],d[nA:nA+nB],d[nA+nB:]])
            # Domain groups are different sizes in WiSig/Shen but fixed within this job.
            np.save(out/'RX_partitions.npy',np.array([[np.concatenate(a) for a in parts]])[0])
            riskdata={n:tensors(pools[n],device) for n in ['C1','C2']}
        historical_config=None;oldhist=None
        if job['stage']=='horizon':
            oldpath=historical/'outputs/receiver_r01/formal'/job['historical_run_id']
            historical_config=json.loads((oldpath/'config.json').read_text());assert initial==historical_config['initial_sha256']
            assert hashes(train)==historical_config['fitting_data_sha256']
            assert np.array_equal(schedule[:100],np.load(oldpath/'schedule.npy'))
            with (oldpath/'history.csv').open() as f:oldhist=list(csv.DictReader(f))
        dump(out/'config.json',dict(job,initial_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),scaler=stats,
            fitting_data_sha256=hashes(train),pools={n:dict(packets=len(a['y']),RX=np.unique(a['rx']).tolist(),hashes=hashes(a)) for n,a in pools.items()},
            schedule_sha256=sha(out/'schedule.npy'),rx_partitions_sha256=sha(out/'RX_partitions.npy') if parts else None,
            optimizer=dict(type='Adam',lr=job['lr'],epsilon=1e-7 if job['dataset']=='wisig' else 1e-8,clip=5),
            numpy=np.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name(),gpu_index=os.environ.get('CUDA_VISIBLE_DEVICES')))
        history=[];updates=[];riskrows=[];best=float('inf');best_epoch=0;best_state=None;bestopt=None;best100=None
        cumulative_delta=0.;effective_packets=0;torch.cuda.reset_peak_memory_stats();fit_start=time.perf_counter()
        for epoch in range(job['epochs']):
            torch.cuda.synchronize();etick=time.perf_counter();total=0.
            for batch,ids in enumerate(schedule[epoch]):
                ix=torch.tensor(ids,device=device);x=data['x'][ix];y=data['y'][ix];rx=data['rx'][ix];day=data['day'][ix]
                if parts:
                    a,b,c=parts[steps];tr=torch.isin(rx,torch.tensor(a,device=device));he=torch.isin(rx,torch.tensor(b,device=device));cc=c
                else:tr,he=meta_masks(rx,day,rng)
                model.train();opt.zero_grad(set_to_none=True)
                if job['method']=='mldg_rx':loss=mldg_loss(model,x,y,tr,he,job['alpha'],job['beta'])
                elif job['method']=='mldg_rx_date':
                    d0,d1=drng.permutation([0,1])
                    loss=(mldg_loss(model,x,y,tr&(day==d0),he&(day==d1),job['alpha'],job['beta'])+
                          mldg_loss(model,x,y,he&(day==d0),tr&(day==d1),job['alpha'],job['beta']))/2
                else:
                    mask=tr|he if parts else torch.ones(len(y),dtype=torch.bool,device=device)
                    loss=F.cross_entropy(model(x[mask]),y[mask])+regularization(model)
                assert torch.isfinite(loss);loss.backward()
                observe=batch==0;track=bool(parts) or observe
                before=clone_parameters(model) if track else None
                before_opt=copy.deepcopy(opt.state_dict()) if parts and observe else None
                gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
                clipped_norm=squared_norm(p.grad for p in model.parameters())**.5 if track else None
                gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()} if parts and observe else None
                opt.step();steps+=1;total+=float(loss.detach())
                used=int((tr|he).sum()) if parts else len(y);effective_packets+=used
                if track:
                    delta={n:p.detach()-before[n] for n,p in model.named_parameters()};dn=squared_norm(delta.values())**.5
                    if parts:cumulative_delta+=dn
                if observe:
                    moments=[s['exp_avg'] for s in opt.state.values()];seconds=[s['exp_avg_sq'] for s in opt.state.values()]
                    updates.append(dict(epoch=epoch+1,step=steps,gradient_norm_preclip=gradient_norm,gradient_norm_postclip=clipped_norm,clip_applied=gradient_norm>5,
                        candidate_delta_L2=dn,accepted_delta_L2=dn,scale=1.,moment1_L2=squared_norm(moments)**.5,moment2_L2=squared_norm(seconds)**.5,
                        optimizer_state_sha256=state_hash(opt.state_dict()),adam_step_min=min(float(s['step']) for s in opt.state.values()),adam_step_max=max(float(s['step']) for s in opt.state.values()),
                        stage=job['stage'],effective_packets_this_step=used))
                    if parts:
                        cpools={n:{k:v[torch.isin(p['rx'],torch.tensor(cc,device=device))] for k,v in p.items()} for n,p in riskdata.items()}
                        rows,raw=risk_measurement(model,before,cpools,epoch+1,steps,job['seed']);riskrows+=rows
                        snapshot=out/f'actual_adam_epoch{epoch+1:03d}.pt'
                        save_state(snapshot,model,opt,job,epoch+1,rng,drng,pre_parameters=before,pre_optimizer=before_opt,clipped_gradient=gradients,parameter_delta=delta,risk_packet_measurements=raw)
                        updates[-1]['snapshot_sha256']=sha(snapshot)
            result=validate(model,vdata)
            if result['CE']<best:
                best=result['CE'];best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict());bestopt=copy.deepcopy(opt.state_dict())
            torch.cuda.synchronize()
            row=dict(epoch=epoch+1,steps=steps,objective=total/len(schedule[epoch]),seconds=time.perf_counter()-etick,**{'source_val_'+k:v for k,v in result.items()})
            if oldhist is not None and epoch<100:
                for k in ['objective','source_val_CE','source_val_accuracy','source_val_macro_f1']:
                    assert abs(float(oldhist[epoch][k])-row[k])<1e-7,('Historical100 prefix mismatch',epoch+1,k,oldhist[epoch][k],row[k])
            history.append(row);table(out/'history.csv',history);table(out/'updates.csv',updates);table(out/'risks.csv',riskrows)
            if job['stage']=='horizon' and epoch+1==100:
                prior=torch.load(oldpath/'selected.pt',map_location='cpu',weights_only=False)
                assert state_hash(best_state)==state_hash(prior['model'])
                best100=dict(epoch=best_epoch,CE=best,parameters_sha256=state_hash(best_state),historical_checkpoint_sha256=sha(oldpath/'selected.pt'),history_prefix_exact=True)
                dump(out/'historical100_replay.json',best100)
                torch.save(dict(model=best_state,optimizer=bestopt,job=dict(job,epochs=100),scaler=stats,selection={'epoch':best_epoch}),out/'selected100.pt')
                save_state(out/'last100.pt',model,opt,job,epoch+1,rng,drng)
            dump(out/'status.json',dict(state='running',pid=os.getpid(),epoch=epoch+1,steps=steps));print(json.dumps(dict(run=job['run_id'],**row)),flush=True)
        fitting_seconds=time.perf_counter()-fit_start
        save_state(out/'last.pt',model,opt,job,job['epochs'],rng,drng)
        model.load_state_dict(best_state);opt.load_state_dict(bestopt)
        selection=dict(rule='minimum source-only validation receiver-equal CE',epoch=best_epoch)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),job=job,scaler=stats,selection=selection),out/'selected.pt')
        dump(out/'selection.json',dict(selection,source_validation=validate(model,vdata),frozen_train_scores=validate(model,data),checkpoint_sha256=sha(out/'selected.pt')))
        dump(out/'status.json',dict(state='completed',steps=steps,epochs=job['epochs'],seconds=time.perf_counter()-tick,fitting_seconds=fitting_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(out/'selected.pt'),cumulative_effective_delta_L2_sum=cumulative_delta if parts else None,
            cumulative_effective_packet_gradient_uses=effective_packets,acceptance_rate=1.,mean_scale=1.,rejection_rate=0.,historical100_prefix_verified=best100 is not None if job['stage']=='horizon' else None))
    except BaseException as error:
        dump(out/'status.json',dict(state='failed',error=repr(error),steps=steps,seconds=time.perf_counter()-tick));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();run(a.index)
