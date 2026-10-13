#!/usr/bin/env python3
"""One registered job; training never loads external/internal evaluation RX."""
import argparse,copy,csv,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from torch.func import functional_call
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from wisig_w01_common import OfficialCNN,sha,dump
from receiver_r01.core import (FeatureCNN,WiSigResNet,payload,regularization,mldg_loss,pcgrad_backward,
                              balanced_schedule,meta_masks,predict,metrics)
from receiver_r01.data import wisig,shen,groups
from receiver_r01.third_party.swa_utils import AveragedModel
from receiver_r01.third_party.swad import LossValley


def table(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def build(job):
    if job['dataset']=='shen':return FeatureCNN()
    return OfficialCNN(10) if job['backbone']=='official' else WiSigResNet()


def tensors(data,device):return {k:torch.as_tensor(v,device=device) for k,v in data.items()}


def fitting_data(reg,job):
    if job['dataset']=='wisig':
        rxs=reg['wisig_inner_train'] if job['stage']=='search' else reg['wisig']['roles']['source']
        train=wisig(reg,rxs,[0,1],[0,120]);stats=None
        val=wisig(reg,reg['wisig_inner_held'],[0,1],[0,160]) if job['stage']=='search' else wisig(reg,rxs,[0,1],[120,160])
    else:
        fold=reg['shen_folds'][job['fold']]
        keys=fold['inner_train'] if job['stage']=='search' else fold['inner_train']+fold['inner_validation']
        train,stats=shen(reg,keys)
        val,_=shen(reg,fold['inner_validation'] if job['stage']=='search' else keys,stats,'all' if job['stage']=='search' else 'val')
    assert np.isfinite(train['x']).all() and np.isfinite(val['x']).all()
    # Disjoint packet identity within each RX/date.
    tr=set(zip(train['rx'].tolist(),train['day'].tolist(),train['y'].tolist(),train['position'].tolist()))
    va=set(zip(val['rx'].tolist(),val['day'].tolist(),val['y'].tolist(),val['position'].tolist()))
    assert not tr&va
    return train,val,stats


def validate(model,data):
    z=predict(model,data['x']).detach().cpu().numpy();yy=data['y'].cpu().numpy();rr=data['rx'].cpu().numpy()
    ms=[metrics(z[rr==r],yy[rr==r]) for r in np.unique(rr)]
    return dict(CE=float(np.mean([m['CE'] for m in ms])),accuracy=float(np.mean([m['accuracy'] for m in ms])),
                macro_f1=float(np.mean([m['macro_f1'] for m in ms])))


def probe_before(model,x,y,rx,day,train,held,alpha):
    """Independent read-only deterministic probe; never used by training or selection."""
    model.eval();params=dict(model.named_parameters())
    lt=F.cross_entropy(model(x[train]),y[train]);lh=F.cross_entropy(model(x[held]),y[held])
    gt=torch.autograd.grad(lt,tuple(params.values()));gh=torch.autograd.grad(lh,tuple(params.values()))
    dot=sum((a*b).sum() for a,b in zip(gt,gh));nt=sum(a.square().sum() for a in gt);nh=sum(b.square().sum() for b in gh)
    cosine=float(dot/(nt*nh).sqrt().clamp_min(1e-12))
    new={name:p-alpha*g for (name,p),g in zip(params.items(),gt)}
    with torch.no_grad():
        before=F.cross_entropy(model(x),y,reduction='none')
        after=F.cross_entropy(functional_call(model,new,(x,)),y,reduction='none')
    return dict(before=before,inner=after,cosine=cosine,source_inner_delta=float((after[train]-before[train]).mean()))


def probe_after(model,x,y,rx,day,held,probe,epoch,step):
    model.eval()
    with torch.no_grad():after=F.cross_entropy(model(x),y,reduction='none')
    rows=[]
    for r in torch.unique(rx[held]).tolist():
        for d in torch.unique(day[(rx==r)&held]).tolist():
            for t in range(10):
                mask=(rx==r)&(day==d)&(y==t)&held
                if not mask.any():continue
                a=float(probe['before'][mask].mean());b=float(probe['inner'][mask].mean());c=float(after[mask].mean())
                rows.append(dict(epoch=epoch,step=step,rx=int(r),date=int(d),tx=t,n=int(mask.sum()),
                    pre_CE=a,inner_post_CE=b,outer_post_CE=c,inner_delta=b-a,outer_delta=c-a,
                    gradient_cosine=probe['cosine'],meta_train_inner_delta=probe['source_inner_delta']))
    return rows


def run(stage,index):
    tick=time.perf_counter();reg=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text())
    job=dict(reg[stage+'_jobs'][index]);run_dir=ROOT/'outputs/receiver_r01'/stage/job['run_id']
    run_dir.mkdir(parents=True,exist_ok=False);steps=0
    dump(run_dir/'status.json',dict(state='running',pid=os.getpid(),steps=0))
    try:
        seal=json.loads((ROOT/'configs/receiver_r01/pre_execution_seal.json').read_text())
        for path,h in seal['files'].items():assert sha(ROOT/path)==h,path
        if stage=='formal':
            sel=json.loads((ROOT/'outputs/receiver_r01/selection.frozen.json').read_text())
            job['lr']=sel['choices'][f"{job['dataset']}_{job['backbone']}_CV{job['fold']}_{job['method']}"]['lr']
        torch.set_num_threads(2);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        device=torch.device('cuda:0');model=build(job).to(device);init=payload(model)
        opt=torch.optim.Adam(model.parameters(),lr=job['lr'],eps=1e-7 if job['dataset']=='wisig' else 1e-8)
        train,val,stats=fitting_data(reg,job);group=groups(train)
        schedule=balanced_schedule(group,job['epochs'],job['seed']);np.save(run_dir/'schedule.npy',schedule)
        data=tensors(train,device);vdata=tensors(val,device)
        dump(run_dir/'config.json',dict(job,initial_sha256=init,parameters=sum(p.numel() for p in model.parameters()),
            schedule_sha256=sha(run_dir/'schedule.npy'),scaler=stats,train_RX=np.unique(train['rx']).tolist(),
            validation_RX=np.unique(val['rx']).tolist(),train_packets=len(train['y']),validation_packets=len(val['y']),
            torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),gpu_index=os.environ.get('CUDA_VISIBLE_DEVICES'),
            registration_sha256=sha(ROOT/'configs/receiver_r01/registration.json'),
            fitting_data_sha256={k:__import__('hashlib').sha256(v.tobytes()).hexdigest() for k,v in train.items()}))
        rng=np.random.default_rng(job['seed']+997);diag_rng=np.random.default_rng(job['seed']+1997)
        n_groups=int(group.max()+1);q=torch.full((n_groups,),-np.log(n_groups),device=device)
        gg=torch.tensor(group,device=device);history=[];diagnostics=[];best=float('inf');best_epoch=0;best_state=None
        segment=None;valley=None
        if job['method']=='swad':
            evaluator=SimpleNamespace(logger=SimpleNamespace(error=lambda s:print(s,flush=True)))
            valley=LossValley(evaluator,3,6,.3);segment=AveragedModel(model)
        torch.cuda.reset_peak_memory_stats();fit_start=time.perf_counter()
        for epoch in range(job['epochs']):
            torch.cuda.synchronize();etick=time.perf_counter();total=0.
            for batch_id,ids in enumerate(schedule[epoch]):
                ix=torch.tensor(ids,device=device);x=data['x'][ix];y=data['y'][ix];rx=data['rx'][ix];day=data['day'][ix]
                tr,he=meta_masks(rx,day,rng)
                do_probe=batch_id==0 and ((epoch+1)%(5 if job['dataset']=='wisig' else 1)==0)
                probe=None
                if do_probe:
                    dt,dh=tr,he;probe=probe_before(model,x,y,rx,day,dt,dh,job['alpha'])
                model.train();opt.zero_grad(set_to_none=True)
                if job['method']=='mldg_rx':loss=mldg_loss(model,x,y,tr,he,job['alpha'],job['beta'])
                elif job['method']=='mldg_rx_date':
                    d0,d1=diag_rng.permutation([0,1])
                    loss=(mldg_loss(model,x,y,tr&(day==d0),he&(day==d1),job['alpha'],job['beta'])+
                          mldg_loss(model,x,y,he&(day==d0),tr&(day==d1),job['alpha'],job['beta']))/2
                elif job['method']=='pcgrad':loss=pcgrad_backward(model,x,y,tr,he)
                else:
                    ps=F.cross_entropy(model(x),y,reduction='none')
                    if job['method']=='groupdro':
                        order=torch.argsort(gg[ix]);means=ps[order].reshape(n_groups,2).mean(1)
                        with torch.no_grad():q.add_(.01*means.detach());q.sub_(torch.logsumexp(q,0))
                        loss=(q.exp()*means).sum()+regularization(model)
                    else:loss=ps.mean()+regularization(model)
                assert torch.isfinite(loss)
                if job['method']!='pcgrad':loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();steps+=1
                if segment is not None:segment.update_parameters(model,step=steps)
                total+=float(loss.detach())
                if probe is not None:diagnostics+=probe_after(model,x,y,rx,day,dh,probe,epoch+1,steps)
            result=validate(model,vdata)
            if result['CE']<best:
                best=result['CE'];best_epoch=epoch+1;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if valley is not None:
                valley.update_and_evaluate(segment,result['accuracy'],result['CE'],lambda *a:None);segment=AveragedModel(model)
            torch.cuda.synchronize()
            history.append(dict(epoch=epoch+1,steps=steps,objective=total/len(schedule[epoch]),seconds=time.perf_counter()-etick,**{'source_val_'+k:v for k,v in result.items()}))
            table(run_dir/'history.csv',history);table(run_dir/'update_diagnostics.csv',diagnostics)
            dump(run_dir/'status.json',dict(state='running',pid=os.getpid(),epoch=epoch+1,steps=steps))
            print(json.dumps(dict(run=job['run_id'],**history[-1])),flush=True)
        assert steps==job['epochs']*schedule.shape[1]
        fit_seconds=time.perf_counter()-fit_start
        if valley is not None:
            converged=valley.is_converged;dead=valley.dead_valley
            final=valley.get_final_model();model.load_state_dict(final.module.state_dict())
            selected=dict(rule='published SWAD LossValley',converged=converged,dead_valley=dead,start_step=final.start_step,end_step=final.end_step,n_segments=int(final.n_averaged),threshold=valley.threshold)
        else:model.load_state_dict(best_state);selected=dict(rule='minimum source-only validation receiver-equal CE',epoch=best_epoch)
        final_validation=validate(model,vdata);train_scores=validate(model,data)
        torch.save(dict(model=model.state_dict(),job=job,scaler=stats,selection=selected),run_dir/'selected.pt')
        dump(run_dir/'selection.json',dict(selected,source_validation=final_validation,frozen_train_scores=train_scores,
                                          checkpoint_sha256=sha(run_dir/'selected.pt')))
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(5):predict(model,data['x'][:min(256,len(data['x']))])
        torch.cuda.synchronize();inference_seconds=(time.perf_counter()-start)/5
        dump(run_dir/'status.json',dict(state='completed',steps=steps,epochs=job['epochs'],seconds=time.perf_counter()-tick,
            fitting_seconds=fit_seconds,inference_seconds_per_packet=inference_seconds/min(256,len(data['x'])),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(run_dir/'selected.pt'),
            final_validation=final_validation,swad=selected if valley is not None else None))
    except BaseException as e:
        dump(run_dir/'status.json',dict(state='failed',error=repr(e),steps=steps,seconds=time.perf_counter()-tick));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['search','formal'],required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();run(a.stage,a.index)
