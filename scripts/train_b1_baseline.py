#!/usr/bin/env python3
"""Single approved search or formal standard-baseline job, fixed10/20 only."""
import argparse,csv,hashlib,json,os,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from b1_baselines.data import SourceCIS,EpochBatches,validate_schedule
from b1_baselines.model import FeatureCNN
from b1_baselines.objectives import GroupDRO,grl,supervised_contrastive
from shen_p2a8.g1_data import fit_scaler
from shen_evidence.runner import seed_all,seed_worker,sync
from prepare_b1_execution import sha

def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def csv_write(path,records):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
def payload(model):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()

@torch.no_grad()
def evaluate(model,root,keys,allowed,stats,seed,device,destination):
    model.eval();destination.mkdir();aggregate=np.zeros((10,10),int);asset_hashes={};metrics=[]
    for key in keys:
        dataset=SourceCIS(root,[key],allowed,stats)
        loader=DataLoader(dataset,batch_size=100,shuffle=False,num_workers=2,pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed))
        values=[];labels=[]
        for x,y,_,_,_ in loader:
            z=model(x.to(device,non_blocking=True));assert torch.isfinite(z).all();values.append(z.cpu().numpy());labels.extend(y.tolist())
        logits=np.concatenate(values).astype('float64');y=np.array(labels,dtype='int64');assert np.array_equal(y,dataset.labels_all)
        c=np.zeros((10,10),int);np.add.at(c,(y,logits.argmax(1)),1);aggregate+=c
        rx=key.removeprefix('shen_').removesuffix('_train');path=destination/(rx+'.npz')
        exp=np.exp(logits-logits.max(1,keepdims=True));prob=exp/exp.sum(1,keepdims=True)
        np.savez(path,logits=logits,probabilities=prob,labels=y,rows=dataset.rows[0]);asset_hashes[path.name]=sha(path)
        f1=sum((Fraction(int(2*c[i,i]),int(c[i].sum()+c[:,i].sum())) for i in range(10)),Fraction(0))/10
        metrics.append(dict(receiver=rx,packets=len(y),accuracy=float(np.trace(c)/len(y)),macro_f1=float(f1),worst_tx_recall=float(np.diag(c).min()/200)))
    exact=sum((Fraction(int(2*aggregate[i,i]),int(aggregate[i].sum()+aggregate[:,i].sum())) for i in range(10)),Fraction(0))/10
    result=dict(accuracy=float(np.trace(aggregate)/aggregate.sum()),macro_f1=float(exact),macro_f1_fraction=dict(numerator=exact.numerator,denominator=exact.denominator),confusion_counts=aggregate.tolist(),per_RX=metrics,assets=asset_hashes,scaler=stats,evaluation_keys=keys)
    dump(destination/'manifest.json',result);return result

def run(stage,index):
    started=time.perf_counter();r=json.loads((ROOT/'configs/b1_baselines/registration.json').read_text());key='search_jobs' if stage=='search' else 'formal_jobs'
    original=r['matrix'][key][index];job=dict(original)
    if stage=='formal':
        selection=json.loads((ROOT/'outputs/b1_baselines/selection.frozen.json').read_text());assert selection['outer_evaluation_used'] is False
        choice=selection['choices'][f"CV{job['fold']}_{job['method']}"];job['lr']=choice['lr']
    out=ROOT/'outputs/b1_baselines'/stage/'runs'/job['run_id'];out.mkdir(parents=True,exist_ok=False)
    dump(out/'status.json',dict(status='running',pid=os.getpid(),optimizer_steps=0,stage=stage))
    steps=0
    try:
        assert torch.__version__=='2.5.1+cu121' and np.__version__=='1.26.4'
        assert 0<=index<16 and job['method'] in r['methods'] and stage in ('search','formal')
        seal=json.loads((ROOT/'configs/b1_baselines/pre_execution_seal.json').read_text())
        for path,digest in seal['files_sha256'].items():assert sha(ROOT/path)==digest,path
        preflight=json.loads((ROOT/'outputs/b1_baselines/preflight.json').read_text());assert sha(ROOT/'outputs/b1_baselines/preflight.json')==seal['preflight_sha256']
        h=Path(r['history_root']);allowed=[f'shen_{rx}_train' for rx in r['source_receivers']]
        eval_keys=job['validation_keys'] if stage=='search' else job['evaluation_keys']
        assert not set(job['train_keys'])&set(eval_keys)
        opened=job['train_keys']+eval_keys
        if stage=='search':assert not set(opened)&set(job['outer_keys_never_read'])
        for key in opened:
            assert key in allowed
            for name in ('strict_cis.npy','rows.npy','labels.npy','manifest.json'):
                p=h/'outputs/shen_p2a8/g1_cache_v1'/key/name;assert sha(p)==r['feature_assets'][str(p.relative_to(h))]['sha256']
        stats=fit_scaler(h,job['train_keys'],'strict_cis');dataset=SourceCIS(h,job['train_keys'],allowed,stats)
        schedule_key=f"{stage}_CV{job['fold']}_seed{job['seed']}";smeta=preflight['schedules'][schedule_key]
        assert job['train_keys']==smeta['train_keys'] and job['epochs']==smeta['epochs']
        spath=ROOT/smeta['path'];assert sha(spath)==smeta['sha256'];schedule=np.load(spath,allow_pickle=False)
        validate_schedule(schedule,dataset.group_ids,job['epochs']);sampler=EpochBatches(schedule)
        seed_all(job['seed']);device=torch.device('cuda:0');model=FeatureCNN().to(device);initial=payload(model)
        head=None;dro=None
        if job['method']=='grl':
            assert job['rx_classifier_classes']==len(job['train_keys'])
            head=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,job['rx_classifier_classes'])).to(device)
            for layer in head.modules():
                if isinstance(layer,nn.Linear):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
        if job['method']=='group_dro_rx_tx':dro=GroupDRO(job['groups'],.01).to(device)
        parameters=list(model.parameters())+(list(head.parameters()) if head is not None else [])
        optimizer=torch.optim.Adam(parameters,lr=job['lr'],weight_decay=0)
        config=dict(job,stage=stage,scaler=stats,initial_base_payload_sha256=initial,base_parameters=sum(p.numel() for p in model.parameters()),aux_parameters=0 if head is None else sum(p.numel() for p in head.parameters()),schedule_sha256=smeta['sha256'],schedule_key=schedule_key,registration_sha256=sha(ROOT/'configs/b1_baselines/registration.json'),selection_sha256=sha(ROOT/'outputs/b1_baselines/selection.frozen.json') if stage=='formal' else None,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,numpy=np.__version__,python=sys.version,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),source_keys_opened=opened,outer_evaluation_during_search=False)
        assert config['base_parameters']==658314
        dump(out/'config.json',config)
        loader=DataLoader(dataset,batch_sampler=sampler,num_workers=2,pin_memory=True,persistent_workers=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(job['seed']))
        history=[]
        for epoch in range(job['epochs']):
            model.train();sampler.epoch=epoch
            if head is not None:head.train()
            total=correct=count=0;ce_total=aux_total=0.;qmin=qmax=entropy=None;sync(device);tick=time.perf_counter()
            for x,y,g,rx,_ in loader:
                x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);g=g.to(device,non_blocking=True);rx=rx.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True);z,features=model.forward_with_features(x);per_sample=F.cross_entropy(z,y,reduction='none');ce=per_sample.mean();aux=torch.zeros((),device=device)
                if dro is not None:
                    loss,_,weights=dro(per_sample,g);qmin=float(weights.min());qmax=float(weights.max());entropy=float(-(weights*dro.log_weights).sum())
                elif head is not None:
                    aux=F.cross_entropy(head(grl(features,.1)),rx);loss=ce+aux
                elif job['method']=='supcon':aux=supervised_contrastive(features,y,.1);loss=ce+.1*aux
                else:loss=ce
                if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True);optimizer.step();steps+=1
                count+=len(y);correct+=int((z.detach().argmax(1)==y).sum());total+=float(loss.detach())*len(y);ce_total+=float(ce.detach())*len(y);aux_total+=float(aux.detach())*len(y)
            sync(device);assert count==job['sampler_packets'] and steps==(epoch+1)*100
            row=dict(epoch=epoch+1,objective_loss=total/count,classification_CE=ce_total/count,auxiliary_loss=aux_total/count,train_accuracy=correct/count,packets=count,optimizer_steps=steps,training_seconds=time.perf_counter()-tick,q_min=qmin,q_max=qmax,q_entropy=entropy)
            history.append(row);csv_write(out/'history.csv',history);dump(out/'status.json',dict(status='running',epoch=epoch+1,optimizer_steps=steps,pid=os.getpid()));print(json.dumps(dict(run=job['run_id'],**row)),flush=True)
        assert steps==job['updates']
        torch.save(dict(model=model.state_dict(),rx_head=None if head is None else head.state_dict(),group_dro=None if dro is None else dro.state_dict(),optimizer=optimizer.state_dict(),config=config,epoch=job['epochs'],optimizer_steps=steps),out/'final.pt')
        cp_sha=sha(out/'final.pt')
        results=evaluate(model,h,eval_keys,allowed,stats,job['seed'],device,out/'predictions')
        dump(out/'metrics.json',results)
        dump(out/'status.json',dict(status='completed',final_epoch=job['epochs'],optimizer_steps=steps,seconds=time.perf_counter()-started,checkpoint_sha256=cp_sha,final_train_accuracy=history[-1]['train_accuracy'],final_eval_accuracy=results['accuracy'],final_eval_macro_f1=results['macro_f1'],fixed_epoch=True))
    except BaseException as error:
        dump(out/'status.json',dict(status='failed',error=repr(error),optimizer_steps=steps,seconds=time.perf_counter()-started,pid=os.getpid()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['search','formal'],required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();run(a.stage,a.index)
