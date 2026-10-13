#!/usr/bin/env python3
"""One of exactly eight approved CE runs; original model/recipe, final20 only."""
import argparse,ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a8.g1_data import G1Dataset,fit_scaler,sha
from shen_evidence.runner import seed_all,loader_for,sync
from shen_p2a.models import FactorizedAuthorCNN
from audit_d0_class_pair import write_json,write_csv

@torch.no_grad()
def prediction(model,dataset,cfg,device):
    model.eval();logits=[];labels=[]
    for x,y,_ in loader_for(dataset,cfg):
        z=model(x.to(device,non_blocking=True));assert torch.isfinite(z).all()
        logits.append(z.cpu().numpy());labels.extend(y.numpy().tolist())
    return np.concatenate(logits).astype('float64'),np.array(labels,dtype='int64')

def run(cfg,out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='running',pid=os.getpid(),started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    try:
        assert torch.__version__=='2.5.1+cu121' and np.__version__=='1.26.4'
        regpath=ROOT/'configs/d05_b0/registration.json';r=json.loads(regpath.read_text())
        seal=json.loads((ROOT/'configs/d05_b0/pre_execution_seal.json').read_text())
        for path,digest in seal['files_sha256'].items():assert sha(ROOT/path)==digest,path
        assert cfg in r['jobs'] and cfg['epochs']==20 and cfg['method'] in r['b0_methods']
        assert cfg['seed'] in r['new_seeds'] and set(cfg['train_keys']).isdisjoint(cfg['validation_keys'])
        h=Path(r['history_root'])
        for key in cfg['train_keys']+cfg['validation_keys']:
            rx=key.removeprefix('shen_').removesuffix('_train');assert rx in r['source_receivers']
            folder=h/'outputs/shen_p2a8/g1_cache_v1'/key
            manifest=json.loads((folder/'manifest.json').read_text());assert manifest['source']['role']=='train'
            for name in (cfg['method']+'.npy','rows.npy','labels.npy','manifest.json'):
                path=folder/name;expected=r['feature_assets'][str(path.relative_to(h))];assert sha(path)==expected['sha256']
        seed_all(cfg['seed']);device=torch.device('cuda:0')
        stats=fit_scaler(h,cfg['train_keys'],cfg['method'])
        historical=json.loads((h/f"outputs/shen_p2a8/g1_shen_v2/runs/CV{cfg['fold']}_{cfg['method']}/config.json").read_text())
        assert stats==historical['scaler'] and cfg['train_keys']==historical['train_keys'] and cfg['validation_keys']==historical['validation_keys']
        train=G1Dataset(h,cfg['train_keys'],cfg['method'],stats)
        validation=G1Dataset(h,cfg['validation_keys'],cfg['method'],stats)
        assert len(train)==len(validation)==10000
        model=FactorizedAuthorCNN(16).to(device)
        initial_sha=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        config=dict(cfg,scaler=stats,registration_sha256=sha(regpath),initial_model_payload_sha256=initial_sha,parameters=sum(p.numel() for p in model.parameters()),gpu_name=torch.cuda.get_device_name(0),torch_version=torch.__version__,numpy_version=np.__version__,python=sys.version,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),historical_model_source_sha256={path:sha(ROOT/path) for path in historical['source_sha256']},selection_rule='fixed final epoch20; no validation-based branch')
        assert config['parameters']==658314
        source_checks=[]
        for path,digest in historical['source_sha256'].items():
            actual=config['historical_model_source_sha256'][path]
            if actual==digest:
                source_checks.append(dict(path=path,status='exact_hash_match'))
            elif path=='src/shen_p2a8/g1_data.py':
                original=subprocess.check_output(['git','show','5661a32:'+path],cwd=ROOT,text=True)
                assert hashlib.sha256(original.encode()).hexdigest()==digest
                nodes=lambda text:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
                old,new=nodes(original),nodes((ROOT/path).read_text())
                assert all(new[k]==v for k,v in old.items()) and set(new)-set(old)=={'validate_run_roles'}
                source_checks.append(dict(path=path,status='exact_historical_hash_and_unchanged_AST',historical_revision='5661a32',added_function_not_used='validate_run_roles'))
            elif path=='src/shen_p2a/runner.py':
                assert 'shen_p2a.runner' not in sys.modules
                source_checks.append(dict(path=path,status='unused_directory_snapshot_file',not_imported=True))
            else:raise AssertionError('Changed execution dependency: '+path)
        config['historical_source_equivalence_checks']=source_checks
        write_json(out/'config.json',config)
        optimizer=torch.optim.Adam(model.parameters(),lr=r['b0']['lr'],weight_decay=r['b0']['weight_decay'])
        loader=loader_for(train,cfg,True);history=[]
        for epoch in range(1,21):
            model.train();count=correct=0;total=0.;sync(device);tick=time.perf_counter()
            for x,y,_ in loader:
                x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True)
                z=model(x);loss=F.cross_entropy(z,y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                count+=len(y);correct+=int((z.detach().argmax(1)==y).sum());total+=float(loss.detach())*len(y)
            sync(device);row=dict(epoch=epoch,train_loss=total/count,train_accuracy=correct/count,training_seconds=time.perf_counter()-tick)
            valz,valy=prediction(model,validation,cfg,device)
            row['validation_accuracy']=float((valz.argmax(1)==valy).mean())
            row['validation_used_for_selection']=False;history.append(row)
            write_csv(out/'history.csv',history);write_json(out/'status.json',dict(status='running',epoch=epoch,pid=os.getpid()))
            print(json.dumps(dict(run=cfg['run_id'],**row)),flush=True)
        torch.save(dict(model=model.state_dict(),config=config,epoch=20,optimizer=optimizer.state_dict()),out/'final.pt')
        checkpoint_sha=sha(out/'final.pt');dst=out/'predictions';dst.mkdir();metrics=[]
        # Last epoch's validation output is reused, not another fitting/inference run.
        offset=0
        for key in cfg['validation_keys']:
            rx=key.removeprefix('shen_').removesuffix('_train');folder=h/'outputs/shen_p2a8/g1_cache_v1'/key
            rows=np.load(folder/'rows.npy',allow_pickle=False);y=np.load(folder/'labels.npy',allow_pickle=False)
            z=valz[offset:offset+2000];assert np.array_equal(y,valy[offset:offset+2000]);offset+=2000
            exp=np.exp(z-z.max(1,keepdims=True));prob=exp/exp.sum(1,keepdims=True)
            np.savez(dst/(rx+'.npz'),logits=z,probabilities=prob,labels=y,rows=rows)
            c=np.zeros((10,10),int);np.add.at(c,(y,z.argmax(1)),1)
            for tx in range(10):metrics.append(dict(receiver=rx,tx=tx+31,packets=200,correct=int(c[tx,tx]),recall=float(c[tx,tx]/200)))
        assert offset==10000
        write_csv(out/'per_RX_TX.csv',metrics)
        write_json(dst/'manifest.json',dict(run=cfg['run_id'],seed=cfg['seed'],fold=cfg['fold'],method=cfg['method'],epoch=20,checkpoint_sha256=checkpoint_sha,registration_sha256=sha(regpath),validation_keys=cfg['validation_keys'],scaler=stats,prediction_dtype='float64 converted from GPU float32 logits',last_validation_reused=True,assets={p.name:sha(p) for p in dst.glob('*.npz')}))
        write_json(out/'status.json',dict(status='completed',final_epoch=20,seconds=time.perf_counter()-started,checkpoint_sha256=checkpoint_sha,final_train_accuracy=history[-1]['train_accuracy'],final_train_loss=history[-1]['train_loss'],final_validation_accuracy=history[-1]['validation_accuracy'],selected_epoch=20,validation_selected=False))
    except BaseException as error:
        write_json(out/'status.json',dict(status='failed',error=repr(error),pid=os.getpid(),seconds=time.perf_counter()-started));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-index',type=int,required=True);a=p.parse_args()
    reg=json.loads((ROOT/'configs/d05_b0/registration.json').read_text());assert 0<=a.job_index<8
    cfg=reg['jobs'][a.job_index];run(cfg,ROOT/'outputs/d05_b0/b0/runs'/cfg['run_id'])
