#!/usr/bin/env python3
"""Source receiver CV and fixed-epoch confirmation, never select on developer RX."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a8.g1_data import G1Dataset,fit_scaler,sha,validate_run_roles
from shen_p2a.cache import write_json
from shen_evidence.runner import seed_all,loader_for,sync
from shen_evidence.inputs import build_input
from shen_evidence.models import PacketCNN
from shen_p2a.models import FactorizedAuthorCNN
from report_shen_p2a import write_csv

def make(method,classes):
    if method in ('full_iq','strict_iq'):return PacketCNN(num_classes=classes)
    model=FactorizedAuthorCNN(16)
    if classes!=10:
        model.classifier=torch.nn.Linear(128,classes);torch.nn.init.xavier_uniform_(model.classifier.weight);torch.nn.init.zeros_(model.classifier.bias)
    return model

def prepare(x,method,device):
    x=x.to(device,non_blocking=True)
    if method in ('full_iq','strict_iq'):return build_input(x,'full_iq' if method=='full_iq' else 'strict_ib')
    return x

@torch.no_grad()
def evaluate(model,ds,cfg,device,save=None):
    model.eval();logits=[];ys=[];ids=[];seconds=0.
    for x,y,keys in loader_for(ds,cfg):
        x=prepare(x,cfg['method'],device);sync(device);start=time.perf_counter();p=model(x);sync(device);seconds+=time.perf_counter()-start
        logits.append(p.cpu());ys.extend(y.tolist());ids.extend(keys)
    logits=torch.cat(logits);y=np.array(ys);pred=logits.argmax(1).numpy();c=cfg['classes']
    cm=confusion_matrix(y,pred,labels=list(range(c)));metrics=dict(accuracy=float((y==pred).mean()),macro_f1=float(f1_score(y,pred,labels=list(range(c)),average='macro',zero_division=0)),packets=len(y),forward_seconds=seconds,forward_packets_per_second=len(y)/seconds)
    if save:
        save.mkdir(parents=True,exist_ok=False);prob=logits.softmax(1).numpy();records=[]
        for i,key in enumerate(ids):
            other=logits[i].clone();other[y[i]]=-torch.inf
            records.append(dict(sample_id=key,label=int(y[i]),prediction=int(pred[i]),true_class_logit_margin=float(logits[i,y[i]]-other.max()),**{f'p_{j}':float(prob[i,j]) for j in range(c)}))
        assert len(set(ids))==len(ids)
        write_csv(save/'predictions.csv',records);np.savetxt(save/'confusion_matrix.csv',cm,fmt='%d',delimiter=',')
        per=[]
        for k in range(c):
            n=int(cm[k].sum());npred=int(cm[:,k].sum());tp=int(cm[k,k]);per.append(dict(label=k,packets=n,correct=tp,recall=tp/n if n else 0.,precision=tp/max(npred,1),f1=2*tp/max(n+npred,1)))
        write_csv(save/'per_TX.csv',per);write_json(save/'metrics.json',metrics)
    return metrics

def run(cfg,out):
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();write_json(out/'status.json',dict(status='running',pid=os.getpid()))
    try:
        assert torch.__version__=='2.5.1+cu121' and np.__version__=='1.26.4'
        regfile=ROOT/'configs/shen_p2a8/g1_registration.json';assert sha(regfile)==cfg['registration_sha256']
        reg=json.loads(regfile.read_text());assert not reg['locked_access_enabled']
        validate_run_roles(cfg,reg)
        assert cfg['phase'] in ('cv','confirmation','osu')
        if cfg['phase']=='cv':
            assert not set(cfg['train_keys'])&set(cfg['validation_keys'])
            assert all(k in ['shen_'+r+'_train' for r in reg['train_receivers']] for k in cfg['train_keys']+cfg['validation_keys'])
        seed_all(cfg['seed']);device=torch.device('cuda:0');stats=fit_scaler(ROOT,cfg['train_keys'],cfg['method'])
        cfg=dict(cfg,scaler=stats,source_sha256={str(p.relative_to(ROOT)):sha(p) for directory in ('src/shen_p2a8','src/shen_evidence','src/shen_p2a') for p in (ROOT/directory).glob('*.py')},torch_version=torch.__version__,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
        ds=G1Dataset(ROOT,cfg['train_keys'],cfg['method'],stats);model=make(cfg['method'],cfg['classes']).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=reg['lr'],weight_decay=reg['weight_decay'])
        initial_sha=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        cfg.update(parameters=sum(p.numel() for p in model.parameters()),initial_model_payload_sha256=initial_sha)
        write_json(out/'config.json',cfg);loader=loader_for(ds,cfg,True);history=[];best=-1.;bestepoch=None
        validation=G1Dataset(ROOT,cfg['validation_keys'],cfg['method'],stats) if cfg['phase']=='cv' else None
        for epoch in range(1,cfg['epochs']+1):
            model.train();n=correct=0;loss_total=0.;tick=time.perf_counter()
            for x,y,_ in loader:
                x=prepare(x,cfg['method'],device);y=y.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True);logit=model(x);loss=F.cross_entropy(logit,y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                n+=len(y);correct+=int((logit.detach().argmax(1)==y).sum());loss_total+=float(loss.detach())*len(y)
            row=dict(epoch=epoch,train_loss=loss_total/n,train_accuracy=correct/n,training_seconds=time.perf_counter()-tick)
            if validation is not None:
                ev=evaluate(model,validation,cfg,device);row.update(validation_accuracy=ev['accuracy'],validation_macro_f1=ev['macro_f1'])
                if ev['macro_f1']>best+1e-4:
                    best=ev['macro_f1'];bestepoch=epoch;torch.save(dict(model=model.state_dict(),config=cfg,epoch=epoch),out/'best_source.pt')
            history.append(row);write_csv(out/'history.csv',history);write_json(out/'status.json',dict(status='running',epoch=epoch,pid=os.getpid()));print(json.dumps(dict(run=out.name,**row)),flush=True)
        torch.save(dict(model=model.state_dict(),config=cfg,epoch=cfg['epochs']),out/'final.pt')
        if validation is not None:
            cp=torch.load(out/'best_source.pt',map_location=device,weights_only=False);model.load_state_dict(cp['model']);evaluate(model,validation,cfg,device,out/'source_validation')
        metrics=[]
        for key in cfg.get('evaluation_keys',[]):
            eds=G1Dataset(ROOT,[key],cfg['method'],stats);m=evaluate(model,eds,cfg,device,out/'evaluation'/key);metrics.append(dict(key=key,method=cfg['method'],seed=cfg['seed'],**m))
        write_csv(out/'evaluation_metrics.csv',metrics)
        write_json(out/'status.json',dict(status='completed',best_source_epoch=bestepoch,final_epoch=cfg['epochs'],seconds=time.perf_counter()-started,parameters=cfg['parameters'],evaluation_domains=len(metrics)))
    except BaseException as e:write_json(out/'status.json',dict(status='failed',error=repr(e),pid=os.getpid()));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(json.loads(args.config.read_text()),args.output)
