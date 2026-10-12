"""Reproducible packet-level development runner with immutable run directories."""
import csv
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rfhstu.losses import macro_f1_from_logits
from .dataset import EvidencePackets
from .inputs import build_input
from .models import make_model


def dump(path,value):
    path=Path(path)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    random.seed(seed); np.random.seed(seed)


def loader_for(dataset,cfg,train=False):
    return DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=train,
                      num_workers=cfg['workers'],pin_memory=True,
                      persistent_workers=cfg['workers']>0,worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(cfg['seed']),drop_last=False)


def transform(x,cfg,device):
    return build_input(x.to(device,dtype=torch.float32,non_blocking=True),cfg['kind'],
                       cfg['sample_rate_hz'],cfg['bandwidth_hz'],cfg['guard_hz'])


def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model,loader,cfg,device):
    model.eval(); all_logits=[]; all_labels=[]; all_rows=[]; loss=0.; n=0
    inference_seconds=0.; sync(device); start=time.perf_counter()
    for x,y,rows in loader:
        x=transform(x,cfg,device); y=y.to(device)
        sync(device); tick=time.perf_counter()
        logits=model(x)
        sync(device); inference_seconds+=time.perf_counter()-tick
        loss+=float(F.cross_entropy(logits,y,reduction='sum')); n+=len(y)
        all_logits.append(logits.cpu()); all_labels.append(y.cpu()); all_rows.append(rows)
    sync(device); elapsed=time.perf_counter()-start
    logits=torch.cat(all_logits); labels=torch.cat(all_labels); rows=torch.cat(all_rows)
    pred=logits.argmax(1)
    cm=torch.bincount(labels*10+pred,minlength=100).reshape(10,10).numpy()
    metrics={'packet_accuracy':float((pred==labels).float().mean()),
             'macro_f1':macro_f1_from_logits(logits,labels,10),'loss':loss/n,'packets':n,
             'validation_seconds':elapsed,'pipeline_packets_per_second':n/elapsed,
             'model_packets_per_second':n/max(inference_seconds,1e-12),
             'model_batch_amortized_ms_per_packet':inference_seconds*1000/n}
    return metrics,logits,labels,rows,cm


def train(cfg,root,output):
    root=Path(root).resolve(); output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)  # Never overwrite old/failed results.
    dump(output/'status.json',{'status':'running','pid':os.getpid()})
    started=time.perf_counter()
    try:
        seed_all(cfg['seed']); device=torch.device(cfg.get('device','cuda:0'))
        protocol=(root/cfg['protocol']).resolve()
        if hashlib.sha256(protocol.read_bytes()).hexdigest()!=cfg['protocol_sha256_expected']:
            raise ValueError('Protocol differs from the frozen development config')
        cfg=dict(cfg,protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
                 additional_cfo_compensation=False,selection='Day2 entire-dataset macro_f1',
                 metric_version='macro_fixed_10_classes_v2',physical_frequency_metadata='literature_assumed')
        dump(output/'config.json',cfg)
        ds_train=EvidencePackets(root,protocol,cfg['receiver'],'train',cfg['kind'])
        ds_val=EvidencePackets(root,protocol,cfg['receiver'],'validation',cfg['kind'])
        for ds in (ds_train,ds_val):
            if set(ds.labels)!={*range(10)}: raise ValueError('Expected all ten closed-set devices')
        train_loader=loader_for(ds_train,cfg,True); val_loader=loader_for(ds_val,cfg)
        model=make_model(cfg['kind']).to(device)
        params=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        best=-1.; patience=0; history=[]; train_seconds=0.
        for epoch in range(1,cfg['max_epochs']+1):
            model.train(); sync(device); tick=time.perf_counter(); loss_sum=0.; correct=0; n=0
            for x,y,_ in train_loader:
                x=transform(x,cfg,device); y=y.to(device)
                optimizer.zero_grad(set_to_none=True); logits=model(x)
                loss=F.cross_entropy(logits,y)
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True)
                optimizer.step(); loss_sum+=float(loss.detach())*len(y)
                correct+=int((logits.detach().argmax(1)==y).sum()); n+=len(y)
            sync(device); epoch_seconds=time.perf_counter()-tick; train_seconds+=epoch_seconds
            val,*_=evaluate(model,val_loader,cfg,device)
            row={'epoch':epoch,'train_loss':loss_sum/n,'train_accuracy':correct/n,
                 'train_seconds':epoch_seconds,'train_packets_per_second':n/epoch_seconds,
                 **{'val_'+k:v for k,v in val.items()}}
            history.append(row)
            with (output/'history.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
            print(json.dumps({'run':output.name,**row}),flush=True)
            if val['macro_f1']>best+cfg['min_delta']:
                best=val['macro_f1']; patience=0
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                            'config':cfg,'epoch':epoch,'validation':val},output/'best.pt')
            else: patience+=1
            dump(output/'status.json',{'status':'running','epoch':epoch,'best_macro_f1':best,'pid':os.getpid()})
            if patience>=cfg['patience']: break
        checkpoint=torch.load(output/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model'])
        metrics,logits,labels,rows,cm=evaluate(model,val_loader,cfg,device)
        metrics.update({'receiver':cfg['receiver'],'kind':cfg['kind'],'seed':cfg['seed'],'lr':cfg['lr'],
                        'parameters':params,'best_epoch':checkpoint['epoch'],'epochs_completed':len(history),
                        'training_seconds':train_seconds,'total_seconds':time.perf_counter()-started,
                        'mean_train_packets_per_second':len(ds_train)*len(history)/train_seconds})
        dump(output/'metrics.json',metrics)
        np.savetxt(output/'confusion_matrix.csv',cm,fmt='%d',delimiter=',',header=','.join(map(str,range(10))),comments='')
        probs=logits.softmax(1).numpy(); pred=probs.argmax(1); records=[]
        for i,(label,row) in enumerate(zip(labels.tolist(),rows.tolist())):
            records.append({'sample_id':f'Test/{cfg["receiver"]}_day2_wireless_test.h5:{row}',
                            'receiver':cfg['receiver'],'day':2,'row':row,'label':label,'raw_label':label+31,
                            'prediction':int(pred[i]),'confidence':float(probs[i,pred[i]]),
                            **{f'p_{c}':float(probs[i,c]) for c in range(10)}})
        for filename,filtered in [('predictions.csv',records),('errors.csv',[r for r in records if r['label']!=r['prediction']])]:
            with (output/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(filtered)
        dump(output/'status.json',{'status':'completed','metrics':metrics})
        return metrics
    except BaseException as exc:
        dump(output/'status.json',{'status':'failed','error':repr(exc),'pid':os.getpid(),
                                 'elapsed_seconds':time.perf_counter()-started})
        raise
