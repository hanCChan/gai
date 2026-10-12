"""P2-A training: immutable runs, train-only scaling and entire-set metrics."""
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rfhstu.losses import macro_f1_from_logits
from shen_stage1.data import ShenDevelopmentPackets
from shen_evidence.runner import seed_all,loader_for,sync
from .cache import MapPackets,write_json,cache_dir
from .bands import transform
from .models import make_model


def dataset(root,protocol,cfg,role):
    if cfg['family']=='representation':
        return MapPackets(root,protocol,cfg['receiver'],role,cfg['representation'],cfg['map_normalization'])
    return ShenDevelopmentPackets(root,protocol,cfg['receiver'],role,cache_handles=True)


def prepare(x,cfg,device):
    x=x.to(device,dtype=torch.float32,non_blocking=True)
    return x if cfg['family']=='representation' else transform(x,cfg)


def result(model,x):
    out=model(x)
    return {'logits':out} if isinstance(out,torch.Tensor) else out


@torch.no_grad()
def evaluate(model,loader,cfg,device):
    model.eval(); logits_all=[];labels_all=[];rows_all=[];gates=[];loss=0.;n=0;forward_seconds=0.
    sync(device);start=time.perf_counter()
    for x,y,rows in loader:
        x=prepare(x,cfg,device);y=y.to(device,non_blocking=True)
        sync(device);t=time.perf_counter();out=result(model,x);sync(device);forward_seconds+=time.perf_counter()-t
        logits=out['logits'];loss+=float(F.cross_entropy(logits,y,reduction='sum'));n+=len(y)
        logits_all.append(logits.cpu());labels_all.append(y.cpu());rows_all.append(rows)
        if 'gate' in out:gates.append(out['gate'].cpu())
    sync(device);elapsed=time.perf_counter()-start
    logits=torch.cat(logits_all);labels=torch.cat(labels_all);rows=torch.cat(rows_all);pred=logits.argmax(1)
    cm=torch.bincount(labels*10+pred,minlength=100).reshape(10,10).numpy()
    metrics={'packet_accuracy':float((pred==labels).float().mean()),'macro_f1':macro_f1_from_logits(logits,labels,10),
             'loss':loss/n,'packets':n,'pipeline_seconds':elapsed,'pipeline_packets_per_second':n/elapsed,
             'model_packets_per_second':n/max(forward_seconds,1e-12),'model_ms_per_packet':forward_seconds*1000/n,
             'device34_recall':float((pred[labels==3]==3).float().mean())}
    gate=torch.cat(gates).numpy() if gates else None
    if gate is not None:
        metrics.update(gate_mean=float(gate.mean()),gate_p05=float(np.quantile(gate,.05)),gate_p95=float(np.quantile(gate,.95)),
                       gate_fraction_lt_005=float((gate<.05).mean()),gate_fraction_gt_095=float((gate>.95).mean()))
    return metrics,logits,labels,rows,cm,gate


def save_predictions(output,cfg,ev,prefix=''):
    metrics,logits,labels,rows,cm,gate=ev
    probabilities=logits.softmax(1).numpy();pred=probabilities.argmax(1);records=[]
    for i,(label,row) in enumerate(zip(labels.tolist(),rows.tolist())):
        records.append({'sample_id':f'Test/{cfg["receiver"]}_day2_wireless_test.h5:{row}','receiver':cfg['receiver'],'row':row,
                        'label':label,'raw_label':label+31,'prediction':int(pred[i]),'confidence':float(probabilities[i,pred[i]]),
                        'gate':float(gate[i]) if gate is not None else '',**{f'p_{c}':float(probabilities[i,c]) for c in range(10)}})
    for filename,data in [('predictions.csv',records),('errors.csv',[r for r in records if r['label']!=r['prediction']])]:
        with (output/(prefix+filename)).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    np.savetxt(output/(prefix+'confusion_matrix.csv'),cm,fmt='%d',delimiter=',',header=','.join(map(str,range(10))),comments='')
    counts=np.bincount(labels.numpy(),minlength=10)
    per_class=[{'label':c,'raw_label':c+31,'packets':int(counts[c]),'correct':int(cm[c,c]),'recall':float(cm[c,c]/counts[c])} for c in range(10)]
    with (output/(prefix+'per_device.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_class[0]),lineterminator='\n');w.writeheader();w.writerows(per_class)


def train(cfg,root,output):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();write_json(output/'status.json',{'status':'running','pid':os.getpid()})
    try:
        if torch.__version__!='2.5.1+cu121' or np.__version__!='1.26.4':
            raise RuntimeError('Formal P2-A runs require the frozen P1 shen-stage1 environment')
        if cfg.get('stage')!='P2-A':raise PermissionError('P2-B/P3 are not executable by this runner')
        seed_all(cfg['seed']);device=torch.device('cuda:0');protocol=root/cfg['protocol']
        if hashlib.sha256(protocol.read_bytes()).hexdigest()!=cfg['protocol_sha256_expected']:raise ValueError('Protocol SHA changed')
        cfg=dict(cfg,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                 frozen_p1_baseline='eb4ce2c0db783f9241777f8b94328500a1014113',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 torch_version=torch.__version__,cuda_runtime=torch.version.cuda,additional_cfo_compensation=False,
                 selection='Day2 entire-set fixed-10-class Macro-F1',frequency_metadata='literature assumptions, not verified locally')
        train_ds=dataset(root,protocol,cfg,'train');val_ds=dataset(root,protocol,cfg,'validation')
        for ds in (train_ds,val_ds):
            if set(ds.labels)!=set(range(10)):raise ValueError('Ten-class label set changed')
        if cfg['family']=='representation':
            info=json.loads((cache_dir(root,cfg['receiver'],'train')/'manifest.json').read_text())
            cfg['train_feature_scaler']=info['mean_std'][cfg['representation']] if cfg['map_normalization']=='train_scalar' else {'mean':0.,'std':1.}
            cfg['stft']=info['stft']
        write_json(output/'config.json',cfg)
        train_loader=loader_for(train_ds,cfg,True);val_loader=loader_for(val_ds,cfg)
        torch.cuda.set_device(device)  # Initialize the CUDA context before allocator statistics.
        torch.cuda.reset_peak_memory_stats(device)
        free,total=torch.cuda.mem_get_info(device)
        if free<4*1024**3:raise RuntimeError('Less than 4GiB free; preserve GPU memory reserve')
        model=make_model(cfg).to(device);params=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        history=[];best=-1.;patience=0;shadow_stop=None;shadow_best=-1.;train_seconds=0.
        selected_params=[(name,p) for name,p in model.named_parameters()]
        first_param=next(p for name,p in selected_params if p.ndim>=3)
        last_param=next(p for name,p in reversed(selected_params) if p.ndim==2)
        hook_value=[]
        modules=[m for name,m in model.named_modules() if name.endswith('head.2') or name.endswith('embedding.2')]
        handles=[m.register_forward_hook(lambda _,inputs,out:hook_value.append(float((out.detach()==0).float().mean()))) for m in modules]
        for epoch in range(1,cfg['max_epochs']+1):
            model.train();sync(device);tick=time.perf_counter();n=0;correct=0;loss_sum=0.;grad_norms=[];first_grads=[];last_grads=[];dead=[]
            for x,y,_ in train_loader:
                hook_value.clear();x=prepare(x,cfg,device);y=y.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True);out=result(model,x);logits=out['logits'];loss=F.cross_entropy(logits,y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                grad_norms.append(float(norm));first_grads.append(float(first_param.grad.norm()));last_grads.append(float(last_param.grad.norm()))
                dead.extend(hook_value);optimizer.step();loss_sum+=float(loss.detach())*len(y);n+=len(y);correct+=int((logits.detach().argmax(1)==y).sum())
            sync(device);seconds=time.perf_counter()-tick;train_seconds+=seconds
            ev=evaluate(model,val_loader,cfg,device);v=ev[0]
            if v['macro_f1']>best+cfg['min_delta']:
                best=v['macro_f1'];patience=0
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'epoch':epoch,'validation':v},output/'best.pt')
            else:patience+=1
            if shadow_stop is None:
                shadow_best=best
                if epoch>=cfg.get('min_epochs',0) and patience>=cfg['patience']:
                    shadow_stop=epoch;torch.save(torch.load(output/'best.pt',map_location='cpu',weights_only=False),output/'shadow_early_stop.pt')
            row={'epoch':epoch,'train_loss':loss_sum/n,'train_accuracy':correct/n,'train_seconds':seconds,'train_packets_per_second':n/seconds,
                 'preclip_grad_norm_mean':float(np.mean(grad_norms)),'preclip_grad_norm_max':float(np.max(grad_norms)),
                 'gradient_clipped_fraction':float((np.asarray(grad_norms)>5).mean()),'first_conv_grad_mean':float(np.mean(first_grads)),
                 'classifier_grad_mean':float(np.mean(last_grads)),'embedding_dead_fraction':float(np.mean(dead)) if dead else '',
                 **{'val_'+k:x for k,x in v.items()}}
            history.append(row)
            with (output/'history.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');w.writeheader();w.writerows(history)
            print(json.dumps({'run':output.name,**row}),flush=True)
            write_json(output/'status.json',{'status':'running','epoch':epoch,'best_macro_f1':best,'pid':os.getpid()})
            if cfg.get('early_stopping_enabled',True) and epoch>=cfg.get('min_epochs',0) and patience>=cfg['patience']:break
        for h in handles:h.remove()
        checkpoint=torch.load(output/'best.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['model'])
        ev=evaluate(model,val_loader,cfg,device);metrics=ev[0]
        metrics.update(receiver=cfg['receiver'],family=cfg['family'],comparison_group=cfg['comparison_group'],method=cfg['method'],
                       seed=cfg['seed'],lr=cfg['lr'],weight_decay=cfg['weight_decay'],parameters=params,best_epoch=checkpoint['epoch'],
                       epochs_completed=len(history),training_seconds=train_seconds,total_seconds=time.perf_counter()-start,
                       shadow_early_stop_epoch=shadow_stop,shadow_early_stop_macro_f1=shadow_best,
                       source_commit=cfg['source_commit'])
        metrics.update(gpu_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                       gpu_peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2,gpu_total_mib=total/1024**2)
        save_predictions(output,cfg,ev)
        if not cfg.get('early_stopping_enabled',True):
            shadow_path=output/'shadow_early_stop.pt'
            shadow=torch.load(shadow_path,map_location=device,weights_only=False) if shadow_path.exists() else checkpoint
            model.load_state_dict(shadow['model']);shadow_ev=evaluate(model,val_loader,cfg,device)
            save_predictions(output,cfg,shadow_ev,prefix='shadow_')
            metrics.update(shadow_selected_epoch=shadow['epoch'],shadow_packet_accuracy=shadow_ev[0]['packet_accuracy'],
                           shadow_macro_f1=shadow_ev[0]['macro_f1'],shadow_device34_recall=shadow_ev[0]['device34_recall'])
        write_json(output/'metrics.json',metrics)
        write_json(output/'status.json',{'status':'completed','metrics':metrics})
        return metrics
    except BaseException as exc:
        write_json(output/'status.json',{'status':'failed','error':repr(exc),'elapsed_seconds':time.perf_counter()-start,'pid':os.getpid()});raise
