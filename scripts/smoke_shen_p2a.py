#!/usr/bin/env python3
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_evidence.runner import seed_all,sync
from shen_p2a.runner import dataset,prepare,result
from shen_p2a.models import make_model
from shen_p2a.cache import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);checks=[];device=torch.device('cuda:0')
    try:
        for config_path in json.loads(a.configs.read_text()):
            cfg=json.loads(Path(config_path).read_text());seed_all(cfg['seed'])
            ds=dataset(ROOT,ROOT/cfg['protocol'],cfg,'train')
            ids=[int(np.flatnonzero(ds.labels==c)[0]) for c in range(10)];batch=[ds[i] for i in ids]
            x=prepare(torch.from_numpy(np.stack([b[0] for b in batch])),cfg,device);y=torch.tensor([b[1] for b in batch],device=device)
            model=make_model(cfg).to(device).train();opt=torch.optim.Adam(model.parameters(),lr=.003)
            sync(device);start=time.perf_counter();first=None;acc=0.
            for step in range(350):
                opt.zero_grad(set_to_none=True);out=result(model,x);loss=F.cross_entropy(out['logits'],y)
                if first is None:first=float(loss.detach())
                if out['logits'].shape!=(10,10) or not torch.isfinite(loss):raise AssertionError('Invalid shape or loss')
                loss.backward()
                if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()):raise AssertionError('Missing/nonfinite gradients')
                opt.step();acc=float((out['logits'].detach().argmax(1)==y).float().mean())
                if step>=20 and acc==1. and float(loss.detach())<.05:break
            if acc<.95 or float(loss.detach())>first*.5:raise AssertionError('Single-batch overfit failed')
            model.eval()
            with torch.no_grad():expected=result(model,x)['logits']
            file=a.output/f'checkpoint_{len(checks)}.pt';torch.save(model.state_dict(),file)
            restored=make_model(cfg).to(device).eval();restored.load_state_dict(torch.load(file,map_location=device,weights_only=True))
            with torch.no_grad():torch.testing.assert_close(expected,result(restored,x)['logits'],rtol=0,atol=0)
            sync(device)
            row={'config':str(config_path),'method':cfg['method'],'receiver':cfg['receiver'],'family':cfg['family'],
                 'parameters':sum(v.numel() for v in model.parameters()),'shape':[10,10],'labels':[int(y.min()),int(y.max())],
                 'steps':step+1,'initial_loss':first,'final_loss':float(loss.detach()),'overfit_accuracy':acc,
                 'gradient_finite':True,'checkpoint_exact_roundtrip':True,'seconds':time.perf_counter()-start}
            checks.append(row);write_json(a.output/'checks.json',checks);print(json.dumps(row),flush=True)
            if hasattr(ds,'close'):ds.close()
        write_json(a.output/'passed.json',{'passed':True,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'checks':checks})
    except BaseException as exc:
        write_json(a.output/'failed.json',{'error':repr(exc),'completed_checks':checks});raise
