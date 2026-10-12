#!/usr/bin/env python3
"""Actual balanced Day1 batch overfit, gradients and checkpoint round trip."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shen_evidence.dataset import EvidencePackets
from shen_evidence.models import make_model
from shen_evidence.runner import dump,seed_all,transform,sync


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); cfg=json.loads(a.config.read_text()); a.output.mkdir(parents=True,exist_ok=False)
    seed_all(cfg['seed']); device=torch.device('cuda:0'); reports=[]
    for rx in ('n210_1','rtl_6'):
        ds=EvidencePackets(ROOT,ROOT/cfg['protocol'],rx,'train',cfg['kind'])
        indices=[int(np.flatnonzero(ds.labels==c)[0]) for c in range(10)]
        batch=[ds[i] for i in indices]
        x=transform(torch.from_numpy(np.stack([b[0] for b in batch])),cfg,device)
        y=torch.tensor([b[1] for b in batch],device=device)
        model=make_model(cfg['kind']).to(device); model.train()
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        sync(device); tick=time.perf_counter(); losses=[]; accuracy=0.
        for step in range(400):
            opt.zero_grad(set_to_none=True); logits=model(x); loss=F.cross_entropy(logits,y)
            if logits.shape!=(10,10) or not torch.isfinite(loss): raise AssertionError('Shape/loss failure')
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()): raise AssertionError('Invalid/missing gradient')
            opt.step(); losses.append(float(loss.detach()))
            accuracy=float((logits.detach().argmax(1)==y).float().mean())
            if step>=20 and accuracy==1. and losses[-1]<.05: break
        if accuracy<.95 or losses[-1]>=losses[0]*.5: raise AssertionError(f'Overfit failed: {rx} accuracy={accuracy} loss={losses[-1]}')
        model.eval()
        with torch.no_grad(): expected=model(x)
        checkpoint=a.output/f'{rx}_smoke.pt'; torch.save({'model':model.state_dict()},checkpoint)
        restored=make_model(cfg['kind']).to(device).eval()
        restored.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['model'])
        with torch.no_grad(): torch.testing.assert_close(expected,restored(x),rtol=0,atol=0)
        sync(device)
        reports.append({'receiver':rx,'kind':cfg['kind'],'shape':list(x.shape),'label_min':int(y.min()),'label_max':int(y.max()),
                        'gradient_finite':True,'steps':step+1,'initial_loss':losses[0],'final_loss':losses[-1],
                        'overfit_accuracy':accuracy,'checkpoint_exact_roundtrip':True,'seconds':time.perf_counter()-tick,
                        'parameters':sum(p.numel() for p in model.parameters()),'gpu':torch.cuda.get_device_name(0)})
        dump(a.output/'checks.json',reports); ds.close()
        print(json.dumps(reports[-1]),flush=True)
    dump(a.output/'passed.json',{'passed':True,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'checks':reports})

if __name__=='__main__': main()
