#!/usr/bin/env python3
"""Validate source data, label/row alignment, all-class maps, and four physical GPUs."""
import argparse,concurrent.futures,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a.cache import write_json
from shen_p2a8.g1_data import cache,G1Dataset,fit_scaler,sha
from shen_p2a.representations import paired_maps

def cpu_check(keys,classes):
    checks=[]
    for key in keys:
        folder=cache(ROOT,key);m=json.loads((folder/'manifest.json').read_text());assert m['status']=='completed'
        y=np.load(folder/'labels.npy');rows=np.load(folder/'rows.npy');assert set(y)==set(range(classes)) and len(set(zip(y.tolist(),rows.tolist())))==len(y)
        for method in ('full_cis','strict_cis'):
            values=np.load(folder/f'{method}.npy',mmap_mode='r');assert np.isfinite(values).all()
        iq=np.load(folder/'iq.npy',mmap_mode='r');assert iq.shape==(len(y),2,8192) and np.isfinite(iq).all()
        checks.append(dict(key=key,packets=len(y),classes=classes,source_manifest_sha256=sha(folder/'manifest.json')))
    return checks

def gpu_check(dataset,gpu):
    import subprocess
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,'scripts/preflight_shen_p2a8_g1.py','--dataset',dataset,'--gpu-worker',str(gpu)],cwd=ROOT,env=env,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['shen','osu'],required=True);p.add_argument('--gpu-worker',type=int);a=p.parse_args();reg=json.loads((ROOT/'configs/shen_p2a8/g1_registration.json').read_text())
    keys=[x['key'] for x in reg['files']] if a.dataset=='shen' else ['osu_RX1','osu_RX2'];classes=10 if a.dataset=='shen' else 25
    if a.gpu_worker is not None:
        import torch
        import torch.nn.functional as F
        from shen_evidence.runner import seed_all
        from train_shen_p2a8_g1 import make,prepare
        key='shen_'+reg['train_receivers'][0]+'_train' if a.dataset=='shen' else 'osu_RX1';checks=[]
        for method in reg['methods']:
            seed_all(20261012);ds=G1Dataset(ROOT,[key],method,fit_scaler(ROOT,[key],method));indices=[int(np.flatnonzero(ds.labels_all==c)[0]) for c in range(classes)]
            batch=[ds[i] for i in indices];x=prepare(torch.from_numpy(np.stack([b[0] for b in batch])),method,'cuda:0');y=torch.tensor([b[1] for b in batch],device='cuda:0')
            model=make(method,classes).cuda();opt=torch.optim.Adam(model.parameters(),lr=.003);first=None
            for step in range(40):
                model.train();opt.zero_grad(set_to_none=True);logits=model(x);loss=F.cross_entropy(logits,y)
                if first is None:first=float(loss.detach())
                assert logits.shape==(classes,classes) and torch.isfinite(loss)
                loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters());torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            assert float(loss.detach())<first
            model.eval();expected=model(x).detach();path=ROOT/f'outputs/shen_p2a8/g1_{a.dataset}_gpu{a.gpu_worker}_{method}_smoke.pt';torch.save(model.state_dict(),path)
            restored=make(method,classes).cuda().eval();restored.load_state_dict(torch.load(path,weights_only=True));torch.testing.assert_close(expected,restored(x),rtol=0,atol=0)
            checks.append(dict(method=method,initial_loss=first,final_loss=float(loss.detach()),all_gradients_finite=True,checkpoint_exact=True))
        write_json(ROOT/f'outputs/shen_p2a8/g1_{a.dataset}_gpu{a.gpu_worker}.json',dict(passed=True,checks=checks))
    else:
        checks=cpu_check(keys,classes)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda gpu:gpu_check(a.dataset,gpu),range(4)))
        write_json(ROOT/f'outputs/shen_p2a8/g1_{a.dataset}_preflight.json',dict(passed=True,checks=checks,gpus=4,locked_accessed=False))
