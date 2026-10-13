"""Actual Adam transactions and read-only paired finite-update measurements."""
import copy
import hashlib
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from torch.func import functional_call


def clone_parameters(model):
    return {n:p.detach().clone() for n,p in model.named_parameters()}


def squared_norm(values):
    terms=[v.detach().double().square().sum() for v in values]
    return float(torch.stack(terms).sum()) if terms else 0.


def state_hash(value):
    h=hashlib.sha256()
    def walk(x):
        if isinstance(x,torch.Tensor):
            a=x.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());walk(x[k])
        elif isinstance(x,(list,tuple)):
            for a in x:walk(a)
        else:h.update(repr(x).encode())
    walk(value);return h.hexdigest()


@contextmanager
def isolated_mode_rng(model,training):
    modes=[(m,m.training) for m in model.modules()]
    devices=[torch.cuda.current_device()] if next(model.parameters()).is_cuda else []
    with torch.random.fork_rng(devices=devices):
        try:
            model.train(training);yield
        finally:
            for m,flag in modes:m.training=flag


def commit_scaled(model,optimizer,before_parameters,before_optimizer,scale):
    """Called AFTER real Adam proposal: keep proposal moments iff scale>0.

    Rejection restores all Adam state including step counters. Nonzero scaling
    changes parameters only; moment estimates describe the unscaled gradient.
    R2-A uses scale1 exclusively. This primitive is also tested for future R2-B.
    """
    assert scale in (1.,.5,.25,0.)
    with torch.no_grad():
        for n,p in model.named_parameters():
            if scale==0:p.copy_(before_parameters[n])
            elif scale!=1:p.copy_(before_parameters[n]+scale*(p-before_parameters[n]))
    if scale==0:optimizer.load_state_dict(copy.deepcopy(before_optimizer))


@torch.no_grad()
def risk_measurement(model,before_parameters,pools,epoch,step,seed,scales=(1.,.5,.25,0.),train_replicates=3):
    """C1/C2 sample risks, eval and common-random-number train modes.

    Returns public aggregate/per-cell rows and private paired packet deltas.
    No buffers in the approved backbones; assert to prevent silent BN effects.
    Measurement randomness and model mode are restored even on failure.
    """
    assert not list(model.named_buffers()),'Approved backbones have no mutable buffers'
    proposal=clone_parameters(model)
    deltas={n:proposal[n]-p for n,p in before_parameters.items()}
    rows=[];private=[]
    for mode,reps in [('eval',1),('train',train_replicates)]:
        with isolated_mode_rng(model,mode=='train'):
            for pool_name,data in pools.items():
                for rep in range(reps):
                    losses={scale:[] for scale in scales};pre=[];accepted=[]
                    for start in range(0,len(data['y']),256):
                        x=data['x'][start:start+256];y=data['y'][start:start+256]
                        mask_seed=seed+epoch*100000+rep*1000+start
                        def reseed():
                            torch.manual_seed(mask_seed)
                            if x.is_cuda:torch.cuda.manual_seed_all(mask_seed)
                        reseed();pre.append(F.cross_entropy(functional_call(model,before_parameters,(x,)),y,reduction='none'))
                        for scale in scales:
                            pars=proposal if scale==1 else before_parameters if scale==0 else {n:p+scale*deltas[n] for n,p in before_parameters.items()}
                            reseed();losses[scale].append(F.cross_entropy(functional_call(model,pars,(x,)),y,reduction='none'))
                        reseed();accepted.append(F.cross_entropy(model(x),y,reduction='none'))
                    a=torch.cat(pre).cpu().numpy();acc=torch.cat(accepted).cpu().numpy()
                    rr=data['rx'].cpu().numpy();dd=data['day'].cpu().numpy();yy=data['y'].cpu().numpy()
                    for scale in scales:
                        b=torch.cat(losses[scale]).cpu().numpy();delta=b-a
                        # Equal candidate parameters must reproduce accepted losses.
                        if scale==1:assert np.array_equal(b,acc)
                        if scale==0:assert np.array_equal(a,b)
                        private.append(dict(pool=pool_name,mode=mode,replicate=rep,scale=scale,rx=rr,date=dd,tx=yy,delta=delta,pre=a,post=b))
                        cells=[(r,-999,-999,rr==r) for r in np.unique(rr)]
                        cells += [(r,d,t,(rr==r)&(dd==d)&(yy==t)) for r in np.unique(rr) for d in np.unique(dd[rr==r]) for t in range(10)]
                        for r,d,t,mask in cells:
                            n=int(mask.sum());v=delta[mask]
                            se=float(v.std(ddof=1)/np.sqrt(n)) if n>1 else None
                            mu=float(v.mean()) if n else None
                            resolution=1e-7+1e-6*float(np.abs(a[mask]).mean()) if n else None
                            rows.append(dict(epoch=epoch,step=step,pool=pool_name,mode=mode,replicate=rep,scale=scale,rx=int(r),date=int(d),tx=int(t),n=n,pre_CE=float(a[mask].mean()) if n else None,post_CE=float(b[mask].mean()) if n else None,delta=mu,paired_delta_SE=se,resolution=resolution,significant_harm=bool(mu>2*se+resolution) if n>1 else None,near_zero=bool(abs(mu)<=2*se+resolution) if n>1 else None,sufficient=n>=2,actual_accepted_CE=float(acc[mask].mean()) if n else None))
    return rows,private
