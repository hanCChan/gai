"""Equivalent finite-scale decision execution; original rule/constants unchanged."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from torch.func import functional_call
from receiver_r2.measurement import clone_parameters,commit_scaled,isolated_mode_rng
from receiver_r2.controller import choose_scale,SCALES


@torch.no_grad()
def losses(model,parameters,data,counter,batch=256):
    result=[]
    for start in range(0,len(data['y']),batch):
        x=data['x'][start:start+batch];y=data['y'][start:start+batch]
        result.append(F.cross_entropy(functional_call(model,parameters,(x,)),y,reduction='none'))
        counter['forward_calls']+=1;counter['packets_forwarded']+=len(y)
    return torch.cat(result).cpu().numpy()


def RX_statistics(before,after,data):
    rr=data['rx'].cpu().numpy();delta=after-before;rows=[]
    for rx in np.unique(rr):
        mask=rr==rx;d=delta[mask];n=len(d)
        rows.append(dict(rx=int(rx),n=n,delta=float(d.mean()),paired_SE=float(d.std(ddof=1)/np.sqrt(n)) if n>1 else float('nan'),resolution=1e-7+1e-6*float(np.abs(before[mask]).mean())))
    return rows


def optimized_adam_step(model,optimizer,C1,C2=None,multi_rx=True,diagnose=False):
    """Receives clipped gradients; proposal, cached C1, early exit, then sparse C2.

    C2 diagnoses only the chosen scale, never all candidates or the decision.
    Cache is per-transaction; it is never reused after parameters change.
    """
    assert not list(model.named_buffers())
    before=clone_parameters(model);old=copy.deepcopy(optimizer.state_dict());optimizer.step();candidate=clone_parameters(model)
    delta={n:candidate[n]-p for n,p in before.items()};counter=dict(forward_calls=0,packets_forwarded=0);checked={};scale=0.
    with isolated_mode_rng(model,False):
        baseline=losses(model,before,C1,counter)
        for trial in SCALES[:-1]:
            params=candidate if trial==1 else {n:p+trial*delta[n] for n,p in before.items()}
            after=losses(model,params,C1,counter);checked[trial]=RX_statistics(baseline,after,C1)
            # Original choose_scale selects the largest passing scale. Restricting
            # its input to this trial preserves the exact guard arithmetic.
            if choose_scale({trial:checked[trial]},multi_rx)==trial:
                scale=trial;chosen_loss=after;break
        if scale==0:chosen_loss=baseline
        c2rows=[]
        if diagnose and C2 is not None:
            b=losses(model,before,C2,counter)
            pars=before if scale==0 else candidate if scale==1 else {n:p+scale*delta[n] for n,p in before.items()}
            a=b if scale==0 else losses(model,pars,C2,counter)
            c2rows=RX_statistics(b,a,C2)
    commit_scaled(model,optimizer,before,old,scale)
    # Exact parameter commit is tested against the candidate; cached accepted
    # losses refer to precisely those committed parameters, without a redundant pass.
    return dict(scale=scale,checked_scales=list(checked),candidate_C1=checked,actual_accepted_C1=RX_statistics(baseline,chosen_loss,C1),actual_accepted_C2=c2rows,**counter)
