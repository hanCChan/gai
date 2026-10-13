"""Minimal source-only finite Adam scaling policy; no controlled fitting in R2-A.

This policy is frozen before the aggregate C2 analysis. Its noise tolerances
are recomputed from C1 paired-packet estimation errors only, never from C2.
"""
import math

SCALES=(1.,.5,.25,0.)
NOISE_MULTIPLIER=2.


def choose_scale(c1_by_scale,multi_rx=True):
    """Input: each nonzero scale -> per-RX (delta, paired_SE, resolution, n).

    First guard: mean risk <= mean2SE tolerance. Multi-RX adds a harm-budget
    guard: each RX's excess over its2SE noise margin may not exceed the mean
    measured improvement plus the mean noise allowance. Positive individual
    RX changes are permitted. All coefficients are fixed; no fitted threshold.
    """
    for scale in SCALES[:-1]:
        cells=c1_by_scale.get(scale,[])
        if not cells or any(c['n']<2 or not all(math.isfinite(c[k]) for k in ['delta','paired_SE','resolution']) for c in cells):continue
        means=[c['delta'] for c in cells];noise=[NOISE_MULTIPLIER*c['paired_SE']+c['resolution'] for c in cells]
        mean=sum(means)/len(cells);mean_noise=sum(noise)/len(cells)
        if mean>mean_noise:continue
        budget=max(0.,-mean)+mean_noise
        if multi_rx and max(d-e for d,e in zip(means,noise))>budget:continue
        return scale
    return 0.


def controlled_adam_step(model,optimizer,C1,C2,epoch,step,seed,multi_rx=True):
    """Executable minimal controller, after caller computes/clips A/B gradients.

    Complete proposal/decision/remeasurement/commit transaction. The C2 call
    occurs after the scale is fixed; no C2 field enters the decision. This is
    tested on CPU only in this stage; it does not authorize a controlled fit.
    """
    import copy
    from .measurement import clone_parameters,risk_measurement,commit_scaled
    before=clone_parameters(model);old=copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    c1,_=risk_measurement(model,before,{'C1':C1},epoch,step,seed,train_replicates=0)
    inputs={s:[dict(delta=r['delta'],paired_SE=r['paired_delta_SE'],resolution=r['resolution'],n=r['n'])
               for r in c1 if r['scale']==s and r['tx']==-999] for s in SCALES[:-1]}
    scale=choose_scale(inputs,multi_rx=multi_rx)
    c2,_=risk_measurement(model,before,{'C2':C2},epoch,step,seed,train_replicates=0) if C2 is not None else ([],[])
    commit_scaled(model,optimizer,before,old,scale)
    accepted,_=risk_measurement(model,before,{'C1':C1},epoch,step,seed,scales=(1.,0.),train_replicates=0)
    return dict(scale=scale,candidate_C1=c1,candidate_C2=c2,actual_accepted_C1=accepted)
