#!/usr/bin/env python3
"""Deterministic CPU tests of state transactions and risk measurement isolation."""
import copy,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from receiver_r2.measurement import clone_parameters,state_hash,commit_scaled,risk_measurement


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(81)
    model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.ReLU(),torch.nn.Dropout(.5),torch.nn.Linear(5,10))
    opt=torch.optim.Adam(model.parameters(),lr=.0003)
    x=torch.randn(80,3);y=torch.arange(80)%10
    model.train();model(x).sum().backward();opt.step();opt.zero_grad()
    before=clone_parameters(model);old=copy.deepcopy(opt.state_dict());model(x).sum().backward();opt.step()
    candidate=clone_parameters(model);new=copy.deepcopy(opt.state_dict())
    result={}
    for scale in [1.,.5,.25,0.]:
        with torch.no_grad():
            for n,p in model.named_parameters():p.copy_(candidate[n])
        opt.load_state_dict(copy.deepcopy(new));commit_scaled(model,opt,before,old,scale)
        for n,p in model.named_parameters():
            expected=before[n] if scale==0 else candidate[n] if scale==1 else before[n]+scale*(candidate[n]-before[n])
            assert torch.equal(p,expected)
        assert state_hash(opt.state_dict())==state_hash(old if scale==0 else new)
        result[str(scale)]=True
    # Rejected Adam history gives exactly the same next step as an untouched clone.
    reference=copy.deepcopy(model);refopt=torch.optim.Adam(reference.parameters(),lr=.0003);refopt.load_state_dict(copy.deepcopy(old))
    for target,optimizer in [(model,opt),(reference,refopt)]:
        optimizer.zero_grad();torch.manual_seed(99);torch.nn.functional.cross_entropy(target(x),y).backward();optimizer.step()
    assert state_hash(model.state_dict())==state_hash(reference.state_dict())
    assert state_hash(opt.state_dict())==state_hash(refopt.state_dict())
    data=dict(x=x,y=y,rx=torch.arange(80)//40,day=torch.zeros(80,dtype=torch.long))
    before_probe=clone_parameters(model)
    with torch.no_grad():
        for p in model.parameters():p.add_(.0001)
    rng=torch.get_rng_state().clone();mode=model.training;parameter_hash=state_hash(model.state_dict());oh=state_hash(opt.state_dict())
    rows,raw=risk_measurement(model,before_probe,{'C1':data,'C2':data},1,1,999)
    assert torch.equal(rng,torch.get_rng_state()) and mode==model.training
    assert parameter_hash==state_hash(model.state_dict()) and oh==state_hash(opt.state_dict())
    rows2,_=risk_measurement(model,before_probe,{'C1':data,'C2':data},1,1,999)
    assert rows==rows2
    result.update(rollback_next_update_exact=True,dropout_common_random_numbers=True,rng_and_mode_restored=True,candidate_equals_actual_scale1=True,scale0_exact_zero=True,repeated_measurement_exact=True)
    path=ROOT/'reports/receiver_r2/preflight/deterministic_tests.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
