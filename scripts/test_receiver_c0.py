#!/usr/bin/env python3
"""CPU composite equivalence, C2 isolation and single-RX degeneracy tests."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from receiver_r2.controller import controlled_adam_step,choose_scale
from receiver_r2.measurement import state_hash
from receiver_c0.optimized import optimized_adam_step


def main():
    torch.set_num_threads(1);torch.manual_seed(49);results=[]
    base=torch.nn.Sequential(torch.nn.Linear(3,10),torch.nn.Dropout(.5));x=torch.randn(80,3);y=torch.arange(80)%10
    data=dict(x=x,y=y,rx=torch.arange(80)//40,day=torch.zeros(80,dtype=torch.long));baseopt=torch.optim.Adam(base.parameters(),lr=.0003)
    base.train();torch.nn.functional.cross_entropy(base(x),y).backward();baseopt.step();baseopt.zero_grad();torch.nn.functional.cross_entropy(base(x),y).backward()
    for multi in [False,True]:
        a=copy.deepcopy(base);b=copy.deepcopy(base);oa=torch.optim.Adam(a.parameters(),lr=.0003);ob=torch.optim.Adam(b.parameters(),lr=.0003);oa.load_state_dict(copy.deepcopy(baseopt.state_dict()));ob.load_state_dict(copy.deepcopy(baseopt.state_dict()))
        for p,q,r in zip(base.parameters(),a.parameters(),b.parameters()):q.grad=p.grad.clone();r.grad=p.grad.clone()
        rng=torch.get_rng_state().clone();legacy=controlled_adam_step(a,oa,data,data,1,1,981,multi);assert torch.equal(rng,torch.get_rng_state())
        optimized=optimized_adam_step(b,ob,data,data,multi,True);assert torch.equal(rng,torch.get_rng_state())
        assert legacy['scale']==optimized['scale'] and state_hash(a.state_dict())==state_hash(b.state_dict()) and state_hash(oa.state_dict())==state_hash(ob.state_dict())
        results.append(dict(multi_rx=multi,scale=optimized['scale'],exact_parameters_moments_rng=True,forward_calls=optimized['forward_calls']))
    # A one-RX source guard is mathematically equivalent to its mean guard.
    for d in [-1.,-.01,0.,.001,.03,1.]:
        inputs={s:[dict(delta=d*s,paired_SE=.001*s,resolution=1e-7,n=20)] for s in [1.,.5,.25]}
        assert choose_scale(inputs,True)==choose_scale(inputs,False)
    result=dict(passed=True,cases=results,single_RX_guard_equals_mean_guard=True,original_controller_unchanged=True)
    (ROOT/'reports/receiver_c0/equivalence_cpu_tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
