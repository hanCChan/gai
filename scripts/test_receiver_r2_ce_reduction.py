#!/usr/bin/env python3
"""CPU check explaining CE logging reductions without a fitting optimizer step."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from wisig_w01_common import OfficialCNN


def main():
    torch.set_num_threads(2);cases=[]
    for seed in [20261013,20261014,20261015]:
        torch.manual_seed(seed);m=OfficialCNN(10);m.train()
        for n in [560,400]:
            x=torch.randn(n,1,256,2);y=torch.arange(n)%10;torch.manual_seed(seed+41)
            z=m(x);direct=torch.nn.functional.cross_entropy(z,y);packet_mean=torch.nn.functional.cross_entropy(z,y,reduction='none').mean()
            a=torch.autograd.grad(direct,tuple(m.parameters()),retain_graph=True);b=torch.autograd.grad(packet_mean,tuple(m.parameters()))
            exact=all(torch.equal(u,v) for u,v in zip(a,b));assert exact
            cases.append(dict(seed=seed,batch=n,logged_objective_difference=abs(float(direct)-float(packet_mean)),all_parameter_gradients_exact=exact))
    result=dict(passed=True,optimizer_steps=0,CPU_synthetic_only=True,explanation='Different loss scalar reductions; identical per-logit and parameter gradients',cases=cases)
    (ROOT/'reports/receiver_r2/preflight/ce_reduction_equivalence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
