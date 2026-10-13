#!/usr/bin/env python3
"""Synthetic numerical/gradient/sampling tests; zero dataset fitting jobs."""
import math,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from b1_baselines.objectives import GroupDRO,grl,supervised_contrastive
from b1_baselines.data import schedule_from_groups,validate_schedule
from b1_baselines.model import FeatureCNN
from shen_p2a.models import FactorizedAuthorCNN

def main():
    torch.set_num_threads(2)
    features=torch.tensor([[1.,2.],[3.,4.]],requires_grad=True);head=torch.tensor([[2.],[1.]],requires_grad=True)
    value=(grl(features,.1)@head).sum();value.backward()
    assert torch.allclose(features.grad,torch.tensor([[-.2,-.1],[-.2,-.1]]))
    assert torch.allclose(head.grad,torch.tensor([[4.],[6.]]))
    losses=torch.tensor([1.,3.,2.,4.],requires_grad=True);groups=torch.tensor([0,1,0,1]);dro=GroupDRO(2)
    loss,means,weights=dro(losses,groups);expected=torch.softmax(torch.tensor([1.5,3.5])*.01,0)
    assert torch.allclose(weights,expected) and weights[1]>weights[0]
    loss.backward();assert torch.allclose(losses.grad,expected[groups]/2)
    dro(losses.detach(),groups);assert torch.allclose(dro.log_weights.exp(),torch.softmax(torch.tensor([1.5,3.5])*.02,0))
    z=torch.tensor([[1.,0.],[.8,.2],[0.,1.],[.2,.8]],requires_grad=True);y=torch.tensor([0,0,1,1])
    got=supervised_contrastive(z,y);zn=z.detach().numpy()/np.linalg.norm(z.detach().numpy(),axis=1)[:,None]
    independent=[]
    for i in range(4):
        scores=zn@zn[i]/.1;other=[j for j in range(4) if j!=i];pos=next(j for j in other if y[j]==y[i])
        maximum=max(scores[j] for j in other);den=maximum+math.log(sum(math.exp(float(scores[j]-maximum)) for j in other));independent.append(den-float(scores[pos]))
    assert abs(got.item()-np.mean(independent))<2e-6
    got.backward();assert torch.isfinite(z.grad).all()
    assert torch.allclose(got.detach(),supervised_contrastive(z.detach()*7,y),atol=1e-6)
    g=np.repeat(np.arange(50),200);schedule=schedule_from_groups(g,2,20261013);validate_schedule(schedule,g,2)
    assert np.array_equal(schedule,schedule_from_groups(g,2,20261013))
    assert not np.array_equal(schedule,schedule_from_groups(g,2,20261014))
    torch.manual_seed(42);original=FactorizedAuthorCNN(16);torch.manual_seed(42);wrapper=FeatureCNN()
    assert all(torch.equal(v,wrapper.state_dict()[k]) for k,v in original.state_dict().items())
    x=torch.randn(2,1,52,126)
    with torch.no_grad():assert torch.equal(original(x),wrapper(x))
    assert sum(p.numel() for p in wrapper.parameters())==658314
    print('PASS: GRL feature/head gradients, DRO update/gradient, independent SupCon math, exact schedule coverage/reproducibility, unchanged model/init/logits/capacity')
if __name__=='__main__':main()
