"""Known Group DRO, GRL and SupCon objectives. References are in docs/b1_baselines."""
import math
import torch
import torch.nn.functional as F
from torch import nn

class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, coefficient):
        ctx.coefficient=coefficient
        return features.view_as(features)
    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient*gradient, None

def grl(features,coefficient=.1):return ReverseGradient.apply(features,coefficient)

class GroupDRO(nn.Module):
    def __init__(self,groups,eta=.01):
        super().__init__();self.groups=groups;self.eta=eta
        self.register_buffer('log_weights',torch.full((groups,),-math.log(groups)))
    def forward(self,individual_losses,group_ids):
        assert len(individual_losses)==2*self.groups
        order=torch.argsort(group_ids)
        expected=torch.arange(self.groups,device=group_ids.device).repeat_interleave(2)
        assert torch.equal(group_ids[order],expected),'Each batch must include every group exactly twice'
        means=individual_losses[order].reshape(self.groups,2).mean(1)
        with torch.no_grad():
            self.log_weights.add_(self.eta*means.detach())
            self.log_weights.sub_(torch.logsumexp(self.log_weights,dim=0))
        weights=self.log_weights.exp().detach()
        return (weights*means).sum(),means,weights

def supervised_contrastive(features,labels,temperature=.1):
    """One-view standard SupCon, base_temperature=temperature; no custom pairs."""
    features=F.normalize(features,p=2,dim=1,eps=1e-12)
    scores=features@features.T/temperature
    scores=scores-scores.max(dim=1,keepdim=True).values.detach()
    diagonal=torch.eye(len(labels),dtype=torch.bool,device=labels.device)
    positives=(labels[:,None]==labels[None,:])&~diagonal
    positive_count=positives.sum(1)
    assert torch.all(positive_count>0),'SupCon anchors need at least one positive'
    log_denominator=torch.logsumexp(scores.masked_fill(diagonal,-torch.inf),dim=1,keepdim=True)
    log_probability=scores-log_denominator
    return -((log_probability*positives).sum(1)/positive_count).mean()
