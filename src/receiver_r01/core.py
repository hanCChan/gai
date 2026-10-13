"""Frozen R0/R1 methods. MLDG uses the original differentiable SGD objective."""
import hashlib
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from b1_baselines.model import FeatureCNN
from shen_evidence.models import Residual2D


class WiSigResNet(nn.Module):
    """Fixed capacity control: 16/32/64 channels, six residual blocks, no BN."""
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),
            Residual2D(16,16),Residual2D(16,16),nn.AvgPool2d((2,1)),
            Residual2D(16,32),Residual2D(32,32),nn.AvgPool2d((2,2)),
            Residual2D(32,64),Residual2D(64,64),nn.AvgPool2d((16,1)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(256,80),nn.ReLU(),nn.Dropout(.5),nn.Linear(80,10))
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x):return self.head(self.body(x))
    def regularization(self):return .0001*(self.head[1].weight.square().sum()+self.head[4].weight.square().sum())


def payload(model):
    return hashlib.sha256(b''.join(p.detach().cpu().contiguous().numpy().tobytes() for p in model.state_dict().values())).hexdigest()


def regularization(model):
    return model.regularization() if hasattr(model,'regularization') else next(model.parameters()).new_zeros(())


def mldg_loss(model,x,y,train,held,alpha=.01,beta=1.):
    """Full second-order MLDG, SGD inner step; no held samples in inner loss."""
    assert train.any() and held.any() and not (train & held).any()
    params=OrderedDict(model.named_parameters())
    source=F.cross_entropy(model(x[train]),y[train])
    grads=torch.autograd.grad(source,tuple(params.values()),create_graph=True)
    updated=OrderedDict((name,p-alpha*g) for (name,p),g in zip(params.items(),grads))
    transferred=F.cross_entropy(functional_call(model,updated,(x[held],)),y[held])
    # Divide by 1+beta to make the outer gradient scale comparable to CE.
    return (source+beta*transferred)/(1+beta)+regularization(model)


def pcgrad_backward(model,x,y,train,held):
    """Standard PCGrad on two RX partitions; symmetric projections, equal task mean."""
    params=list(model.parameters())
    losses=[F.cross_entropy(model(x[mask]),y[mask]) for mask in (train,held)]
    grads=[torch.autograd.grad(loss,params) for loss in losses]
    dot=sum((a*b).sum() for a,b in zip(*grads))
    norms=[sum(g.square().sum() for g in gs).clamp_min(1e-12) for gs in grads]
    a=[g-dot.clamp_max(0)/norms[1]*h for g,h in zip(grads[0],grads[1])]
    b=[h-dot.clamp_max(0)/norms[0]*g for g,h in zip(grads[0],grads[1])]
    reg=regularization(model)
    rg=torch.autograd.grad(reg,params,allow_unused=True) if reg.requires_grad else [None]*len(params)
    for p,g,h,r in zip(params,a,b,rg):p.grad=(g+h)/2+(r if r is not None else 0)
    return (losses[0]+losses[1]).detach()/2+reg.detach()


def balanced_schedule(group,epochs,seed):
    members=[np.flatnonzero(group==g) for g in np.unique(group)]
    n=len(members[0]);assert n%2==0 and all(len(a)==n for a in members)
    rng=np.random.default_rng(seed)
    result=[]
    for _ in range(epochs):
        a=np.stack([rng.permutation(m) for m in members]).reshape(len(members),n//2,2)
        result.append(np.stack([a[rng.permutation(len(members)),i].reshape(-1) for i in range(n//2)]))
    return np.array(result)


def meta_masks(rx,day,rng):
    domains=np.unique(rx.cpu().numpy());held_ids=rng.permutation(domains)[:max(1,len(domains)//2)]
    held=torch.isin(rx,torch.tensor(held_ids,device=rx.device));return ~held,held


@torch.no_grad()
def predict(model,x,batch=256):
    model.eval()
    return torch.cat([model(x[i:i+batch]) for i in range(0,len(x),batch)])


def metrics(logits,y):
    pred=logits.argmax(1);cm=np.zeros((10,10),np.int64);np.add.at(cm,(y,pred),1)
    present=cm.sum(1)>0
    recall=np.divide(np.diag(cm),cm.sum(1),out=np.full(10,np.nan),where=present)
    # Supported true classes only. Predictions into absent classes still count as errors.
    f1=np.divide(2*np.diag(cm),cm.sum(1)+cm.sum(0),out=np.zeros(10),where=(cm.sum(1)+cm.sum(0))>0)
    z=logits-logits.max(1,keepdims=True);ce=float(np.mean(np.log(np.exp(z).sum(1))-z[np.arange(len(y)),y]))
    return dict(accuracy=float(np.trace(cm)/len(y)),macro_f1=float(f1[present].mean()),
                worst_tx_recall=float(np.nanmin(recall)),balanced_accuracy=float(np.nanmean(recall)),
                CE=ce,packets=len(y),classes_present=int(present.sum()),complete_coverage=bool(present.all()),
                recall=[float(r) if np.isfinite(r) else None for r in recall],confusion=cm.tolist())
