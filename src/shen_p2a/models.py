"""Capacity control changes only author's dense feature layer to rank 16.

Spec/CIS use identical geometry/encoder. Fusion includes exact IB+IB capacity
controls, so auxiliary branch capacity is not mistaken for OOB information.
"""
import torch
from torch import nn
from shen_evidence.models import PacketCNN,AuthorCISCNN


class FactorizedAuthorCNN(AuthorCISCNN):
    def __init__(self,rank=16):
        super().__init__()
        self.feature=nn.Sequential(nn.Linear(13*31*64,rank,bias=False),nn.Linear(rank,512))
        for layer in self.feature:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:nn.init.zeros_(layer.bias)


class PacketEncoder(nn.Module):
    def __init__(self):
        super().__init__(); original=PacketCNN()
        self.features=original.features; self.embedding=nn.Sequential(*list(original.head.children())[:-1])
    def forward(self,x):return self.embedding(self.features(x))


class SimpleFusion(nn.Module):
    def __init__(self,mode):
        super().__init__(); self.main=PacketEncoder(); self.aux=PacketEncoder(); self.mode=mode
        if mode=='concat':self.classifier=nn.Linear(256,10)
        elif mode=='scalar_gate':
            self.gate=nn.Linear(256,1); self.classifier=nn.Linear(128,10)
        else:raise ValueError(mode)
    def forward(self,x):
        a,b=self.main(x['main']),self.aux(x['aux'])
        combined=torch.cat((a,b),1)
        if self.mode=='concat':return {'logits':self.classifier(combined)}
        gate=torch.sigmoid(self.gate(combined))
        return {'logits':self.classifier((1-gate)*a+gate*b),'gate':gate[:,0]}


class SingleIB(nn.Module):
    def __init__(self):super().__init__(); self.net=PacketCNN()
    def forward(self,x):return {'logits':self.net(x['main'])}


def make_model(cfg):
    if cfg['family']=='representation':
        return AuthorCISCNN() if cfg['model']=='author' else FactorizedAuthorCNN(cfg.get('rank',16))
    if cfg['family']=='fusion':return SingleIB() if cfg['fusion']=='single_ib' else SimpleFusion(cfg['fusion'])
    return PacketCNN()
