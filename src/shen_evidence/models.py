"""Equal-capacity 1D baselines and port of author's non_adv_net.

CIS CNN matches the local receiverAgnosticRFFI non_adv_net architecture;
CE/Adam training here is an explicit baseline recipe, not paper reproduction.
"""
import torch
import torch.nn.functional as F
from torch import nn


class Residual1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(dim,dim,7,padding=3),nn.BatchNorm1d(dim),nn.ReLU(),
                                 nn.Conv1d(dim,dim,7,padding=3),nn.BatchNorm1d(dim))
    def forward(self,x):
        return F.relu(x+self.net(x))


class PacketCNN(nn.Module):
    def __init__(self, num_classes=10, width=32):
        super().__init__()
        layers=[]
        prev=2
        for dim in (width,width*2,width*4):
            layers += [nn.Conv1d(prev,dim,9,stride=4,padding=4),nn.BatchNorm1d(dim),nn.ReLU(),Residual1D(dim)]
            prev=dim
        # 8192 / 4**3 = 128; fixed pooling gives 16 observations and has a
        # deterministic CUDA backward (adaptive pooling does not on V100).
        self.features=nn.Sequential(*layers,nn.AvgPool1d(8,stride=8))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(prev*16,128),nn.ReLU(),nn.Linear(128,num_classes))
    def forward(self,x):
        return self.head(self.features(x))


class Residual2D(nn.Module):
    def __init__(self, in_dim,out_dim):
        super().__init__()
        self.conv1=nn.Conv2d(in_dim,out_dim,3,padding=1)
        self.conv2=nn.Conv2d(out_dim,out_dim,3,padding=1)
        self.skip=nn.Conv2d(in_dim,out_dim,1) if in_dim!=out_dim else nn.Identity()
    def forward(self,x):
        return F.relu(self.skip(x)+self.conv2(F.relu(self.conv1(x))))


class AuthorCISCNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.stem=nn.Conv2d(1,32,7,stride=2)
        self.blocks=nn.Sequential(Residual2D(32,32),Residual2D(32,32),Residual2D(32,64),Residual2D(64,64),nn.AvgPool2d(2))
        self.feature=nn.Linear(13*31*64,512)
        self.tx_input=nn.Linear(512,128)
        self.classifier=nn.Linear(128,num_classes)
        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    def forward(self,x):
        if tuple(x.shape[1:])!=(1,52,126):
            raise ValueError('Author CNN requires CIS [B,1,52,126]')
        # Keras SAME padding for even input with odd kernel/stride 2.
        x=F.relu(self.stem(F.pad(x,(2,3,2,3))))
        x=self.blocks(x).permute(0,2,3,1).reshape(x.shape[0],-1)
        x=F.normalize(self.feature(x),p=2,dim=1,eps=1e-12)
        return self.classifier(F.relu(self.tx_input(x)))


def make_model(kind):
    return AuthorCISCNN() if kind=='cis' else PacketCNN()
