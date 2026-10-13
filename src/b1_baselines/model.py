"""Expose the unchanged rank16 baseline's existing normalized512-D feature."""
import torch.nn.functional as F
from shen_p2a.models import FactorizedAuthorCNN

class FeatureCNN(FactorizedAuthorCNN):
    def __init__(self):super().__init__(16)
    def forward_with_features(self,x):
        if tuple(x.shape[1:])!=(1,52,126):raise ValueError('Expected CIS[B,1,52,126]')
        x=F.relu(self.stem(F.pad(x,(2,3,2,3))))
        x=self.blocks(x).permute(0,2,3,1).reshape(x.shape[0],-1)
        features=F.normalize(self.feature(x),p=2,dim=1,eps=1e-12)
        return self.classifier(F.relu(self.tx_input(features))),features
    def forward(self,x):return self.forward_with_features(x)[0]
