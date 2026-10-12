"""Same IQ tokenizer for RF-HSTU and standard Transformer; distinct physical axes."""
import torch
from torch import nn
from rfhstu.models import RFHSTUBlock
from rfhstu.oob_fusion import OOBCrossAttentionFusion
from shen_p2a.bands import spectrum,selected_view

class AxisTokenizer(nn.Module):
    def __init__(self,dim=128,patch=256,axis='time'):
        super().__init__();self.patch=patch;self.axis=axis;self.projection=nn.Linear(2*patch,dim);self.position=nn.Parameter(torch.zeros(1,8192//patch,dim))
    def forward(self,x):
        if x.shape[1:]!=(2,8192):raise ValueError('8192-point physical coordinate required')
        patches=x.reshape(x.shape[0],2,8192//self.patch,self.patch).permute(0,2,1,3).flatten(2)
        return self.projection(patches)+self.position

class FairAttention(nn.Module):
    def __init__(self,encoder='hstu',fusion='none',dim=128,depth=7,ffn_dim=64,patch=256):
        super().__init__();self.encoder=encoder;self.fusion=fusion
        self.main_tokens=AxisTokenizer(dim,patch,'time');n=8192//patch
        if encoder=='hstu':self.blocks=nn.ModuleList([RFHSTUBlock(dim,n,0.) for _ in range(depth)])
        elif encoder=='transformer':
            self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(dim,4,ffn_dim,dropout=0.,activation='gelu',batch_first=True,norm_first=False) for _ in range(depth)])
        else:raise ValueError(encoder)
        self.norm=nn.LayerNorm(dim);self.classifier=nn.Linear(dim,10)
        if fusion=='cross_attention':
            self.aux_tokens=AxisTokenizer(dim,patch,'frequency');self.cross=OOBCrossAttentionFusion(dim,4,0.)
        elif fusion!='none':raise ValueError(fusion)
    def tokenize(self,iq):
        s=spectrum(iq)
        main=selected_view(s,'ib',coordinate='time',normalization='independent')
        tokens=self.main_tokens(main)
        if self.fusion=='cross_attention':
            aux=selected_view(s,'oob',coordinate='frequency',normalization='independent')
            # No shared time/chirp embedding: these are ordered frequency bins.
            tokens=self.cross(tokens,self.aux_tokens(aux))
        return tokens
    def forward(self,iq):
        x=self.tokenize(iq)
        for block in self.blocks:x=block(x)
        return self.classifier(self.norm(x).mean(1))
