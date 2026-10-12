"""Representative GRL and Spec+cross-RX Siamese components, not paper reproduction."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from shen_p2a.models import FactorizedAuthorCNN

class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):ctx.weight=weight;return x.view_as(x)
    @staticmethod
    def backward(ctx,grad):return -ctx.weight*grad,None

class SpecFeatures(FactorizedAuthorCNN):
    def forward(self,x):
        if x.shape[1:]!=(1,52,126):raise ValueError('Same paired STFT geometry required')
        x=F.relu(self.stem(F.pad(x,(2,3,2,3))));x=self.blocks(x).permute(0,2,3,1).flatten(1)
        return F.relu(self.tx_input(F.normalize(self.feature(x),dim=1,eps=1e-12)))

class ReceiverAdversarial(nn.Module):
    def __init__(self,num_train_receivers):
        super().__init__()
        if num_train_receivers<2:raise ValueError('GRL needs at least two training receiver domains')
        self.features=SpecFeatures();self.receiver_head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,num_train_receivers))
    def forward(self,x,grl_weight=1.):
        z=self.features(x)
        return {'logits':self.features.classifier(z),'receiver_logits':self.receiver_head(Reverse.apply(z,grl_weight))}

class SpecSiamese(nn.Module):
    def __init__(self):
        super().__init__();self.features=SpecFeatures();self.projection=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,64))
    def forward(self,a,b):
        z=torch.cat((self.features(a),self.features(b)))
        return {'logits':self.features.classifier(z),'contrastive':F.normalize(self.projection(z),dim=1)}

def nt_xent(z,temperature=.1):
    if z.ndim!=2 or len(z)%2 or len(z)<4 or temperature<=0:raise ValueError('Expect two aligned halves, at least two device pairs')
    z=F.normalize(z,dim=1);n=len(z)//2;target=(torch.arange(2*n,device=z.device)+n)%(2*n)
    score=z@z.T/temperature;score=score.masked_fill(torch.eye(2*n,dtype=torch.bool,device=z.device),-torch.inf)
    return F.cross_entropy(score,target)

def grl_loss(output,transmitter_labels,training_receiver_labels):
    # Domain coefficient belongs in Reverse.apply; multiplying twice would square it.
    tx=F.cross_entropy(output['logits'],transmitter_labels)
    rx=F.cross_entropy(output['receiver_logits'],training_receiver_labels)
    return tx+rx,{'transmitter_ce':tx,'receiver_ce':rx}

def siamese_loss(output,pair_labels,contrastive_weight=1.,temperature=.1):
    if contrastive_weight<0:raise ValueError('Contrastive weight must be nonnegative')
    if len(pair_labels)!=len(torch.unique(pair_labels)):
        raise ValueError('One transmitter pair per batch is required to avoid false negatives')
    if output['logits'].shape[0]!=2*len(pair_labels):raise ValueError('Paired views must be aligned')
    ce=F.cross_entropy(output['logits'],torch.cat((pair_labels,pair_labels)))
    contrastive=nt_xent(output['contrastive'],temperature)
    return ce+contrastive_weight*contrastive,{'transmitter_ce':ce,'nt_xent':contrastive}

def cross_receiver_pairs(labels,receivers,seed=20261012):
    """One device pair per batch prevents same-device false negatives; indices only."""
    labels=np.asarray(labels);receivers=np.asarray(receivers)
    if labels.shape!=receivers.shape or labels.ndim!=1:raise ValueError('Aligned training metadata required')
    rng=np.random.default_rng(seed);pairs=[]
    for label in rng.permutation(np.unique(labels)):
        domains=np.unique(receivers[labels==label])
        if len(domains)<2:raise ValueError('Every transmitter needs two TRAINING receivers')
        a,b=rng.choice(domains,2,replace=False)
        left=int(rng.choice(np.flatnonzero((labels==label)&(receivers==a))));right=int(rng.choice(np.flatnonzero((labels==label)&(receivers==b))))
        pairs.append((left,right,int(label)))
    return pairs

def fractional_tdl_awgn(iq,fs=1e6,delay_spread_ns=100.,doppler_hz=0.,snr_db=25.,seed=None):
    """Finite-support sinc fractional delays, exponential PDP, static sinusoidal Doppler.

    This approximates a short TDL channel; it is NOT a full Jakes reproduction.
    Inputs are assumed sampled at Fs=1MHz. Normalize CIR energy then received
    power defines AWGN. Zero-pad and align interpolation latency; finite-record
    filter transients remain at packet boundaries. Never fit on development.
    """
    if iq.ndim!=1 or not np.iscomplexobj(iq):raise ValueError('Complex packet vector required')
    if seed is None:raise ValueError('An explicit per-packet seed is required for reproducible online augmentation')
    if not 5<=delay_spread_ns<=300 or not 0<=doppler_hz<=5 or not 10<=snr_db<=40:raise ValueError('Reference augmentation ranges exceeded')
    rng=np.random.default_rng(seed);spread=delay_spread_ns*1e-9
    factors=np.array([0.,.5,1.,2.,3.,5.])
    power=np.exp(-factors);power/=power.sum()
    pdp_std=np.sqrt(np.sum(power*(factors-np.sum(power*factors))**2))
    delays=factors/pdp_std*spread*fs
    taps=(rng.normal(size=6)+1j*rng.normal(size=6))*np.sqrt(power/2)
    k=np.arange(-32,34);kernel=np.sinc(k[None,:]-delays[:,None])*np.hanning(len(k))[None,:]
    kernel/=kernel.sum(1,keepdims=True);h=(taps[:,None]*kernel).sum(0);h/=np.sqrt((np.abs(h)**2).sum())
    # Full convolution with zero padding; select the aligned finite observation.
    y=np.convolve(iq,h,mode='full')[32:32+len(iq)];t=np.arange(len(iq))/fs
    y=y*np.exp(2j*np.pi*doppler_hz*t);p=np.mean(np.abs(y)**2)
    noise=(rng.normal(size=len(iq))+1j*rng.normal(size=len(iq)))*np.sqrt(p/(2*10**(snr_db/10)))
    return y+noise,{'seed':int(seed),'delay_samples':delays.tolist(),'sampling_assumption_hz':fs,'rms_delay_target_ns':delay_spread_ns,'snr_db_target':snr_db,'doppler_hz_target':doppler_hz,'effective_discrete_kernel_energy':float(np.sum(np.abs(h)**2)),'doppler_model':'single sinusoid, not Jakes','paper_full_reproduction':False}

def augment_training_packet(iq,rng,fs=1e6):
    if not isinstance(rng,np.random.Generator):raise ValueError('Pass a seeded training-only generator')
    return fractional_tdl_awgn(iq,fs=fs,delay_spread_ns=float(rng.uniform(5,300)),
                              doppler_hz=float(rng.uniform(0,5)),snr_db=float(rng.uniform(10,40)),
                              seed=int(rng.integers(0,2**32)))
