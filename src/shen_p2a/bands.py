"""Explicit original-FFT projection, common coordinate system and controlled scale.

All formal inputs project before windowing. Hann-before-projection is reserved
for leakage diagnostics and is rejected here: it mixes original Fourier bins.
ib_relative OOB contains explicitly declared IB-energy side information.
"""
import torch
from shen_evidence.inputs import build_input as p1_input

BANDS=('full','ib','near','mid','far','oob')


def masks(n,device,fs=1e6,bw=125e3,guard=12500.):
    if not 0<bw<fs or not 0<=guard<62500.: raise ValueError('Invalid frequency definition')
    f=torch.fft.fftshift(torch.fft.fftfreq(n,d=1/fs,device=device))
    a=f.abs(); edge=bw/2+guard
    result={'full':torch.ones_like(a,dtype=torch.bool),'ib':a<=bw/2,
            'near':(a>edge)&(a<=125000.),'mid':(a>125000.)&(a<=250000.),
            'far':(a>250000.)&(a<=fs/2),'oob':a>edge}
    if edge>=125000. or fs/2<=250000.:raise ValueError('Configured ring geometry requires Fs=1MHz/BW=125kHz assumptions')
    return f,result


def spectrum(iq):
    if iq.ndim!=3 or iq.shape[1]!=2 or not torch.isfinite(iq).all():raise ValueError('Expected finite IQ [B,2,T]')
    return torch.fft.fftshift(torch.fft.fft(torch.complex(iq[:,0],iq[:,1]),norm='ortho'),dim=-1)


def selected_view(spec,band,coordinate='frequency',normalization='independent',window='rect',fs=1e6,bw=125e3,guard=12500.):
    _,mask=masks(spec.shape[-1],spec.device,fs,bw,guard)
    if band not in BANDS:raise ValueError(band)
    selected=spec*mask[band]
    if window=='projection_hann':
        time=torch.fft.ifft(torch.fft.ifftshift(selected,dim=-1),norm='ortho')
        windowed=time*torch.hann_window(time.shape[-1],periodic=True,device=time.device,dtype=time.real.dtype)
        selected=torch.fft.fftshift(torch.fft.fft(windowed,norm='ortho'),dim=-1)*mask[band]
    elif window!='rect':raise ValueError('Pre-projection windows are not strict information isolation')
    if normalization=='independent':reference=selected
    elif normalization=='ib_relative':
        if band in ('ib','full'):raise ValueError('IB-relative normalization is auxiliary evidence only')
        reference=spec*mask['ib']
    else:raise ValueError(normalization)
    power=reference.abs().square().mean(-1,keepdim=True)
    if not torch.isfinite(power).all() or (power<=0).any():raise ValueError('Nonfinite/zero selected energy')
    selected=selected/power.sqrt()
    if coordinate=='time':selected=torch.fft.ifft(torch.fft.ifftshift(selected,dim=-1),norm='ortho')
    elif coordinate!='frequency':raise ValueError(coordinate)
    return torch.stack((selected.real,selected.imag),dim=1)


def transform(iq,cfg):
    if cfg['family']=='stability':
        return p1_input(iq,'full_iq' if cfg['band']=='full' else 'strict_ib',cfg['sample_rate_hz'],cfg['bandwidth_hz'],cfg['guard_hz'])
    spec=spectrum(iq)
    common=dict(fs=cfg['sample_rate_hz'],bw=cfg['bandwidth_hz'],guard=cfg['guard_hz'])
    if cfg['family']=='fusion':
        # Main path cannot see auxiliary/global energy or pre-windowed IQ.
        main=p1_input(iq,'strict_ib',cfg['sample_rate_hz'],cfg['bandwidth_hz'],cfg['guard_hz'])
        band='ib' if cfg['aux_band']=='ib_control' else cfg['aux_band']
        aux=main if band=='none' else selected_view(spec,band,coordinate='time',normalization='independent',window='rect',**common)
        return {'main':main,'aux':aux}
    return selected_view(spec,cfg['band'],coordinate=cfg['coordinate'],normalization=cfg['normalization'],window=cfg['window'],**common)
