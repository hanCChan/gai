"""Matched STFT geometry; author's CIS is unchanged, Spec aligns numerator frames."""
import math
import numpy as np
from scipy import signal
from shen_stage1.representations import rms_normalize_complex

STFT_CONFIG = {'window':'boxcar','win_len':128,'hop':64,'crop_ratio':.3,
               'boundary':None,'padded':False,'epsilon':1e-12,'spec_frames':'numerator_columns_1_to_126'}


def stft_pair(iq):
    x=rms_normalize_complex(iq)
    _,_,spec=signal.stft(x,window='boxcar',nperseg=128,noverlap=64,nfft=128,
                         return_onesided=False,padded=False,boundary=None,axis=-1)
    spec=np.fft.fftshift(spec,axes=-2)+1e-12
    lo,hi=math.floor(128*.3),math.ceil(128*.7)
    return spec[:,lo:hi,:-1],spec[:,lo:hi,1:]


def paired_maps(iq):
    denominator,numerator=stft_pair(iq)
    cis=np.log10(np.abs(numerator/denominator)**2)
    spec=np.log10(np.abs(numerator)**2)
    if not np.isfinite(cis).all() or not np.isfinite(spec).all():
        raise ValueError('Nonfinite representation; no silent clipping is allowed')
    return spec[:,None].astype(np.float32),cis[:,None].astype(np.float32),denominator
