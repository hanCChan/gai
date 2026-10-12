"""Frequency boundaries in assumed physical Hz; no CFO correction.

Normalization follows isolation: excluded bins cannot influence the scale.
Full/IB use packet RMS; OOB uses OOB-only spectral RMS. The guard is excluded
from both inputs. Fs/BW are literature assumptions, not verified metadata.
"""
import torch

KINDS = ('full_iq', 'strict_ib', 'oob_only', 'cis')


def frequency_masks(length, device, sample_rate_hz=1e6, bandwidth_hz=125e3, guard_hz=12500.):
    if not 0 < bandwidth_hz < sample_rate_hz or not 0 <= guard_hz < (sample_rate_hz-bandwidth_hz)/2:
        raise ValueError('Invalid sampling rate, bandwidth or guard')
    f = torch.fft.fftshift(torch.fft.fftfreq(length, d=1/sample_rate_hz, device=device))
    ib = f.abs() <= bandwidth_hz/2
    oob = f.abs() > bandwidth_hz/2 + guard_hz
    return f, ib, oob


def rms_channels(x):
    power = x.square().sum(1, keepdim=True).mean(-1, keepdim=True)
    if not torch.isfinite(power).all() or (power <= 0).any():
        raise ValueError('Nonfinite or zero-energy input')
    return x / power.sqrt()


def build_input(iq, kind, sample_rate_hz=1e6, bandwidth_hz=125e3, guard_hz=12500.):
    if kind == 'cis':
        return iq  # Author-derived float64 SciPy transform occurs in the worker.
    if iq.ndim != 3 or iq.shape[1] != 2 or not torch.isfinite(iq).all():
        raise ValueError('Expected finite IQ [B,2,T]')
    if kind == 'full_iq':
        return rms_channels(iq)
    z = torch.complex(iq[:,0], iq[:,1])
    spectrum = torch.fft.fftshift(torch.fft.fft(z, norm='ortho'), dim=-1)
    _, ib, oob = frequency_masks(iq.shape[-1], iq.device, sample_rate_hz, bandwidth_hz, guard_hz)
    if kind == 'strict_ib':
        isolated = torch.fft.ifft(torch.fft.ifftshift(spectrum * ib, dim=-1), norm='ortho')
    elif kind == 'oob_only':
        isolated = spectrum * oob
    else:
        raise ValueError(kind)
    # Mask first; neither normalization nor any other view sees excluded IQ.
    return rms_channels(torch.stack((isolated.real, isolated.imag), dim=1))
