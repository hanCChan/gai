"""Receiver physics for Shen Strict CIS (P2E). Torch port of the cached representation
(shen_p2a.representations.paired_maps on the +-62.5 kHz band-limited IQ, g1_data.save_features)
so receiver counterfactuals can be rendered at the IQ level on the GPU, plus per-RX estimates.

Strict CIS = log10 |S(f, n+1) / S(f, n)|^2 over boxcar-128 / hop-64 STFT frames of the rms-normalised,
band-limited packet, rows 38..89 of the fftshifted 128-bin spectrum, 126 adjacent-frame pairs.
A receiver's LTI response cancels in the ratio, so what a receiver can still change is (i) additive
noise (noise floor / ADC), rendered here at the IQ level, and (ii) a non-LTI log-domain pattern common
to all transmitters, estimated as the class-balanced mean CIS of that receiver.
"""
import math

import numpy as np
import torch

FS = 1e6
N = 8192
BAND = 62500.
LO, HI = math.floor(128 * .3), math.ceil(128 * .7)


def _band_mask(device):
    f = torch.fft.fftfreq(N, d=1 / FS, device=device)
    return (f.abs() <= BAND).to(torch.complex128)


def strict_cis(iq):
    """iq: complex (B, 8192) tensor -> (B, 1, 52, 126) float32, matches cached strict_cis."""
    z = torch.fft.ifft(torch.fft.fft(iq.to(torch.complex128), dim=-1) * _band_mask(iq.device), dim=-1)
    z = z / torch.sqrt((z.abs() ** 2).mean(-1, keepdim=True))
    fr = z.unfold(-1, 128, 64)                                   # (B, 127, 128)
    spec = torch.fft.fftshift(torch.fft.fft(fr, dim=-1) / 128, dim=-1) + 1e-12
    spec = spec[..., LO:HI].transpose(-1, -2)                    # (B, 52, 127)
    cis = torch.log10((spec[..., 1:] / spec[..., :-1]).abs() ** 2)
    return cis[:, None].float()


def to_complex(iq2):
    """cached iq (B, 2, 8192) float -> complex128 tensor."""
    iq2 = torch.as_tensor(iq2)
    return torch.complex(iq2[:, 0].double(), iq2[:, 1].double())


def snr_db(iq, noise_band=(1e5, 4e5)):
    """Per-packet in-band SNR (dB) from the raw (not band-limited) IQ: noise density from the
    out-of-band annulus noise_band, signal = in-band power minus that density."""
    p = torch.fft.fft(iq.to(torch.complex128), dim=-1).abs() ** 2
    f = torch.fft.fftfreq(N, d=1 / FS, device=iq.device).abs()
    ib = f <= BAND
    nb = (f > noise_band[0]) & (f < noise_band[1])
    nd = p[:, nb].mean(-1)
    sig = (p[:, ib].sum(-1) - nd * ib.sum()).clamp_min(1e-30)
    return 10 * torch.log10(sig / (nd * ib.sum()))


def add_noise_to_snr(iq, target_db, generator=None):
    """Add complex white Gaussian noise so the in-band SNR drops to target_db (per packet; packets
    already below target are left unchanged). Returns the noisy IQ."""
    p = torch.fft.fft(iq.to(torch.complex128), dim=-1).abs() ** 2
    f = torch.fft.fftfreq(N, d=1 / FS, device=iq.device).abs()
    ib = f <= BAND
    nb = (f > 1e5) & (f < 4e5)
    nd = p[:, nb].mean(-1)
    nib = ib.sum()
    sig = (p[:, ib].sum(-1) - nd * nib).clamp_min(1e-30)
    want = sig / (10 ** (torch.as_tensor(target_db, dtype=torch.float64, device=iq.device) / 10)) / nib
    add = (want - nd).clamp_min(0.)                              # extra noise density per FFT bin
    sigma = torch.sqrt(add / N / 2)                              # per real component, time domain
    shape = iq.shape
    nr = torch.randn(shape, generator=generator, dtype=torch.float64, device=iq.device)
    ni = torch.randn(shape, generator=generator, dtype=torch.float64, device=iq.device)
    return iq + sigma[:, None] * torch.complex(nr, ni)


def rx_offsets(cis_by_rx, y_by_rx):
    """Class-balanced mean CIS per receiver: {key: (52, 126)} (the TX-independent receiver pattern
    plus the global mean; differences between receivers are TX-independent by construction)."""
    out = {}
    for k, x in cis_by_rx.items():
        y = y_by_rx[k]
        out[k] = np.stack([x[y == t].mean(0) for t in np.unique(y)]).mean(0).reshape(x.shape[-2:])
    return out
