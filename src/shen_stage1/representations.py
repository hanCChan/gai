"""Explicit RMS and author-derived CIS transforms; no CFO compensation.

Inputs are stored complex packet IQ, not assumed to be unprocessed receiver
samples. Normalization uses only the current packet. No dataset statistics are
fitted. Output precision is explicit; the reference path uses float64.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import signal


def rms_normalize_complex(iq):
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite complex IQ [packets, samples]')
    rms = np.sqrt(np.mean(np.abs(x) ** 2, axis=1, keepdims=True))
    if not np.isfinite(rms).all() or np.any(rms <= 0):
        raise ValueError('Packet RMS must be finite and nonzero')
    return x / rms


def iq_rms_channels(iq, dtype=np.float32):
    x = rms_normalize_complex(iq)
    return np.stack((x.real, x.imag), axis=1).astype(dtype)


def channel_independent_spectrogram(iq, win_len=128, crop_ratio=.3, dtype=np.float64):
    if win_len < 2 or not 0 <= crop_ratio < .5:
        raise ValueError('Invalid CIS window or crop ratio')
    x = rms_normalize_complex(iq)
    if x.shape[-1] < win_len:
        raise ValueError('Input shorter than STFT window')
    _, _, spec = signal.stft(x, window='boxcar', nperseg=win_len,
                            noverlap=round(.5 * win_len), nfft=win_len,
                            return_onesided=False, padded=False, boundary=None, axis=-1)
    spec = np.fft.fftshift(spec, axes=-2) + 1e-12
    ratio = spec[..., 1:] / spec[..., :-1]
    cis = np.log10(np.abs(ratio) ** 2)
    lo, hi = math.floor(win_len * crop_ratio), math.ceil(win_len * (1 - crop_ratio))
    cropped = cis[:, lo:hi, :]
    if not np.isfinite(cropped).all():
        raise ValueError('Nonfinite CIS output; numerical stabilization requires a separate, labeled ablation')
    return cropped[:, None, :, :].astype(dtype)
