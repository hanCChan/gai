"""Frequency-response-inspired augmentation; no causal hardware interpretation.

All transforms use complex128 and the unchanged historical CIS frontend.
The six bases are periodic, smooth functions of normalized FFT frequency.
"""
import numpy as np
from shen_stage1.representations import rms_normalize_complex
from shen_p2a.representations import stft_pair


def basis(n=8192):
    f = np.fft.fftfreq(n)
    return np.stack([fn(2 * np.pi * h * f)
                     for h in (1, 2, 3) for fn in (np.cos, np.sin)], axis=1)


def project_style(profile):
    b = basis(len(profile))
    return b @ (b.T @ (profile - np.mean(profile)) / (len(profile) / 2))


def bounded_log_response(difference, max_db=3.0):
    projected = project_style(difference)
    db = 20 / np.log(10) * projected
    # Shrink the entire curve, rather than clip individual bins or amplify it.
    factor = min(1.0, max_db / max(float(np.max(np.abs(db))), 1e-30))
    return projected * factor, factor


def perturb(iq, log_response):
    x = np.asarray(iq, dtype=np.complex128)
    h = np.exp(np.asarray(log_response, dtype=np.float64))
    return np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1)


def log_spectrum(iq):
    x = rms_normalize_complex(iq)
    return np.log(np.abs(np.fft.fft(x, axis=-1)) / np.sqrt(x.shape[-1]) + 1e-12)


def cis_details(iq):
    den, num = stft_pair(iq)
    c = np.log10(np.abs(num / den) ** 2)
    if not np.isfinite(c).all():
        raise ValueError('Nonfinite CIS; no clipping or epsilon retuning')
    return c, np.abs(den) ** 2


def correlation(original, changed):
    a = np.asarray(original, dtype=np.complex128)
    b = np.asarray(changed, dtype=np.complex128)
    return np.abs(np.sum(a.conj() * b, axis=-1)) / np.sqrt(
        np.sum(np.abs(a) ** 2, axis=-1) * np.sum(np.abs(b) ** 2, axis=-1))


def packet_metrics(clean, changed, denominator_power, sigma):
    squared = ((changed - clean) / (sigma + 1e-12)) ** 2
    low = denominator_power <= np.quantile(denominator_power, .01, axis=(1, 2), keepdims=True)
    extreme = np.abs(clean) > 10  # log10(power ratio), i.e. 100 dB
    boundary = np.zeros_like(low)
    boundary[:, :, :2] = True
    boundary[:, :, -2:] = True
    good = ~(low | extreme | boundary)
    energy = squared.sum(axis=(1, 2))
    delta = np.sqrt(squared.mean(axis=(1, 2)))
    core = np.sqrt(np.sum(squared * good, axis=(1, 2)) / good.sum(axis=(1, 2)))
    return {
        'delta': delta, 'delta_core': core,
        'low_energy_change_share': np.sum(squared * low, axis=(1, 2)) / np.maximum(energy, 1e-30),
        'boundary_change_share': np.sum(squared * boundary, axis=(1, 2)) / np.maximum(energy, 1e-30),
        'extreme_clean_fraction': (np.abs(clean) > 10).mean(axis=(1, 2)),
        'extreme_aug_fraction': (np.abs(changed) > 10).mean(axis=(1, 2)),
    }
