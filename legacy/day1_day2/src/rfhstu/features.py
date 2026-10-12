from __future__ import annotations

import math

import numpy as np
import torch


def complex_iq_to_channels(iq: np.ndarray) -> np.ndarray:
    """Convert complex IQ samples to [2, T] float32 channels."""
    return np.stack([iq.real, iq.imag], axis=0).astype(np.float32, copy=False)


def normalize_iq(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    power = np.sqrt(np.mean(x[0] ** 2 + x[1] ** 2) + eps)
    return (x / power).astype(np.float32, copy=False)


def torch_rf_views(
    iq: torch.Tensor,
    sample_rate: float = 1_000_000.0,
    lora_bandwidth: float = 125_000.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build IQ, FFT, OOB, and amplitude/phase views.

    Args:
        iq: Tensor shaped [B, 2, T].

    Returns:
        Tuple of tensors:
            iq view [B, 2, T]
            fft magnitude view [B, 1, T]
            oob magnitude view [B, 1, T]
            amplitude/phase view [B, 2, T]
    """
    complex_x = torch.complex(iq[:, 0], iq[:, 1])
    spectrum = torch.fft.fftshift(torch.fft.fft(complex_x, dim=-1), dim=-1)
    mag = torch.log1p(torch.abs(spectrum)).unsqueeze(1)

    freq = torch.fft.fftshift(torch.fft.fftfreq(iq.shape[-1], d=1.0 / sample_rate)).to(iq.device)
    in_band = torch.abs(freq) <= (lora_bandwidth / 2.0)
    oob = mag * (~in_band).to(mag.dtype).view(1, 1, -1)

    amp = torch.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2 + eps)
    phase = torch.atan2(iq[:, 1], iq[:, 0]) / math.pi
    amp_phase = torch.stack([torch.log1p(amp), phase], dim=1)

    fft_mean = mag.mean(dim=-1, keepdim=True)
    fft_std = mag.std(dim=-1, keepdim=True).clamp_min(eps)
    mag = (mag - fft_mean) / fft_std
    oob = (oob - fft_mean) / fft_std
    return iq, mag, oob, amp_phase


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert [B, C, T] to [B, P, C * patch_size]."""
    bsz, channels, length = x.shape
    if length % patch_size != 0:
        raise ValueError(f"length {length} is not divisible by patch_size {patch_size}")
    patches = length // patch_size
    return x.view(bsz, channels, patches, patch_size).permute(0, 2, 1, 3).reshape(bsz, patches, channels * patch_size)


def build_patch_features(
    iq: torch.Tensor,
    patch_size: int,
    sample_rate: float = 1_000_000.0,
    lora_bandwidth: float = 125_000.0,
    use_oob: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build fused RF patch features from IQ-derived views."""
    iq_view, fft_view, oob_view, amp_phase = torch_rf_views(iq, sample_rate, lora_bandwidth)
    views = {
        "iq": patchify(iq_view, patch_size),
        "fft": patchify(fft_view, patch_size),
        "amp_phase": patchify(amp_phase, patch_size),
    }
    if use_oob:
        views["oob"] = patchify(oob_view, patch_size)
    fused = torch.cat(list(views.values()), dim=-1)
    return fused, views

