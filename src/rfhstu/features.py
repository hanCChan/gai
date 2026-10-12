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


FFT_NORM_CHOICES = ("none", "log_zscore")
OOB_NORM_CHOICES = ("none", "ratio", "log_ratio", "zscore")


def torch_rf_views(
    iq: torch.Tensor,
    sample_rate: float = 1_000_000.0,
    lora_bandwidth: float = 125_000.0,
    eps: float = 1e-6,
    fft_norm: str = "log_zscore",
    oob_norm: str = "zscore",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build IQ, FFT, OOB, and amplitude/phase views.

    Args:
        iq: Tensor shaped [B, 2, T]. (IQ-level normalization, if any, is applied
            earlier at the dataset stage via ``input_norm``.)
        fft_norm: FFT-magnitude normalization. ``log_zscore`` (legacy) = log1p then
            per-window z-score; ``none`` = log1p only (no standardization).
        oob_norm: OOB feature normalization. ``zscore`` (legacy) = masked log1p then
            per-window z-score (shares the FFT z-score stats); ``none`` = masked log1p
            only; ``ratio`` = linear |spectrum| at OOB bins divided by the in-band RMS
            magnitude (removes receiver gain); ``log_ratio`` = log|spectrum| - log(inband RMS).

    Returns:
        Tuple of tensors:
            iq view [B, 2, T]
            fft magnitude view [B, 1, T]
            oob magnitude view [B, 1, T]
            amplitude/phase view [B, 2, T]
    """
    if fft_norm not in FFT_NORM_CHOICES:
        raise ValueError(f"Unknown fft_norm={fft_norm!r}; expected one of {FFT_NORM_CHOICES}")
    if oob_norm not in OOB_NORM_CHOICES:
        raise ValueError(f"Unknown oob_norm={oob_norm!r}; expected one of {OOB_NORM_CHOICES}")

    complex_x = torch.complex(iq[:, 0], iq[:, 1])
    spectrum = torch.fft.fftshift(torch.fft.fft(complex_x, dim=-1), dim=-1)
    abs_spec = torch.abs(spectrum).unsqueeze(1)  # linear magnitude [B, 1, T]
    mag = torch.log1p(abs_spec)  # log magnitude [B, 1, T]

    freq = torch.fft.fftshift(torch.fft.fftfreq(iq.shape[-1], d=1.0 / sample_rate)).to(iq.device)
    in_band = torch.abs(freq) <= (lora_bandwidth / 2.0)
    oob_mask = (~in_band).to(mag.dtype).view(1, 1, -1)
    inband_mask = in_band.to(mag.dtype).view(1, 1, -1)

    amp = torch.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2 + eps)
    phase = torch.atan2(iq[:, 1], iq[:, 0]) / math.pi
    amp_phase = torch.stack([torch.log1p(amp), phase], dim=1)

    # per-window FFT z-score stats (computed on log magnitude, legacy definition)
    fft_mean = mag.mean(dim=-1, keepdim=True)
    fft_std = mag.std(dim=-1, keepdim=True).clamp_min(eps)

    if fft_norm == "log_zscore":
        fft_view = (mag - fft_mean) / fft_std
    else:  # "none"
        fft_view = mag

    if oob_norm == "zscore":
        oob_view = (mag * oob_mask - fft_mean) / fft_std
    elif oob_norm == "none":
        oob_view = mag * oob_mask
    else:
        # in-band RMS magnitude per window (proxy for receiver gain); CFO/peak info kept
        inband_power = (abs_spec ** 2) * inband_mask
        n_inband = inband_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        inband_rms = (inband_power.sum(dim=-1, keepdim=True) / n_inband).clamp_min(eps).sqrt()
        if oob_norm == "ratio":
            oob_view = (abs_spec / inband_rms) * oob_mask
        else:  # "log_ratio"
            oob_view = (torch.log(abs_spec + eps) - torch.log(inband_rms + eps)) * oob_mask

    return iq, fft_view, oob_view, amp_phase


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
    fft_norm: str = "log_zscore",
    oob_norm: str = "zscore",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build fused RF patch features from IQ-derived views."""
    iq_view, fft_view, oob_view, amp_phase = torch_rf_views(
        iq, sample_rate, lora_bandwidth, fft_norm=fft_norm, oob_norm=oob_norm
    )
    views = {
        "iq": patchify(iq_view, patch_size),
        "fft": patchify(fft_view, patch_size),
        "amp_phase": patchify(amp_phase, patch_size),
    }
    if use_oob:
        views["oob"] = patchify(oob_view, patch_size)
    fused = torch.cat(list(views.values()), dim=-1)
    return fused, views

