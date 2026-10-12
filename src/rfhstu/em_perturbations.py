"""Electromagnetic / RF-channel perturbations for LoRa IQ signals.

All functions operate on IQ tensors shaped [B, 2, T] (I/Q channels, float32).
Perturbations are physically motivated and intended for robustness evaluation
and consistency-regularized training (thesis Chapter 5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class PerturbationSpec:
    """Single perturbation with a scalar strength parameter."""

    name: str
    strength: float
    apply_fn: Callable[[torch.Tensor, float, float, float], torch.Tensor]
    # sample_rate and lora_bandwidth passed through for CFO / filter ops


@dataclass
class EmPerturbConfig:
    """Composite EM perturbation profile."""

    awgn_snr_db: float | None = None
    narrowband_amplitude: float | None = None
    narrowband_freq_hz: float | None = None
    narrowband_sir_db: float | None = None
    cfo_hz: float | None = None
    cfo_norm: float | None = None
    phase_noise_std: float | None = None
    iq_imbalance_alpha: float | None = None
    iq_imbalance_beta: float | None = None
    iq_amp_db: float | None = None
    iq_phase_deg: float | None = None
    filter_tilt_db: float | None = None
    filter_tilt_norm: float | None = None
    sample_rate: float = 1_000_000.0
    lora_bandwidth: float = 125_000.0


def _to_complex(iq: torch.Tensor) -> torch.Tensor:
    return torch.complex(iq[:, 0].float(), iq[:, 1].float())


def _from_complex(z: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=1).to(dtype)


def apply_awgn(iq: torch.Tensor, snr_db: float, sample_rate: float, lora_bw: float) -> torch.Tensor:
    """Additive white Gaussian noise at target SNR (dB) relative to complex signal power.

    IQ layout is [B, 2, T] with I/Q channels. Signal power is mean(I^2 + Q^2).
    Noise is split equally across I and Q so total noise power = P_s / SNR.
    """
    del sample_rate, lora_bw
    if snr_db >= 100.0:
        return iq
    power = iq.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power_per_channel = power / (2.0 * snr_linear)
    noise = torch.randn_like(iq) * noise_power_per_channel.sqrt()
    return iq + noise


def apply_narrowband_interference(
    iq: torch.Tensor,
    amplitude: float,
    freq_hz: float,
    sample_rate: float,
    lora_bw: float,
) -> torch.Tensor:
    """Single-tone narrowband interferer: A cos(2π f_i t + φ)."""
    del lora_bw
    bsz, _, length = iq.shape
    device, dtype = iq.device, iq.dtype
    t = torch.arange(length, device=device, dtype=torch.float32) / sample_rate
    phi = torch.empty(bsz, 1, device=device).uniform_(0.0, 2.0 * math.pi)
    tone = amplitude * torch.cos(2.0 * math.pi * freq_hz * t.view(1, -1) + phi)
    return iq + tone.unsqueeze(1).to(dtype)


def apply_cfo(iq: torch.Tensor, delta_f_hz: float, sample_rate: float, lora_bw: float) -> torch.Tensor:
    """Carrier frequency offset: x'(t) = x(t) exp(j 2π Δf t)."""
    del lora_bw
    bsz, _, length = iq.shape
    device = iq.device
    t = torch.arange(length, device=device, dtype=torch.float32) / sample_rate
    phase = 2.0 * math.pi * delta_f_hz * t
    rot = torch.complex(torch.cos(phase), torch.sin(phase))
    z = _to_complex(iq)
    z = z * rot.view(1, -1).to(z.dtype)
    return _from_complex(z, iq.dtype)


def apply_phase_noise(
    iq: torch.Tensor,
    std_rad: float,
    sample_rate: float,
    lora_bw: float,
) -> torch.Tensor:
    """Random-walk phase noise: x'(t) = x(t) exp(j φ(t)), φ ~ cumsum N(0, σ)."""
    del sample_rate, lora_bw
    bsz, _, length = iq.shape
    device = iq.device
    increments = torch.randn(bsz, length, device=device) * std_rad
    phi = torch.cumsum(increments, dim=-1)
    rot = torch.complex(torch.cos(phi), torch.sin(phi))
    z = _to_complex(iq)
    z = z * rot.to(z.dtype)
    return _from_complex(z, iq.dtype)


def apply_iq_imbalance(
    iq: torch.Tensor,
    alpha: float,
    beta: float,
    sample_rate: float,
    lora_bw: float,
) -> torch.Tensor:
    """IQ imbalance model: I' = α I, Q' = β Q (gain mismatch on I/Q branches)."""
    del sample_rate, lora_bw
    out = iq.clone()
    out[:, 0] = alpha * out[:, 0]
    out[:, 1] = beta * out[:, 1]
    return out


def apply_narrowband_sir(
    iq: torch.Tensor,
    sir_db: float,
    sample_rate: float,
    lora_bw: float,
    freq_hz: float | None = None,
) -> torch.Tensor:
    """Narrowband interferer at target signal-to-interference ratio (dB)."""
    power = iq.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    sir_linear = 10.0 ** (sir_db / 10.0)
    intf_power = power / sir_linear
    amplitude = intf_power.sqrt()
    freq = freq_hz if freq_hz is not None else lora_bw * 0.35
    return apply_narrowband_interference(iq, float(amplitude.mean()), freq, sample_rate, lora_bw)


def apply_iq_imbalance_db_phase(
    iq: torch.Tensor,
    amp_db: float,
    phase_deg: float,
    sample_rate: float,
    lora_bw: float,
) -> torch.Tensor:
    """IQ imbalance with amplitude mismatch (dB) and phase mismatch (degrees)."""
    del sample_rate, lora_bw
    amp = 10.0 ** (amp_db / 20.0)
    phi = math.radians(phase_deg)
    i_ch = amp * iq[:, 0]
    q_ch = math.cos(phi) * iq[:, 1] + math.sin(phi) * iq[:, 0]
    return torch.stack([i_ch, q_ch], dim=1)


def apply_cfo_norm(iq: torch.Tensor, cfo_norm: float, sample_rate: float, lora_bw: float) -> torch.Tensor:
    """CFO as normalized offset: Δf = cfo_norm * LoRa bandwidth."""
    return apply_cfo(iq, cfo_norm * lora_bw, sample_rate, lora_bw)


def apply_filter_tilt_norm(iq: torch.Tensor, tilt_norm: float, sample_rate: float, lora_bw: float) -> torch.Tensor:
    """Filter drift with normalized tilt mapped to dB across band edges."""
    return apply_filter_drift(iq, tilt_norm * 10.0, sample_rate, lora_bw)


def apply_filter_drift(
    iq: torch.Tensor,
    tilt_db: float,
    sample_rate: float,
    lora_bw: float,
) -> torch.Tensor:
    """Linear-in-frequency gain drift (receiver filter / cable tilt proxy)."""
    bsz, _, length = iq.shape
    device = iq.device
    z = _to_complex(iq)
    spectrum = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1)
    freq = torch.fft.fftshift(torch.fft.fftfreq(length, d=1.0 / sample_rate)).to(device)
    fmax = freq.abs().max().clamp_min(1.0)
    gain = 10.0 ** ((tilt_db * (freq / fmax).view(1, -1)) / 20.0)
    spectrum = spectrum * gain.to(spectrum.dtype)
    z_out = torch.fft.ifft(torch.fft.ifftshift(spectrum, dim=-1), dim=-1)
    return _from_complex(z_out, iq.dtype)


def apply_em_perturbation(iq: torch.Tensor, cfg: EmPerturbConfig) -> torch.Tensor:
    """Apply all configured perturbations in sequence."""
    x = iq
    sr, bw = cfg.sample_rate, cfg.lora_bandwidth
    if cfg.awgn_snr_db is not None:
        x = apply_awgn(x, cfg.awgn_snr_db, sr, bw)
    if cfg.narrowband_sir_db is not None:
        x = apply_narrowband_sir(x, cfg.narrowband_sir_db, sr, bw, cfg.narrowband_freq_hz)
    if cfg.narrowband_amplitude is not None and cfg.narrowband_amplitude > 0:
        freq = cfg.narrowband_freq_hz if cfg.narrowband_freq_hz is not None else bw * 0.8
        x = apply_narrowband_interference(x, cfg.narrowband_amplitude, freq, sr, bw)
    if cfg.cfo_norm is not None and abs(cfg.cfo_norm) > 0:
        x = apply_cfo_norm(x, cfg.cfo_norm, sr, bw)
    if cfg.cfo_hz is not None and abs(cfg.cfo_hz) > 0:
        x = apply_cfo(x, cfg.cfo_hz, sr, bw)
    if cfg.phase_noise_std is not None and cfg.phase_noise_std > 0:
        x = apply_phase_noise(x, cfg.phase_noise_std, sr, bw)
    if cfg.iq_amp_db is not None and cfg.iq_phase_deg is not None:
        if abs(cfg.iq_amp_db) > 0 or abs(cfg.iq_phase_deg) > 0:
            x = apply_iq_imbalance_db_phase(x, cfg.iq_amp_db, cfg.iq_phase_deg, sr, bw)
    if cfg.iq_imbalance_alpha is not None and cfg.iq_imbalance_beta is not None:
        if abs(cfg.iq_imbalance_alpha - 1.0) > 1e-6 or abs(cfg.iq_imbalance_beta - 1.0) > 1e-6:
            x = apply_iq_imbalance(x, cfg.iq_imbalance_alpha, cfg.iq_imbalance_beta, sr, bw)
    if cfg.filter_tilt_norm is not None and abs(cfg.filter_tilt_norm) > 0:
        x = apply_filter_tilt_norm(x, cfg.filter_tilt_norm, sr, bw)
    if cfg.filter_tilt_db is not None and abs(cfg.filter_tilt_db) > 0:
        x = apply_filter_drift(x, cfg.filter_tilt_db, sr, bw)
    return x


def clean_config(sample_rate: float = 1e6, lora_bw: float = 125e3) -> EmPerturbConfig:
    """Empty perturbation profile (strict no-op)."""
    return EmPerturbConfig(sample_rate=sample_rate, lora_bandwidth=lora_bw)


# Preset sweep grids for robustness curves (thesis Chapter 5)
ROBUSTNESS_SWEEPS: dict[str, list[float]] = {
    "awgn_snr_db": [100.0, 40.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0],
    "narrowband_sir_db": [30.0, 20.0, 10.0, 5.0, 0.0],
    "cfo_norm": [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10],
    "cfo_hz": [0.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0],
    "phase_noise_std": [0.0, 0.01, 0.03, 0.05, 0.10],
    "narrowband_amplitude": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
    "iq_amp_db": [0.0, 1.0, 3.0, 5.0],
    "iq_phase_deg": [0.0, 2.0, 5.0, 10.0],
    "iq_imbalance_alpha": [1.0, 1.02, 1.05, 1.1, 1.15, 1.2],
    "filter_tilt_norm": [0.0, 0.1, 0.2, 0.4],
    "filter_tilt_db": [0.0, 1.0, 2.0, 4.0, 6.0, 8.0],
}

CFO_NORM_MAIN: list[float] = [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03]
CFO_NORM_SEVERE: list[float] = [0.05, 0.10]
AWGN_SNR_MAIN: list[float] = [100.0, 40.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0]

MIXED_STRESS_PRESETS: dict[str, EmPerturbConfig] = {
    "awgn_cfo": EmPerturbConfig(awgn_snr_db=10.0, cfo_norm=0.05),
    "awgn_nbi": EmPerturbConfig(awgn_snr_db=10.0, narrowband_sir_db=10.0),
    "cfo_iq": EmPerturbConfig(cfo_norm=0.05, iq_amp_db=3.0, iq_phase_deg=5.0),
}

PERTURBATION_PHYSICS: dict[str, str] = {
    "awgn_snr_db": "Thermal/environmental noise (AWGN).",
    "narrowband_sir_db": "Adjacent-channel or co-channel narrowband interference.",
    "cfo_norm": "Carrier frequency offset from oscillator / sync error (normalized to LoRa BW).",
    "phase_noise_std": "Oscillator phase jitter (random-walk phase noise).",
    "iq_amp_db": "I-branch gain mismatch in the receiver IQ chain.",
    "iq_phase_deg": "I/Q orthogonality error in the receiver front-end.",
    "filter_tilt_norm": "Receiver filter / cable spectral tilt across the observed band.",
}


def make_sweep_config(perturb_type: str, strength: float, sample_rate: float = 1e6, lora_bw: float = 125e3) -> EmPerturbConfig:
    """Build a single-parameter sweep config."""
    cfg = EmPerturbConfig(sample_rate=sample_rate, lora_bandwidth=lora_bw)
    if perturb_type == "awgn_snr_db":
        if strength >= 100.0:
            return cfg
        cfg.awgn_snr_db = strength
    elif perturb_type == "narrowband_sir_db":
        cfg.narrowband_sir_db = strength
    elif perturb_type == "cfo_norm":
        cfg.cfo_norm = strength
    elif perturb_type == "cfo_hz":
        cfg.cfo_hz = strength
    elif perturb_type == "phase_noise_std":
        cfg.phase_noise_std = strength
    elif perturb_type == "narrowband_amplitude":
        cfg.narrowband_amplitude = strength
        cfg.narrowband_freq_hz = lora_bw * 0.8
    elif perturb_type == "iq_amp_db":
        cfg.iq_amp_db = strength
        cfg.iq_phase_deg = 0.0
    elif perturb_type == "iq_phase_deg":
        cfg.iq_amp_db = 0.0
        cfg.iq_phase_deg = strength
    elif perturb_type == "iq_imbalance":
        cfg.iq_imbalance_alpha = strength
        cfg.iq_imbalance_beta = 1.0 / strength if strength != 0 else 1.0
    elif perturb_type == "filter_tilt_norm":
        cfg.filter_tilt_norm = strength
    elif perturb_type == "filter_tilt_db":
        cfg.filter_tilt_db = strength
    else:
        raise ValueError(f"Unknown perturb_type: {perturb_type}")
    return cfg


def sample_emcr_training_perturb_config(
    sample_rate: float = 1e6,
    lora_bw: float = 125e3,
    rng: torch.Generator | None = None,
) -> EmPerturbConfig:
    """Random moderate EM perturbation for EM-CR smoke/full (no severe ranges)."""
    g = rng or torch.Generator()
    choices = ["awgn_snr_db", "cfo_norm", "narrowband_sir_db"]
    pick = int(torch.randint(0, len(choices), (1,), generator=g).item())
    cfg = EmPerturbConfig(sample_rate=sample_rate, lora_bandwidth=lora_bw)
    if choices[pick] == "awgn_snr_db":
        snr = float(torch.randint(15, 31, (1,), generator=g).item())
        cfg.awgn_snr_db = snr
    elif choices[pick] == "cfo_norm":
        idx = int(torch.randint(0, 4, (1,), generator=g).item())
        cfg.cfo_norm = [0.001, 0.003, 0.005, 0.01][idx]
    else:
        sir = float(torch.randint(10, 31, (1,), generator=g).item())
        cfg.narrowband_sir_db = sir
    return cfg


def sample_emaug_training_perturb_config(
    sample_rate: float = 1e6,
    lora_bw: float = 125e3,
    rng: torch.Generator | None = None,
) -> EmPerturbConfig:
    """AWGN 20–30 dB or NBI 10–30 dB (no CFO)."""
    g = rng or torch.Generator()
    cfg = EmPerturbConfig(sample_rate=sample_rate, lora_bandwidth=lora_bw)
    if int(torch.randint(0, 2, (1,), generator=g).item()) == 0:
        cfg.awgn_snr_db = float(torch.randint(20, 31, (1,), generator=g).item())
    else:
        cfg.narrowband_sir_db = float(torch.randint(10, 31, (1,), generator=g).item())
    return cfg


def sample_weak_cfo_perturb_config(
    sample_rate: float = 1e6,
    lora_bw: float = 125e3,
    rng: torch.Generator | None = None,
) -> EmPerturbConfig:
    """AWGN / NBI / very weak CFO 0.0005–0.001."""
    g = rng or torch.Generator()
    pick = int(torch.randint(0, 3, (1,), generator=g).item())
    cfg = EmPerturbConfig(sample_rate=sample_rate, lora_bandwidth=lora_bw)
    if pick == 0:
        cfg.awgn_snr_db = float(torch.randint(20, 31, (1,), generator=g).item())
    elif pick == 1:
        cfg.narrowband_sir_db = float(torch.randint(10, 31, (1,), generator=g).item())
    else:
        cfg.cfo_norm = 0.0005 if int(torch.randint(0, 2, (1,), generator=g).item()) == 0 else 0.001
    return cfg


def sample_training_perturb_config(
    sample_rate: float = 1e6,
    lora_bw: float = 125e3,
    rng: torch.Generator | None = None,
) -> EmPerturbConfig:
    """Random moderate EM perturbation for EM-Aug / EM-CR training."""
    g = rng or torch.Generator()
    choices = ["awgn_snr_db", "cfo_norm", "phase_noise_std", "narrowband_sir_db", "iq_combo"]
    pick = int(torch.randint(0, len(choices), (1,), generator=g).item())
    cfg = EmPerturbConfig(sample_rate=sample_rate, lora_bandwidth=lora_bw)
    if choices[pick] == "awgn_snr_db":
        cfg.awgn_snr_db = float(torch.randint(5, 26, (1,), generator=g).item())
    elif choices[pick] == "cfo_norm":
        cfg.cfo_norm = float(torch.randint(1, 6, (1,), generator=g).item()) / 100.0
    elif choices[pick] == "phase_noise_std":
        cfg.phase_noise_std = float(torch.randint(1, 6, (1,), generator=g).item()) / 100.0
    elif choices[pick] == "narrowband_sir_db":
        cfg.narrowband_sir_db = float(torch.randint(5, 21, (1,), generator=g).item())
    else:
        cfg.iq_amp_db = float(torch.randint(0, 4, (1,), generator=g).item())
        cfg.iq_phase_deg = float(torch.randint(0, 6, (1,), generator=g).item())
    return cfg
