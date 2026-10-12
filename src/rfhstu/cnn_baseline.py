from __future__ import annotations

import torch
from torch import nn


CNN_INPUT_TYPES = ("iq", "fft", "fft_inband", "fft_oob", "amp_phase")


def _unit_energy(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    power = x.square().sum(dim=(-2, -1), keepdim=True)
    return x / power.clamp_min(eps).sqrt()


def _fft_ri(iq: torch.Tensor, sample_rate: float, lora_bandwidth: float, inband_only: bool) -> torch.Tensor:
    z = torch.complex(iq[:, 0], iq[:, 1])
    spec = torch.fft.fftshift(torch.fft.fft(z, dim=-1, norm="ortho"), dim=-1)
    if inband_only:
        freqs = torch.fft.fftshift(torch.fft.fftfreq(iq.shape[-1], d=1.0 / sample_rate)).to(iq.device)
        mask = freqs.abs() <= (lora_bandwidth / 2.0)
        spec = spec * mask.unsqueeze(0)
    out = torch.stack([spec.real, spec.imag], dim=1)
    return _unit_energy(out)


def build_cnn_input(
    iq: torch.Tensor,
    input_type: str,
    sample_rate: float = 1_000_000.0,
    lora_bandwidth: float = 125_000.0,
) -> torch.Tensor:
    """Build OSU CNN baseline inputs from normalized IQ windows."""

    if input_type not in CNN_INPUT_TYPES:
        raise ValueError(f"Unknown cnn_input_type={input_type!r}. Expected one of {CNN_INPUT_TYPES}.")
    if input_type == "iq":
        return iq
    if input_type in {"fft", "fft_oob"}:
        return _fft_ri(iq, sample_rate=sample_rate, lora_bandwidth=lora_bandwidth, inband_only=False)
    if input_type == "fft_inband":
        return _fft_ri(iq, sample_rate=sample_rate, lora_bandwidth=lora_bandwidth, inband_only=True)
    amp = torch.sqrt(iq[:, 0].square() + iq[:, 1].square()).unsqueeze(1)
    phase = torch.atan2(iq[:, 1], iq[:, 0]).unsqueeze(1)
    return _unit_energy(torch.cat([amp, phase], dim=1))


class OSUCNNBaseline(nn.Module):
    """PyTorch reproduction of the OSU LoRa CNN baseline path."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 2,
        hidden_dim: int = 25,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=4, padding="same"),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=4, padding="same"),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=4, padding="same"),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=4, padding="same"),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, hidden_dim)
        self.act = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.domain_head = None

    def forward(
        self,
        x: torch.Tensor,
        adv_lambda: float = 1.0,
        return_features: bool = False,
        return_supcon_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        del adv_lambda
        h = self.features(x).flatten(1)
        embedding = self.act(self.fc(h))
        logits = self.classifier(self.dropout(embedding))
        out = {"embedding": embedding, "logits": logits}
        if return_features:
            out["features"] = embedding
        if return_supcon_features:
            out["supcon_features"] = embedding
        return out
