from __future__ import annotations

import torch
from torch import nn


def parse_kernels(value: str | list[int] | tuple[int, ...]) -> tuple[int, int]:
    if isinstance(value, str):
        parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        parts = [int(part) for part in value]
    if len(parts) != 2:
        raise ValueError(f"cnn_stem_kernels must contain exactly two kernel sizes, got {value!r}")
    return parts[0], parts[1]


class CNNStemPatchEmbed(nn.Module):
    """Local convolutional RF front-end followed by patch tokenization."""

    def __init__(
        self,
        in_channels: int,
        dim: int,
        patch_size: int = 256,
        stem_dim: int = 32,
        kernels: str | list[int] | tuple[int, int] = "7,5",
    ) -> None:
        super().__init__()
        k1, k2 = parse_kernels(kernels)
        self.patch_size = patch_size
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_dim, kernel_size=k1, stride=1, padding=k1 // 2),
            nn.BatchNorm1d(stem_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(stem_dim, stem_dim, kernel_size=k2, stride=1, padding=k2 // 2),
            nn.BatchNorm1d(stem_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.tokenize = nn.Conv1d(stem_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x_main: torch.Tensor) -> torch.Tensor:
        x = self.stem(x_main)
        tokens = self.tokenize(x)
        return tokens.transpose(1, 2)
