from __future__ import annotations

import torch
from torch import nn


def parse_ratios(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, str):
        ratios = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        ratios = [int(item) for item in value]
    if not ratios:
        raise ValueError("multiscale ratios must not be empty")
    if any(ratio < 1 for ratio in ratios):
        raise ValueError(f"multiscale ratios must be >= 1, got {ratios}")
    return ratios


def merge_tokens(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """Merge adjacent tokens by mean pooling.

    If the sequence length is not divisible by ratio, the tail is truncated.
    This keeps the first implementation simple and deterministic.
    """
    if ratio == 1:
        return x
    length = x.shape[1]
    usable = (length // ratio) * ratio
    if usable == 0:
        raise ValueError(f"Cannot merge sequence length {length} with ratio {ratio}")
    x = x[:, :usable, :]
    return x.reshape(x.shape[0], usable // ratio, ratio, x.shape[-1]).mean(dim=2)


class MultiScaleTokenFusion(nn.Module):
    def __init__(
        self,
        encoders: list[nn.Module],
        ratios: list[int],
        dim: int,
        fusion_type: str = "concat_pool",
    ) -> None:
        super().__init__()
        if fusion_type != "concat_pool":
            raise ValueError(f"Only concat_pool is supported in the first multi-scale implementation, got {fusion_type}")
        if len(encoders) != len(ratios):
            raise ValueError("encoders and ratios must have the same length")
        self.encoders = nn.ModuleList(encoders)
        self.ratios = ratios
        self.fusion_type = fusion_type
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(dim * len(ratios)),
            nn.Linear(dim * len(ratios), dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = []
        for ratio, encoder in zip(self.ratios, self.encoders):
            tokens = merge_tokens(x, ratio)
            encoded = encoder(tokens)
            pooled.append(encoded.mean(dim=1))
        return self.fusion_mlp(torch.cat(pooled, dim=-1))

