from __future__ import annotations

import torch
from torch import nn


class OOBCrossAttentionFusion(nn.Module):
    """Single-layer OOB-guided cross-attention fusion.

    Main RF tokens query OOB tokens. OOB tokens do not attend back to main tokens.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_main = nn.LayerNorm(dim)
        self.norm_oob = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(dim * 2, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x_main: torch.Tensor, x_oob: torch.Tensor) -> torch.Tensor:
        main = self.norm_main(x_main)
        oob = self.norm_oob(x_oob)
        attended, _ = self.attn(query=main, key=oob, value=oob, need_weights=False)
        gate = torch.sigmoid(self.gate(torch.cat([x_main, attended], dim=-1)))
        fused = x_main + gate * self.out_proj(attended)
        return self.norm_out(fused)

