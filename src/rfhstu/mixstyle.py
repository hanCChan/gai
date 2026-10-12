from __future__ import annotations

import torch
from torch import nn


class MixStyle(nn.Module):
    """MixStyle: randomize feature mean/std during training (domain style augmentation)."""

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.alpha <= 0:
            return x
        b = x.shape[0]
        mu = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        sig = (var + 1e-6).sqrt()
        perm = torch.randperm(b, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((b, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sig_mix = lam * sig + (1 - lam) * sig2
        return (x - mu) / sig * sig_mix + mu_mix
