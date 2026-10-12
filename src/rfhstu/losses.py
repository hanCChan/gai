from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    batch = labels.shape[0]
    eye = torch.eye(batch, device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positive.any(dim=1)
    if not valid.any():
        return z.new_zeros(())
    exp_logits = torch.exp(logits) * (~eye).to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(eps))
    mean_log_prob = (positive.to(log_prob.dtype) * log_prob).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return -mean_log_prob[valid].mean()


class SupConLoss(nn.Module):
    """Supervised contrastive loss over one embedding per sample."""

    def __init__(self, temperature: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=-1)
        logits = features @ features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        batch = labels.shape[0]
        eye = torch.eye(batch, device=features.device, dtype=torch.bool)
        positive = labels[:, None].eq(labels[None, :]) & ~eye
        valid = positive.any(dim=1)
        if not valid.any():
            return features.new_zeros(())
        exp_logits = torch.exp(logits) * (~eye).to(logits.dtype)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))
        mean_log_prob = (positive.to(log_prob.dtype) * log_prob).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        return -mean_log_prob[valid].mean()


class CenterLoss(nn.Module):
    """Class-center compactness loss over classifier embeddings."""

    def __init__(self, num_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.zeros(num_classes, feat_dim))
        nn.init.normal_(self.centers, std=0.02)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers = self.centers[labels]
        return (features - centers).square().sum(dim=-1).mean()


def reconstruction_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    mask: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    weights = weights or {"iq": 1.0, "fft": 1.0, "oob": 1.0, "amp_phase": 0.25}
    losses = []
    for name, pred_value in pred.items():
        if name not in target or name not in weights:
            continue
        target_value = target[name].detach()
        patch_loss = F.mse_loss(pred_value, target_value, reduction="none").mean(dim=-1)
        masked = patch_loss[mask]
        if masked.numel() == 0:
            masked = patch_loss.reshape(-1)
        losses.append(weights[name] * masked.mean())
    if not losses:
        raise ValueError("No matching reconstruction heads and targets.")
    return torch.stack(losses).sum()


class DomainAdversarialHead(nn.Module):
    def __init__(self, dim: int, domain_sizes: dict[str, int], hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.fields = list(domain_sizes.keys())
        self.heads = nn.ModuleDict()
        for field, size in domain_sizes.items():
            self.heads[field] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, max(1, size)),
            )

    def forward(self, z: torch.Tensor, lambd: float = 1.0) -> dict[str, torch.Tensor]:
        rev = grad_reverse(z, lambd)
        return {field: head(rev) for field, head in self.heads.items()}

    def loss(self, logits: dict[str, torch.Tensor], domains: torch.Tensor, field_to_col: dict[str, int]) -> torch.Tensor:
        losses = []
        for field, pred in logits.items():
            target = domains[:, field_to_col[field]]
            if pred.shape[-1] <= 1:
                continue
            losses.append(F.cross_entropy(pred, target))
        if not losses:
            return domains.new_zeros((), dtype=torch.float32)
        return torch.stack(losses).mean()


def coral_loss(source: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Deep CORAL: align second-order statistics between source and target embeddings."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("coral_loss expects 2-D feature tensors [B, D].")
    d = source.shape[1]
    ns, nt = source.shape[0], target.shape[0]
    if ns < 2 or nt < 2:
        return source.new_zeros(())
    source_c = source - source.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    cov_s = (source_c.T @ source_c) / max(1, ns - 1)
    cov_t = (target_c.T @ target_c) / max(1, nt - 1)
    diff = cov_s - cov_t
    return diff.square().sum() / max(eps, 4.0 * d * d)


def information_maximization_loss(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """IM loss on unlabeled target: minimize conditional entropy, maximize marginal entropy."""
    probs = torch.softmax(logits, dim=-1)
    cond_entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1).mean()
    mean_prob = probs.mean(dim=0)
    marginal_entropy = -(mean_prob * torch.log(mean_prob.clamp_min(eps))).sum()
    return cond_entropy - marginal_entropy


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, weight=weight, label_smoothing=label_smoothing, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=-1)
    tp = torch.zeros(num_classes, device=logits.device)
    fp = torch.zeros(num_classes, device=logits.device)
    fn = torch.zeros(num_classes, device=logits.device)
    for c in range(num_classes):
        tp[c] = ((pred == c) & (labels == c)).sum()
        fp[c] = ((pred == c) & (labels != c)).sum()
        fn[c] = ((pred != c) & (labels == c)).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    support = tp + fn
    if support.sum() <= 0:
        return 0.0
    return float((f1 * support).sum() / support.sum().clamp_min(1.0))
