from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_prototypes(embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    labels_unique = torch.sort(torch.unique(labels)).values
    protos = []
    for label in labels_unique:
        proto = embeddings[labels == label].mean(dim=0)
        protos.append(proto)
    return F.normalize(torch.stack(protos), dim=-1), labels_unique


@torch.no_grad()
def prototype_predict(embeddings: torch.Tensor, prototypes: torch.Tensor, prototype_labels: torch.Tensor) -> torch.Tensor:
    sims = F.normalize(embeddings, dim=-1) @ prototypes.T
    return prototype_labels[sims.argmax(dim=-1)]

