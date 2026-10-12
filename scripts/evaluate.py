from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfhstu.cnn_baseline import OSUCNNBaseline, build_cnn_input
from rfhstu.data import DOMAIN_FIELDS, SigMFIQDataset, load_manifest
from rfhstu.features import patchify, torch_rf_views
from rfhstu.models import DeviceClassifier, RFPatchEmbedder
from rfhstu.prototype import build_prototypes
from rfhstu.train_utils import add_common_args, format_metrics, load_checkpoint, make_datasets, make_loader, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RF-HSTU classifier or prototype head.")
    add_common_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prototype", action="store_true")
    parser.add_argument("--mode", choices=["classifier", "prototype"], default=None)
    parser.add_argument("--file-vote-mode", choices=["mean_logits", "mean_prob", "confidence_weighted"], default="mean_logits")
    parser.add_argument("--score-fusion", action="store_true")
    parser.add_argument("--fusion-alpha", type=float, default=0.5)
    parser.add_argument("--tta-mode", choices=["none", "bn_adapt", "tent"], default="none")
    parser.add_argument("--tent-steps", type=int, default=1)
    parser.add_argument("--tent-lr", type=float, default=1e-4)
    parser.add_argument("--tent-episodic", action="store_true")
    parser.add_argument("--adapt-mode", choices=["none", "bn_adapt", "entropy_min", "pseudo_proto", "oob_cfo_pseudo_proto"], default=None)
    parser.add_argument("--adapt-steps", type=int, default=None)
    parser.add_argument("--adapt-lr", type=float, default=None)
    parser.add_argument("--pseudo-threshold", type=float, default=0.8)
    parser.add_argument("--pseudo-topk-per-class", type=int, default=128)
    parser.add_argument("--pseudo-min-per-class", type=int, default=16)
    parser.add_argument("--prototype-momentum", type=float, default=0.5)
    parser.add_argument("--adapt-batch-size", type=int, default=None)
    parser.add_argument("--cfo-max-z", type=float, default=2.0)
    parser.add_argument("--oob-sim-threshold", type=float, default=0.3)
    parser.add_argument("--pseudo-balance-topk", type=int, default=128)
    parser.add_argument("--pseudo-require-cls-proto-agree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cfo-consistency-weight", type=float, default=1.0)
    parser.add_argument("--oob-consistency-weight", type=float, default=1.0)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def build_model(args: argparse.Namespace, ckpt: dict, device: torch.device) -> torch.nn.Module:
    ckpt_args = ckpt.get("args", {})
    model_type = ckpt.get("model_type", ckpt_args.get("model_type", args.model_type))
    num_classes = ckpt.get("num_classes")
    if num_classes is None:
        num_classes = ckpt_args.get("num_classes", 25)
    if model_type == "osu_cnn":
        model = OSUCNNBaseline(
            num_classes=int(num_classes),
            in_channels=2,
            hidden_dim=int(ckpt_args.get("cnn_hidden_dim", args.cnn_hidden_dim)),
            dropout=float(ckpt_args.get("cnn_dropout", args.cnn_dropout)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        return model
    dim = ckpt_args.get("dim", args.dim)
    depth = ckpt_args.get("depth", args.depth)
    dropout = ckpt_args.get("dropout", args.dropout)
    window_size = ckpt_args.get("window_size", args.window_size)
    patch_size = ckpt_args.get("patch_size", args.patch_size)
    sample_rate = ckpt_args.get("sample_rate", args.sample_rate)
    lora_bandwidth = ckpt_args.get("lora_bandwidth", args.lora_bandwidth)
    spreading_factor = ckpt_args.get("spreading_factor", args.spreading_factor)
    use_chirp_embedding = ckpt_args.get("use_chirp_embedding", args.use_chirp_embedding)
    oob_fusion_type = ckpt_args.get("oob_fusion_type", args.oob_fusion_type)
    use_oob_cross_attention = ckpt_args.get("use_oob_cross_attention", args.use_oob_cross_attention)
    oob_num_heads = ckpt_args.get("oob_num_heads", args.oob_num_heads)
    use_multiscale = ckpt_args.get("use_multiscale", args.use_multiscale)
    multiscale_ratios = ckpt_args.get("multiscale_ratios", args.multiscale_ratios)
    multiscale_fusion_type = ckpt_args.get("multiscale_fusion_type", args.multiscale_fusion_type)
    use_cfo_feature = ckpt_args.get("use_cfo_feature", args.use_cfo_feature)
    cfo_feature_type = ckpt_args.get("cfo_feature_type", args.cfo_feature_type)
    cfo_feature_norm = ckpt_args.get("cfo_feature_norm", args.cfo_feature_norm)
    oob_dropout = float(ckpt_args.get("oob_dropout", getattr(args, "oob_dropout", 0.0)))
    mixstyle = bool(ckpt_args.get("mixstyle", getattr(args, "mixstyle", False)))
    mixstyle_alpha = float(ckpt_args.get("mixstyle_alpha", getattr(args, "mixstyle_alpha", 0.1)))
    patch_embed_type = ckpt_args.get("patch_embed_type", args.patch_embed_type)
    cnn_stem_dim = ckpt_args.get("cnn_stem_dim", args.cnn_stem_dim)
    cnn_stem_kernels = ckpt_args.get("cnn_stem_kernels", args.cnn_stem_kernels)
    use_oob = not ckpt_args.get("no_oob", args.no_oob)
    embedder = RFPatchEmbedder(
        window_size=window_size,
        patch_size=patch_size,
        sample_rate=sample_rate,
        lora_bandwidth=lora_bandwidth,
        spreading_factor=spreading_factor,
        use_oob=use_oob and oob_fusion_type != "no_oob",
        oob_fusion_type=oob_fusion_type,
        use_oob_cross_attention=use_oob_cross_attention,
        patch_embed_type=patch_embed_type,
        dim=dim,
        cnn_stem_dim=cnn_stem_dim,
        cnn_stem_kernels=cnn_stem_kernels,
        fft_norm=args.fft_norm,
        oob_norm=args.oob_norm,
    )
    model = DeviceClassifier(
        embedder,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        dropout=dropout,
        domain_sizes=ckpt.get("domain_sizes"),
        use_chirp_embedding=use_chirp_embedding,
        oob_num_heads=oob_num_heads,
        use_multiscale=use_multiscale,
        multiscale_ratios=multiscale_ratios,
        multiscale_fusion_type=multiscale_fusion_type,
        use_cfo_feature=use_cfo_feature,
        cfo_feature_type=cfo_feature_type,
        cfo_feature_norm=cfo_feature_norm,
        oob_dropout=oob_dropout,
        mixstyle=mixstyle,
        mixstyle_alpha=mixstyle_alpha,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def prepare_model_input(iq: torch.Tensor, args: argparse.Namespace, ckpt_args: dict[str, Any]) -> torch.Tensor:
    model_type = ckpt_args.get("model_type", args.model_type)
    if model_type != "osu_cnn":
        return iq
    return build_cnn_input(
        iq,
        input_type=ckpt_args.get("cnn_input_type", args.cnn_input_type),
        sample_rate=ckpt_args.get("sample_rate", args.sample_rate),
        lora_bandwidth=ckpt_args.get("lora_bandwidth", args.lora_bandwidth),
    )


def resolve_mode(args: argparse.Namespace) -> str:
    if args.mode is not None:
        return args.mode
    return "prototype" if args.prototype else "classifier"


def default_out_dir(args: argparse.Namespace, mode: str) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    stem = Path(args.checkpoint).parent.name or Path(args.checkpoint).stem
    return Path("outputs") / "eval" / f"{stem}_{mode}"


def macro_f1(labels: list[int], preds: list[int], num_classes: int) -> float:
    scores = []
    for label in range(num_classes):
        tp = sum(1 for y, p in zip(labels, preds) if y == label and p == label)
        fp = sum(1 for y, p in zip(labels, preds) if y != label and p == label)
        fn = sum(1 for y, p in zip(labels, preds) if y == label and p != label)
        if tp + fp + fn == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / max(1, len(scores))


def confusion_matrix(labels: list[int], preds: list[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, pred in zip(labels, preds):
        if 0 <= label < num_classes and 0 <= pred < num_classes:
            matrix[label][pred] += 1
    return matrix


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_confusion(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", *[f"pred_{idx}" for idx in range(len(matrix))]])
        for idx, row in enumerate(matrix):
            writer.writerow([idx, *row])


def set_bn_train_only(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()


def count_bn_modules(model: torch.nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)))


def collect_bn_affine_params(model: torch.nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.affine:
                if module.weight is not None:
                    params.append(module.weight)
                if module.bias is not None:
                    params.append(module.bias)
    return params


def configure_tta(model: torch.nn.Module, args: argparse.Namespace):
    adapt_mode = resolve_adapt_mode(args)
    if adapt_mode in {"none", "pseudo_proto", "oob_cfo_pseudo_proto"}:
        model.eval()
        return adapt_mode, None, None
    if adapt_mode == "bn_adapt":
        return "bn_adapt", None, None
    for param in model.parameters():
        param.requires_grad_(False)
    bn_params = collect_bn_affine_params(model)
    if not bn_params:
        print("warning: entropy_min requested but no BatchNorm affine parameters were found; falling back to no adaptation.")
        model.eval()
        return "none", None, None
    for param in bn_params:
        param.requires_grad_(True)
    set_bn_train_only(model)
    optimizer = torch.optim.Adam(bn_params, lr=resolve_adapt_lr(args))
    episodic_state = copy.deepcopy(model.state_dict()) if args.tent_episodic else None
    return "entropy_min", optimizer, episodic_state


def resolve_adapt_mode(args: argparse.Namespace) -> str:
    if args.adapt_mode is not None:
        return args.adapt_mode
    if args.tta_mode == "tent":
        return "entropy_min"
    return args.tta_mode


def resolve_adapt_steps(args: argparse.Namespace) -> int:
    if args.adapt_steps is not None:
        return args.adapt_steps
    return args.tent_steps


def resolve_adapt_lr(args: argparse.Namespace) -> float:
    if args.adapt_lr is not None:
        return args.adapt_lr
    return args.tent_lr


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()


def restore_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor] | None) -> None:
    if state is not None:
        model.load_state_dict(state, strict=True)


def tent_adapt_batch(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    if optimizer is None:
        return
    for _ in range(max(1, resolve_adapt_steps(args))):
        optimizer.zero_grad(set_to_none=True)
        out = model(model_input)
        loss = entropy_loss(out["logits"])
        loss.backward()
        optimizer.step()


@torch.no_grad()
def bn_adapt_model(model: torch.nn.Module, loader, device: torch.device, args: argparse.Namespace, ckpt_args: dict[str, Any]) -> int:
    num_bn = count_bn_modules(model)
    if num_bn == 0:
        print("warning: bn_adapt requested but no BatchNorm modules were found; skipping BN adaptation.")
        model.eval()
        return 0
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    steps = max(1, resolve_adapt_steps(args))
    for _ in range(steps):
        for batch in tqdm(loader, leave=False, desc="bn_adapt"):
            iq = batch["iq"].to(device)
            _ = model(prepare_model_input(iq, args, ckpt_args))
    model.eval()
    return num_bn


@torch.no_grad()
def collect_train_embeddings(model: torch.nn.Module, loader, device: torch.device, args: argparse.Namespace, ckpt_args: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = []
    labels = []
    for batch in tqdm(loader, leave=False, desc="prototype_train"):
        iq = batch["iq"].to(device)
        out = model(prepare_model_input(iq, args, ckpt_args))
        embeddings.append(out["embedding"].cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(embeddings), torch.cat(labels)


@torch.no_grad()
def collect_unlabeled_target_embeddings(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    ckpt_args: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = []
    preds = []
    confidences = []
    model.eval()
    for batch in tqdm(loader, leave=False, desc="pseudo_target"):
        iq = batch["iq"].to(device)
        out = model(prepare_model_input(iq, args, ckpt_args))
        probs = F.softmax(out["logits"], dim=-1)
        conf, pred = probs.max(dim=-1)
        embeddings.append(F.normalize(out["embedding"].detach().cpu(), dim=-1))
        preds.append(pred.detach().cpu())
        confidences.append(conf.detach().cpu())
    return torch.cat(embeddings), torch.cat(preds), torch.cat(confidences)


def pseudo_proto_adapt(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    ckpt_args: dict[str, Any],
    source_prototypes: torch.Tensor,
    prototype_labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target_z, pseudo_y, confidence = collect_unlabeled_target_embeddings(model, loader, device, args, ckpt_args)
    selected_by_class: dict[int, torch.Tensor] = {}
    distribution: dict[str, int] = {}
    updated_labels = set()
    mixed = source_prototypes.clone()
    label_to_proto = {int(label.item()): idx for idx, label in enumerate(prototype_labels)}
    for label in prototype_labels.tolist():
        label = int(label)
        class_idx = torch.nonzero((pseudo_y == label) & (confidence >= args.pseudo_threshold), as_tuple=False).flatten()
        if class_idx.numel() > 0:
            order = torch.argsort(confidence[class_idx], descending=True)
            if args.pseudo_topk_per_class > 0:
                order = order[: args.pseudo_topk_per_class]
            class_idx = class_idx[order]
        distribution[str(label)] = int(class_idx.numel())
        if class_idx.numel() < args.pseudo_min_per_class:
            continue
        target_proto = F.normalize(target_z[class_idx].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        proto_idx = label_to_proto[label]
        mixed_proto = args.prototype_momentum * source_prototypes[proto_idx] + (1.0 - args.prototype_momentum) * target_proto
        mixed[proto_idx] = F.normalize(mixed_proto, dim=-1)
        selected_by_class[label] = class_idx
        updated_labels.add(label)
    stats = {
        "num_pseudo_selected": int(sum(distribution.values())),
        "num_classes_updated": int(len(updated_labels)),
        "pseudo_class_distribution": json.dumps(distribution, sort_keys=True),
    }
    return mixed, stats


def empty_pseudo_stats() -> dict[str, Any]:
    return {
        "num_pseudo_selected": "",
        "num_classes_updated": "",
        "pseudo_class_distribution": "",
        "num_rejected_by_confidence": "",
        "num_rejected_by_cls_proto_disagree": "",
        "num_rejected_by_cfo": "",
        "num_rejected_by_oob": "",
        "mean_cfo_z_selected": "",
        "mean_oob_sim_selected": "",
    }


def compute_cfo_features(model: torch.nn.Module, iq: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "_compute_cfo_features"):
        raise RuntimeError("oob_cfo_pseudo_proto requires an RF-HSTU/Hybrid model with CFO feature support.")
    return model._compute_cfo_features(iq)  # type: ignore[attr-defined]


def compute_oob_ratio_embedding(model: torch.nn.Module, iq: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    embedder = getattr(model, "embedder", None)
    if embedder is None:
        raise RuntimeError("oob_cfo_pseudo_proto requires an RF-HSTU/Hybrid model with an embedder.")
    _, _, oob_view, _ = torch_rf_views(
        iq,
        sample_rate=float(getattr(embedder, "sample_rate", args.sample_rate)),
        lora_bandwidth=float(getattr(embedder, "lora_bandwidth", args.lora_bandwidth)),
        fft_norm="log_zscore",
        oob_norm="ratio",
    )
    patch_size = int(getattr(embedder, "patch_size", args.patch_size))
    oob_tokens = patchify(oob_view, patch_size)
    return F.normalize(oob_tokens.mean(dim=1), dim=-1)


@torch.no_grad()
def collect_rf_adapt_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    ckpt_args: dict[str, Any],
    include_labels: bool,
) -> dict[str, torch.Tensor]:
    embeddings = []
    logits = []
    cfo = []
    oob = []
    labels = []
    model.eval()
    for batch in tqdm(loader, leave=False, desc="rf_adapt_features"):
        iq = batch["iq"].to(device)
        out = model(prepare_model_input(iq, args, ckpt_args))
        embeddings.append(F.normalize(out["embedding"].detach().cpu(), dim=-1))
        logits.append(out["logits"].detach().cpu())
        cfo.append(compute_cfo_features(model, iq).detach().cpu())
        oob.append(compute_oob_ratio_embedding(model, iq, args).detach().cpu())
        if include_labels:
            labels.append(batch["label"].detach().cpu())
    result = {
        "embedding": torch.cat(embeddings),
        "logits": torch.cat(logits),
        "cfo": torch.cat(cfo),
        "oob": torch.cat(oob),
    }
    if include_labels:
        result["label"] = torch.cat(labels)
    return result


def class_mean_std(features: torch.Tensor, labels: torch.Tensor, prototype_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    means = []
    stds = []
    for label in prototype_labels.tolist():
        class_features = features[labels == int(label)]
        if class_features.numel() == 0:
            means.append(torch.zeros(features.shape[1], dtype=features.dtype))
            stds.append(torch.ones(features.shape[1], dtype=features.dtype))
            continue
        means.append(class_features.mean(dim=0))
        std = class_features.std(dim=0, unbiased=False)
        stds.append(std.clamp_min(1e-6))
    return torch.stack(means), torch.stack(stds)


def class_oob_prototypes(oob_features: torch.Tensor, labels: torch.Tensor, prototype_labels: torch.Tensor) -> torch.Tensor:
    protos = []
    for label in prototype_labels.tolist():
        class_features = oob_features[labels == int(label)]
        if class_features.numel() == 0:
            protos.append(torch.zeros(oob_features.shape[1], dtype=oob_features.dtype))
        else:
            protos.append(class_features.mean(dim=0))
    return F.normalize(torch.stack(protos), dim=-1)


def oob_cfo_pseudo_proto_adapt(
    model: torch.nn.Module,
    train_loader,
    target_loader,
    device: torch.device,
    args: argparse.Namespace,
    ckpt_args: dict[str, Any],
    source_prototypes: torch.Tensor,
    prototype_labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = collect_rf_adapt_features(model, train_loader, device, args, ckpt_args, include_labels=True)
    target = collect_rf_adapt_features(model, target_loader, device, args, ckpt_args, include_labels=False)
    labels = source["label"]
    cfo_mean, cfo_std = class_mean_std(source["cfo"], labels, prototype_labels)
    oob_proto = class_oob_prototypes(source["oob"], labels, prototype_labels)
    label_to_proto = {int(label.item()): idx for idx, label in enumerate(prototype_labels)}

    cls_probs = F.softmax(target["logits"], dim=-1)
    confidence, cls_pred = cls_probs.max(dim=-1)
    proto_scores = target["embedding"] @ source_prototypes.T
    proto_pred = prototype_labels[proto_scores.argmax(dim=-1)]

    selected_by_class: dict[int, list[tuple[float, int, float, float]]] = defaultdict(list)
    rejected_conf = 0
    rejected_disagree = 0
    rejected_cfo = 0
    rejected_oob = 0
    for idx in range(cls_pred.shape[0]):
        pred = int(cls_pred[idx].item())
        conf = float(confidence[idx].item())
        if conf < args.pseudo_threshold:
            rejected_conf += 1
            continue
        if args.pseudo_require_cls_proto_agree and pred != int(proto_pred[idx].item()):
            rejected_disagree += 1
            continue
        if pred not in label_to_proto:
            rejected_disagree += 1
            continue
        proto_idx = label_to_proto[pred]
        cfo_z_vec = (target["cfo"][idx] - cfo_mean[proto_idx]).abs() / cfo_std[proto_idx]
        cfo_z = float(torch.sqrt((cfo_z_vec ** 2).mean()).item())
        if cfo_z > args.cfo_max_z:
            rejected_cfo += 1
            continue
        oob_sim = float((target["oob"][idx] * oob_proto[proto_idx]).sum().item())
        if oob_sim < args.oob_sim_threshold:
            rejected_oob += 1
            continue
        score = conf - args.cfo_consistency_weight * cfo_z + args.oob_consistency_weight * oob_sim
        selected_by_class[pred].append((score, idx, cfo_z, oob_sim))

    distribution: dict[str, int] = {}
    updated_labels = set()
    selected_indices = []
    selected_cfo_z = []
    selected_oob_sim = []
    mixed = source_prototypes.clone()
    for label in prototype_labels.tolist():
        label = int(label)
        candidates = sorted(selected_by_class.get(label, []), key=lambda item: item[0], reverse=True)
        if args.pseudo_balance_topk > 0:
            candidates = candidates[: args.pseudo_balance_topk]
        distribution[str(label)] = len(candidates)
        if len(candidates) < args.pseudo_min_per_class:
            continue
        class_indices = torch.tensor([item[1] for item in candidates], dtype=torch.long)
        selected_indices.extend(class_indices.tolist())
        selected_cfo_z.extend(item[2] for item in candidates)
        selected_oob_sim.extend(item[3] for item in candidates)
        target_proto = F.normalize(target["embedding"][class_indices].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        proto_idx = label_to_proto[label]
        mixed_proto = args.prototype_momentum * source_prototypes[proto_idx] + (1.0 - args.prototype_momentum) * target_proto
        mixed[proto_idx] = F.normalize(mixed_proto, dim=-1)
        updated_labels.add(label)

    stats = {
        "num_pseudo_selected": int(len(selected_indices)),
        "num_classes_updated": int(len(updated_labels)),
        "pseudo_class_distribution": json.dumps(distribution, sort_keys=True),
        "num_rejected_by_confidence": int(rejected_conf),
        "num_rejected_by_cls_proto_disagree": int(rejected_disagree),
        "num_rejected_by_cfo": int(rejected_cfo),
        "num_rejected_by_oob": int(rejected_oob),
        "mean_cfo_z_selected": float(sum(selected_cfo_z) / len(selected_cfo_z)) if selected_cfo_z else "",
        "mean_oob_sim_selected": float(sum(selected_oob_sim) / len(selected_oob_sim)) if selected_oob_sim else "",
    }
    return mixed, stats


def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    mode: str,
    args: argparse.Namespace,
    ckpt_args: dict[str, Any],
    prototypes=None,
    prototype_labels=None,
    num_classes: int | None = None,
    tta_mode: str = "none",
    tta_optimizer: torch.optim.Optimizer | None = None,
    episodic_state: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.score_fusion and (prototypes is None or prototype_labels is None):
        raise RuntimeError("--score-fusion requires train prototypes; use an RF-HSTU/hybrid checkpoint with prototype support.")
    for batch in tqdm(loader, leave=False, desc="eval"):
        labels = batch["label"]
        iq = batch["iq"].to(device)
        model_input = prepare_model_input(iq, args, ckpt_args)
        if tta_mode == "entropy_min":
            restore_model_state(model, episodic_state)
            tent_adapt_batch(model, model_input, args, tta_optimizer)
        with torch.no_grad():
            out = model(model_input)
        if args.score_fusion:
            if ckpt_args.get("model_type", args.model_type) == "osu_cnn":
                raise RuntimeError("--score-fusion is only supported for RF-HSTU/hybrid checkpoints.")
            cls_probs = F.softmax(out["logits"].detach().cpu(), dim=-1)
            embeddings = F.normalize(out["embedding"].detach().cpu(), dim=-1)
            proto_scores = embeddings @ prototypes.T
            proto_probs = F.softmax(proto_scores, dim=-1)
            full_proto_probs = torch.zeros(cls_probs.shape[0], int(num_classes or cls_probs.shape[-1]), dtype=proto_probs.dtype)
            full_proto_probs[:, prototype_labels.long()] = proto_probs
            scores = args.fusion_alpha * cls_probs + (1.0 - args.fusion_alpha) * full_proto_probs
            preds = scores.argmax(dim=-1)
            probs = scores
        elif mode == "classifier":
            scores = out["logits"].detach().cpu()
            probs = F.softmax(scores, dim=-1)
            preds = probs.argmax(dim=-1)
        else:
            embeddings = F.normalize(out["embedding"].detach().cpu(), dim=-1)
            scores = embeddings @ prototypes.T
            probs = F.softmax(scores, dim=-1)
            score_pred_idx = probs.argmax(dim=-1)
            preds = prototype_labels[score_pred_idx]
        confidences = probs.max(dim=-1).values
        domains = batch["domains"]
        for idx in range(labels.shape[0]):
            row = {
                "file_path": batch["file_path"][idx],
                "window_index": int(batch["window_index"][idx].item()),
                "sample_offset": int(batch["sample_offset"][idx].item()),
                "label": int(labels[idx].item()),
                "pred": int(preds[idx].item()),
                "correct": int(preds[idx].item() == labels[idx].item()),
                "split": batch["split"][idx],
                "setup": batch["setup"][idx],
                "confidence": float(confidences[idx].item()),
            }
            for col, field in enumerate(DOMAIN_FIELDS):
                row[field] = int(domains[idx, col].item())
            row["_scores"] = scores[idx]
            rows.append(row)
        if tta_mode == "entropy_min" and args.tent_episodic:
            restore_model_state(model, episodic_state)
    return rows


def file_level_predictions(
    window_rows: list[dict[str, Any]],
    mode: str,
    vote_mode: str = "mean_logits",
    prototype_labels: torch.Tensor | None = None,
    scores_are_probs: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        grouped[row["file_path"]].append(row)
    file_rows = []
    for path, rows in grouped.items():
        scores = torch.stack([row["_scores"] for row in rows])
        if scores_are_probs:
            probs = scores
            if vote_mode in {"mean_logits", "mean_prob"}:
                vote_probs = probs.mean(dim=0)
            elif vote_mode == "confidence_weighted":
                confidence = probs.max(dim=-1).values
                weights = confidence / confidence.sum().clamp_min(1e-8)
                vote_probs = (probs * weights.unsqueeze(-1)).sum(dim=0)
            else:
                raise ValueError(f"Unknown file_vote_mode={vote_mode!r}")
            file_scores = vote_probs
        elif vote_mode == "mean_logits":
            file_scores = scores.mean(dim=0)
            vote_probs = F.softmax(file_scores, dim=-1)
        else:
            probs = F.softmax(scores, dim=-1)
            if vote_mode == "mean_prob":
                vote_probs = probs.mean(dim=0)
            elif vote_mode == "confidence_weighted":
                confidence = probs.max(dim=-1).values
                weights = confidence / confidence.sum().clamp_min(1e-8)
                vote_probs = (probs * weights.unsqueeze(-1)).sum(dim=0)
            else:
                raise ValueError(f"Unknown file_vote_mode={vote_mode!r}")
            file_scores = vote_probs
        pred_idx = int(file_scores.argmax().item())
        pred = int(prototype_labels[pred_idx].item()) if mode == "prototype" and prototype_labels is not None else pred_idx
        label = int(rows[0]["label"])
        out = {key: rows[0][key] for key in ["file_path", "label", "split", "setup", *DOMAIN_FIELDS]}
        out.update(
            {
                "pred": pred,
                "correct": int(pred == label),
                "num_windows": len(rows),
                "confidence": float(vote_probs.max().item()),
            }
        )
        file_rows.append(out)
    return file_rows


def per_device_accuracy(window_rows: list[dict[str, Any]], file_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_file_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        by_label[int(row["label"])].append(row)
    for row in file_rows:
        by_file_label[int(row["label"])].append(row)
    labels = sorted(set(by_label) | set(by_file_label))
    rows = []
    for label in labels:
        windows = by_label.get(label, [])
        files = by_file_label.get(label, [])
        rows.append(
            {
                "label": label,
                "device_name": f"Device{label + 1}",
                "num_samples": len(windows),
                "num_files": len(files),
                "window_acc": sum(row["correct"] for row in windows) / max(1, len(windows)),
                "file_acc": sum(row["correct"] for row in files) / max(1, len(files)),
            }
        )
    return rows


def per_domain_accuracy(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ["setup", "split", *DOMAIN_FIELDS]
    result = []
    for field in fields:
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in window_rows:
            if field in row:
                groups[row[field]].append(row)
        for value in sorted(groups, key=lambda x: str(x)):
            rows = groups[value]
            result.append(
                {
                    "field": field,
                    "value": value,
                    "num_samples": len(rows),
                    "window_acc": sum(row["correct"] for row in rows) / max(1, len(rows)),
                }
            )
    return result


def clean_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "_scores"}
        cleaned.append(item)
    return cleaned


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    mode = resolve_mode(args)
    if args.eval_samples_per_file is None:
        args.eval_samples_per_file = args.samples_per_file
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    # Resolve normalization from the checkpoint so eval matches training exactly.
    # Missing keys (legacy checkpoints) fall back to the legacy pipeline behavior.
    ckpt_args_for_norm = ckpt.get("args", {})
    args.input_norm = ckpt_args_for_norm.get("input_norm", "iq_rms")
    args.fft_norm = ckpt_args_for_norm.get("fft_norm", "log_zscore")
    args.oob_norm = ckpt_args_for_norm.get("oob_norm", "zscore")
    model = build_model(args, ckpt, device)
    eval_split = args.eval_split or getattr(args, "val_split", "val")
    train_split = getattr(args, "train_split", "train")
    fold = getattr(args, "fold", None)
    train_rows = load_manifest(args.manifest, root=args.root, split=train_split, fold=fold)
    eval_rows = load_manifest(args.manifest, root=args.root, split=eval_split, fold=fold)
    input_norm = ckpt.get("args", {}).get("input_norm", getattr(args, "input_norm", "iq_rms"))
    eval_samples = args.eval_samples_per_file or args.samples_per_file
    if not eval_rows:
        _, eval_ds = make_datasets(args)
    else:
        eval_ds = SigMFIQDataset(
            eval_rows,
            window_size=args.window_size,
            samples_per_file=eval_samples,
            random_windows=False,
            seed=args.seed,
            input_norm=input_norm,
        )
    if args.eval_seed is not None:
        eval_ds.random_windows = True
        eval_ds.seed = args.eval_seed
    train_ds = SigMFIQDataset(
        train_rows,
        window_size=args.window_size,
        samples_per_file=args.samples_per_file,
        random_windows=False,
        seed=args.seed,
        input_norm=input_norm,
    )
    train_loader = make_loader(train_ds, args, shuffle=False)
    val_loader = make_loader(eval_ds, args, shuffle=False)
    adapt_loader = val_loader
    if args.adapt_batch_size is not None:
        adapt_args = copy.copy(args)
        adapt_args.batch_size = args.adapt_batch_size
        adapt_loader = make_loader(eval_ds, adapt_args, shuffle=False)
    ckpt_args = ckpt.get("args", {})
    num_classes = int(ckpt.get("num_classes", ckpt_args.get("num_classes", 25)))

    prototypes = None
    prototype_labels = None
    if args.score_fusion:
        model_type = ckpt.get("model_type", ckpt_args.get("model_type", args.model_type))
        if model_type == "osu_cnn":
            raise RuntimeError("--score-fusion is only supported for RF-HSTU/hybrid checkpoints.")
    if mode == "prototype" or args.score_fusion:
        train_z, train_y = collect_train_embeddings(model, train_loader, device, args, ckpt_args)
        prototypes, prototype_labels = build_prototypes(train_z, train_y)

    tta_mode, tta_optimizer, episodic_state = configure_tta(model, args)
    bn_modules_adapted = ""
    pseudo_stats = empty_pseudo_stats()
    if tta_mode == "bn_adapt":
        bn_modules_adapted = bn_adapt_model(model, adapt_loader, device, args, ckpt_args)
    if tta_mode == "pseudo_proto":
        if mode != "prototype":
            print("warning: pseudo_proto only changes prototype evaluation; classifier evaluation will remain source-only.")
        elif prototypes is None or prototype_labels is None:
            raise RuntimeError("pseudo_proto requires source prototypes.")
        else:
            prototypes, pseudo_stats = pseudo_proto_adapt(
                model,
                adapt_loader,
                device,
                args,
                ckpt_args,
                prototypes,
                prototype_labels,
            )
            pseudo_stats = {**empty_pseudo_stats(), **pseudo_stats}
    if tta_mode == "oob_cfo_pseudo_proto":
        if mode != "prototype":
            print("warning: oob_cfo_pseudo_proto only changes prototype evaluation; classifier evaluation will remain source-only.")
        elif prototypes is None or prototype_labels is None:
            raise RuntimeError("oob_cfo_pseudo_proto requires source prototypes.")
        else:
            prototypes, pseudo_stats = oob_cfo_pseudo_proto_adapt(
                model,
                train_loader,
                adapt_loader,
                device,
                args,
                ckpt_args,
                prototypes,
                prototype_labels,
            )

    window_rows = collect_predictions(
        model,
        val_loader,
        device,
        mode,
        args,
        ckpt_args,
        prototypes=prototypes,
        prototype_labels=prototype_labels,
        num_classes=num_classes,
        tta_mode=tta_mode,
        tta_optimizer=tta_optimizer,
        episodic_state=episodic_state,
    )
    file_rows = file_level_predictions(
        window_rows,
        mode,
        vote_mode=args.file_vote_mode,
        prototype_labels=prototype_labels,
        scores_are_probs=args.score_fusion,
    )
    window_labels = [int(row["label"]) for row in window_rows]
    window_preds = [int(row["pred"]) for row in window_rows]
    file_labels = [int(row["label"]) for row in file_rows]
    file_preds = [int(row["pred"]) for row in file_rows]

    window_acc = sum(row["correct"] for row in window_rows) / max(1, len(window_rows))
    file_acc = sum(row["correct"] for row in file_rows) / max(1, len(file_rows))
    window_macro_f1 = macro_f1(window_labels, window_preds, num_classes)
    file_macro_f1 = macro_f1(file_labels, file_preds, num_classes)
    ckpt_path = Path(args.checkpoint)
    ckpt_meta = load_checkpoint(ckpt_path, map_location="cpu")
    ckpt_extra_epoch = ckpt_meta.get("epoch", "")
    ckpt_val_acc = ckpt_meta.get("val_acc", "")
    ckpt_val_macro_f1 = ckpt_meta.get("val_macro_f1", "")
    ckpt_args_saved = ckpt_meta.get("args", {})
    checkpoint_metric = ckpt_args_saved.get("checkpoint_metric", "acc")
    eval_checkpoint = ckpt_path.name
    metrics = {
        "window_acc": window_acc,
        "file_acc": file_acc,
        "macro_f1": window_macro_f1,
        "window_macro_f1": window_macro_f1,
        "file_macro_f1": file_macro_f1,
        "num_windows": len(window_rows),
        "num_files": len(file_rows),
        "num_classes": num_classes,
        "eval_mode": mode,
        "file_vote_mode": args.file_vote_mode,
        "score_fusion": bool(args.score_fusion),
        "fusion_alpha": args.fusion_alpha if args.score_fusion else "",
        "tta_mode": tta_mode,
        "tent_steps": args.tent_steps,
        "tent_lr": args.tent_lr,
        "tent_episodic": bool(args.tent_episodic),
        "adapt_mode": tta_mode,
        "adapt_steps": resolve_adapt_steps(args),
        "adapt_lr": resolve_adapt_lr(args) if tta_mode == "entropy_min" else "",
        "pseudo_threshold": args.pseudo_threshold if tta_mode in {"pseudo_proto", "oob_cfo_pseudo_proto"} else "",
        "pseudo_topk_per_class": args.pseudo_topk_per_class if tta_mode == "pseudo_proto" else "",
        "pseudo_min_per_class": args.pseudo_min_per_class if tta_mode in {"pseudo_proto", "oob_cfo_pseudo_proto"} else "",
        "prototype_momentum": args.prototype_momentum if tta_mode in {"pseudo_proto", "oob_cfo_pseudo_proto"} else "",
        "adapt_batch_size": args.adapt_batch_size if args.adapt_batch_size is not None else "",
        "cfo_max_z": args.cfo_max_z if tta_mode == "oob_cfo_pseudo_proto" else "",
        "oob_sim_threshold": args.oob_sim_threshold if tta_mode == "oob_cfo_pseudo_proto" else "",
        "pseudo_balance_topk": args.pseudo_balance_topk if tta_mode == "oob_cfo_pseudo_proto" else "",
        "pseudo_require_cls_proto_agree": bool(args.pseudo_require_cls_proto_agree) if tta_mode == "oob_cfo_pseudo_proto" else "",
        "cfo_consistency_weight": args.cfo_consistency_weight if tta_mode == "oob_cfo_pseudo_proto" else "",
        "oob_consistency_weight": args.oob_consistency_weight if tta_mode == "oob_cfo_pseudo_proto" else "",
        "bn_modules_adapted": bn_modules_adapted,
        "num_pseudo_selected": pseudo_stats["num_pseudo_selected"],
        "num_classes_updated": pseudo_stats["num_classes_updated"],
        "pseudo_class_distribution": pseudo_stats["pseudo_class_distribution"],
        "num_rejected_by_confidence": pseudo_stats["num_rejected_by_confidence"],
        "num_rejected_by_cls_proto_disagree": pseudo_stats["num_rejected_by_cls_proto_disagree"],
        "num_rejected_by_cfo": pseudo_stats["num_rejected_by_cfo"],
        "num_rejected_by_oob": pseudo_stats["num_rejected_by_oob"],
        "mean_cfo_z_selected": pseudo_stats["mean_cfo_z_selected"],
        "mean_oob_sim_selected": pseudo_stats["mean_oob_sim_selected"],
        "eval_seed": args.eval_seed if args.eval_seed is not None else "",
        "checkpoint": str(args.checkpoint),
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_epoch": ckpt_extra_epoch,
        "checkpoint_val_acc": ckpt_val_acc,
        "checkpoint_val_macro_f1": ckpt_val_macro_f1,
        "eval_checkpoint": eval_checkpoint,
        "manifest": str(args.manifest),
        "model_type": ckpt.get("model_type", ckpt_args.get("model_type", args.model_type)),
        "cnn_input_type": ckpt.get("cnn_input_type", ckpt_args.get("cnn_input_type", args.cnn_input_type)),
        "oob_fusion_type": ckpt_args.get("oob_fusion_type", args.oob_fusion_type),
        "use_oob_cross_attention": bool(ckpt_args.get("use_oob_cross_attention", args.use_oob_cross_attention)),
        "use_chirp_embedding": bool(ckpt_args.get("use_chirp_embedding", args.use_chirp_embedding)),
        "use_multiscale": bool(ckpt_args.get("use_multiscale", args.use_multiscale)),
        "multiscale_ratios": ckpt_args.get("multiscale_ratios", args.multiscale_ratios),
        "patch_embed_type": ckpt_args.get("patch_embed_type", args.patch_embed_type),
        "cnn_stem_dim": ckpt_args.get("cnn_stem_dim", args.cnn_stem_dim),
        "cnn_stem_kernels": ckpt_args.get("cnn_stem_kernels", args.cnn_stem_kernels),
        "use_supcon": bool(ckpt_args.get("use_supcon", False)),
        "supcon_weight": ckpt_args.get("supcon_weight", ""),
        "supcon_temperature": ckpt_args.get("supcon_temperature", ""),
        "use_supcon_proj": bool(ckpt_args.get("use_supcon_proj", False)),
        "supcon_proj_dim": ckpt_args.get("supcon_proj_dim", ""),
        "augment_rf": bool(ckpt_args.get("augment_rf", False)),
        "aug_phase_std": ckpt_args.get("aug_phase_std", ""),
        "aug_amp_std": ckpt_args.get("aug_amp_std", ""),
        "aug_noise_std": ckpt_args.get("aug_noise_std", ""),
        "aug_time_shift": ckpt_args.get("aug_time_shift", ""),
        "balanced_batch": bool(ckpt_args.get("balanced_batch", False)),
        "devices_per_batch": ckpt_args.get("devices_per_batch", ""),
        "samples_per_device": ckpt_args.get("samples_per_device", ""),
        "use_hard_margin": bool(ckpt_args.get("use_hard_margin", False)),
        "hard_margin_weight": ckpt_args.get("hard_margin_weight", ""),
        "hard_margin": ckpt_args.get("hard_margin", ""),
        "use_center_loss": bool(ckpt_args.get("use_center_loss", False)),
        "center_loss_weight": ckpt_args.get("center_loss_weight", ""),
        "input_norm": ckpt_args.get("input_norm", args.input_norm),
        "fft_norm": ckpt_args.get("fft_norm", args.fft_norm),
        "oob_norm": ckpt_args.get("oob_norm", args.oob_norm),
        "augment_receiver_style": bool(ckpt_args.get("augment_receiver_style", False)),
        "rx_gain_db_min": ckpt_args.get("rx_gain_db_min", ""),
        "rx_gain_db_max": ckpt_args.get("rx_gain_db_max", ""),
        "rx_noise_std_min": ckpt_args.get("rx_noise_std_min", ""),
        "rx_noise_std_max": ckpt_args.get("rx_noise_std_max", ""),
        "rx_spectral_tilt_db_min": ckpt_args.get("rx_spectral_tilt_db_min", ""),
        "rx_spectral_tilt_db_max": ckpt_args.get("rx_spectral_tilt_db_max", ""),
        "rx_oob_scale_min": ckpt_args.get("rx_oob_scale_min", ""),
        "rx_oob_scale_max": ckpt_args.get("rx_oob_scale_max", ""),
        "rx_inband_scale_min": ckpt_args.get("rx_inband_scale_min", ""),
        "rx_inband_scale_max": ckpt_args.get("rx_inband_scale_max", ""),
        "use_cfo_feature": bool(ckpt_args.get("use_cfo_feature", False)),
        "cfo_feature_type": ckpt_args.get("cfo_feature_type", ""),
        "cfo_feature_norm": ckpt_args.get("cfo_feature_norm", ""),
        "use_target_unlabeled": bool(ckpt_args.get("use_target_unlabeled", False)),
        "target_manifest": ckpt_args.get("target_manifest", ""),
        "domain_align_loss": ckpt_args.get("domain_align_loss", ""),
        "domain_align_weight": ckpt_args.get("domain_align_weight", ""),
        "im_weight": ckpt_args.get("im_weight", ""),
        "target_loader_ratio": ckpt_args.get("target_loader_ratio", ""),
        "device": str(device),
    }

    out_dir = default_out_dir(args, mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    run_config = {"eval_args": vars(args), "checkpoint_args": ckpt_args}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    prediction_fields = [
        "file_path",
        "window_index",
        "sample_offset",
        "label",
        "pred",
        "correct",
        "split",
        "setup",
        *DOMAIN_FIELDS,
        "confidence",
    ]
    write_csv(out_dir / "predictions.csv", clean_prediction_rows(window_rows), prediction_fields)
    file_prediction_fields = [
        "file_path",
        "label",
        "pred",
        "correct",
        "num_windows",
        "split",
        "setup",
        *DOMAIN_FIELDS,
        "confidence",
    ]
    write_csv(out_dir / "file_predictions.csv", file_rows, file_prediction_fields)
    write_confusion(out_dir / "confusion_matrix.csv", confusion_matrix(window_labels, window_preds, num_classes))
    write_csv(
        out_dir / "per_device_accuracy.csv",
        per_device_accuracy(window_rows, file_rows),
        ["label", "device_name", "num_samples", "num_files", "window_acc", "file_acc"],
    )
    write_csv(out_dir / "per_domain_accuracy.csv", per_domain_accuracy(window_rows), ["field", "value", "num_samples", "window_acc"])

    print(format_metrics({"window_acc": window_acc, "file_acc": file_acc, "window_macro_f1": window_macro_f1, "file_macro_f1": file_macro_f1}))
    print(f"outputs={out_dir}")


if __name__ == "__main__":
    main()
