from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfhstu.cnn_baseline import OSUCNNBaseline, build_cnn_input
from rfhstu.data import DOMAIN_FIELDS, infer_domain_sizes
from rfhstu.losses import CenterLoss, SupConLoss, supervised_contrastive_loss
from rfhstu.models import DeviceClassifier, RFPatchEmbedder
from rfhstu.train_utils import accuracy, add_common_args, format_metrics, load_checkpoint, make_datasets, make_loader, resolve_device, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RF-HSTU for device classification.")
    add_common_args(parser)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--out-dir", default="runs/finetune")
    parser.add_argument("--use-contrastive", action="store_true")
    parser.add_argument("--use-adversarial", action="store_true")
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    parser.add_argument("--adv-lambda", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--use-supcon", action="store_true")
    parser.add_argument("--supcon-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--use-supcon-proj", action="store_true")
    parser.add_argument("--supcon-proj-dim", type=int, default=64)
    parser.add_argument("--augment-rf", action="store_true")
    parser.add_argument("--aug-phase-std", type=float, default=0.0)
    parser.add_argument("--aug-amp-std", type=float, default=0.0)
    parser.add_argument("--aug-noise-std", type=float, default=0.0)
    parser.add_argument("--aug-time-shift", type=int, default=0)
    parser.add_argument("--use-hard-margin", action="store_true")
    parser.add_argument("--hard-margin-weight", type=float, default=0.0)
    parser.add_argument("--hard-margin", type=float, default=0.2)
    parser.add_argument("--use-center-loss", action="store_true")
    parser.add_argument("--center-loss-weight", type=float, default=0.0)
    return parser.parse_args()


class SupConProjector(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def maybe_load_pretrained(model: torch.nn.Module, path: str | None, device: torch.device) -> None:
    if not path:
        return
    ckpt = load_checkpoint(path, map_location=device)
    state = ckpt["model"]
    encoder_state = {key.replace("encoder.", "", 1): value for key, value in state.items() if key.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    print(f"loaded_pretrained={path} missing={len(missing)} unexpected={len(unexpected)}")


def prepare_model_input(iq: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.model_type == "osu_cnn":
        return build_cnn_input(
            iq,
            input_type=args.cnn_input_type,
            sample_rate=args.sample_rate,
            lora_bandwidth=args.lora_bandwidth,
        )
    return iq


def augment_iq(iq: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if not args.augment_rf:
        return iq
    x = iq
    bsz = x.shape[0]
    if args.aug_phase_std > 0:
        phi = torch.randn(bsz, 1, device=x.device, dtype=x.dtype) * args.aug_phase_std
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        i = x[:, 0] * cos_phi - x[:, 1] * sin_phi
        q = x[:, 0] * sin_phi + x[:, 1] * cos_phi
        x = torch.stack([i, q], dim=1)
    if args.aug_amp_std > 0:
        scale = 1.0 + torch.randn(bsz, 1, 1, device=x.device, dtype=x.dtype) * args.aug_amp_std
        x = x * scale.clamp_min(0.05)
    if args.aug_noise_std > 0:
        rms = torch.sqrt(x.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8))
        x = x + torch.randn_like(x) * (args.aug_noise_std * rms)
    if args.aug_time_shift > 0:
        shifts = torch.randint(-args.aug_time_shift, args.aug_time_shift + 1, (bsz,), device=x.device)
        x = torch.stack([torch.roll(sample, int(shift.item()), dims=-1) for sample, shift in zip(x, shifts)], dim=0)
    return x


def hard_negative_margin_loss(features: torch.Tensor, labels: torch.Tensor, classifier: nn.Module, margin: float) -> torch.Tensor:
    if not hasattr(classifier, "weight"):
        raise RuntimeError("--use-hard-margin requires a linear classifier with a weight matrix.")
    weight = classifier.weight
    f = F.normalize(features, dim=-1)
    w = F.normalize(weight, dim=-1)
    sim = f @ w.T
    pos = sim.gather(1, labels.view(-1, 1)).squeeze(1)
    neg_mask = torch.ones_like(sim, dtype=torch.bool)
    neg_mask.scatter_(1, labels.view(-1, 1), False)
    neg = sim.masked_fill(~neg_mask, -torch.inf).max(dim=1).values
    return F.relu(margin + neg - pos).mean()


def run_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    device: torch.device,
    args: argparse.Namespace,
    train: bool,
    supcon_projector: torch.nn.Module | None = None,
    center_loss: CenterLoss | None = None,
) -> dict[str, float]:
    model.train(train)
    total = {"loss": 0.0, "acc": 0.0}
    count = 0
    field_to_col = {field: idx for idx, field in enumerate(DOMAIN_FIELDS)}
    supcon = SupConLoss(args.supcon_temperature)
    for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
        iq = batch["iq"].to(device)
        if train:
            iq = augment_iq(iq, args)
        model_input = prepare_model_input(iq, args)
        labels = batch["label"].to(device)
        domains = batch["domains"].to(device)
        with torch.set_grad_enabled(train):
            out = model(model_input, adv_lambda=args.adv_lambda, return_features=args.use_supcon)
            logits = out["logits"]
            z = out.get("features", out["embedding"])
            loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
            if args.use_hard_margin:
                if args.use_supcon:
                    raise RuntimeError("--use-hard-margin and --use-supcon should not be enabled together in this first version.")
                loss = loss + args.hard_margin_weight * hard_negative_margin_loss(z, labels, model.classifier, args.hard_margin)
            if args.use_center_loss:
                if center_loss is None:
                    raise RuntimeError("--use-center-loss requires a CenterLoss module.")
                loss = loss + args.center_loss_weight * center_loss(z, labels)
            if args.use_supcon:
                supcon_z = supcon_projector(z) if supcon_projector is not None else z
                loss = loss + args.supcon_weight * supcon(supcon_z, labels)
            if args.use_contrastive:
                loss = loss + args.contrastive_weight * supervised_contrastive_loss(z, labels, args.temperature)
            if args.use_adversarial and model.domain_head is not None:
                domain_loss = model.domain_head.loss(out["domain_logits"], domains, field_to_col)
                loss = loss + args.adversarial_weight * domain_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bsz = model_input.shape[0]
        total["loss"] += loss.item() * bsz
        total["acc"] += accuracy(logits.detach(), labels) * bsz
        count += bsz
    return {key: value / max(1, count) for key, value in total.items()}


def main() -> None:
    args = parse_args()
    enabled_metric_losses = sum(bool(flag) for flag in [args.use_hard_margin, args.use_supcon, args.use_center_loss])
    if enabled_metric_losses > 1:
        raise ValueError("Enable only one of --use-hard-margin, --use-supcon, or --use-center-loss in this first version.")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_ds, val_ds = make_datasets(args)
    num_classes = max(row.label for row in [*train_ds.rows, *val_ds.rows]) + 1
    if args.model_type == "osu_cnn":
        domain_sizes = None
        model = OSUCNNBaseline(
            num_classes=num_classes,
            in_channels=2,
            hidden_dim=args.cnn_hidden_dim,
            dropout=args.cnn_dropout,
        ).to(device)
    else:
        embedder = RFPatchEmbedder(
            window_size=args.window_size,
            patch_size=args.patch_size,
            sample_rate=args.sample_rate,
            lora_bandwidth=args.lora_bandwidth,
            spreading_factor=args.spreading_factor,
            use_oob=not args.no_oob and args.oob_fusion_type != "no_oob",
            oob_fusion_type=args.oob_fusion_type,
            use_oob_cross_attention=args.use_oob_cross_attention,
            patch_embed_type=args.patch_embed_type,
            dim=args.dim,
            cnn_stem_dim=args.cnn_stem_dim,
            cnn_stem_kernels=args.cnn_stem_kernels,
        )
        domain_sizes = infer_domain_sizes([*train_ds.rows, *val_ds.rows]) if args.use_adversarial else None
        model = DeviceClassifier(
            embedder,
            num_classes=num_classes,
            dim=args.dim,
            depth=args.depth,
            dropout=args.dropout,
            domain_sizes=domain_sizes,
            use_chirp_embedding=args.use_chirp_embedding,
            oob_num_heads=args.oob_num_heads,
            use_multiscale=args.use_multiscale,
            multiscale_ratios=args.multiscale_ratios,
            multiscale_fusion_type=args.multiscale_fusion_type,
        ).to(device)
    maybe_load_pretrained(model, args.pretrained, device)
    supcon_projector = None
    if args.use_supcon and args.use_supcon_proj:
        supcon_projector = SupConProjector(args.dim if args.model_type == "rf_hstu" else args.cnn_hidden_dim, args.supcon_proj_dim).to(device)
    center_loss = None
    if args.use_center_loss:
        feat_dim = args.dim if args.model_type == "rf_hstu" else args.cnn_hidden_dim
        center_loss = CenterLoss(num_classes, feat_dim).to(device)
    optim_params = list(model.parameters())
    if supcon_projector is not None:
        optim_params.extend(supcon_projector.parameters())
    if center_loss is not None:
        optim_params.extend(center_loss.parameters())
    optimizer = AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, train=True, supcon_projector=supcon_projector, center_loss=center_loss)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, train=False, supcon_projector=supcon_projector, center_loss=center_loss)
        print(f"epoch={epoch} train_{format_metrics(train_metrics)} val_{format_metrics(val_metrics)}")
        extra = {
            "epoch": epoch,
            "val_acc": val_metrics["acc"],
            "num_classes": num_classes,
            "domain_sizes": domain_sizes,
            "model_type": args.model_type,
            "cnn_input_type": args.cnn_input_type,
        }
        if supcon_projector is not None:
            extra["supcon_projector"] = supcon_projector.state_dict()
        if center_loss is not None:
            extra["center_loss"] = center_loss.state_dict()
        save_checkpoint(Path(args.out_dir) / "last.pt", model, args, extra)
        if val_metrics["acc"] > best:
            best = val_metrics["acc"]
            save_checkpoint(Path(args.out_dir) / "best.pt", model, args, extra)


if __name__ == "__main__":
    main()
