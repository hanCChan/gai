from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DOMAIN_FIELDS, SigMFIQDataset, load_manifest
from .samplers import DeviceBalancedBatchSampler


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", default="data/raw/osu_lora/manifest_all.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--setup", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--model-type", choices=["rf_hstu", "osu_cnn"], default="rf_hstu")
    parser.add_argument("--cnn-input-type", choices=["iq", "fft", "fft_inband", "fft_oob", "amp_phase"], default="iq")
    parser.add_argument("--input-norm", choices=["none", "iq_rms"], default="none",
                        help="Per-window IQ normalization. none=raw, iq_rms=divide by RMS power.")
    parser.add_argument("--fft-norm", choices=["none", "log_zscore"], default="none",
                        help="FFT magnitude normalization (Hybrid only). none=log1p, log_zscore=log1p+per-window zscore.")
    parser.add_argument("--oob-norm", choices=["none", "ratio", "log_ratio", "zscore"], default="none",
                        help="OOB feature normalization (Hybrid only). none=masked log1p; ratio/log_ratio normalize by in-band RMS; zscore=masked log1p+zscore.")
    # Receiver-style augmentation (train-only; default OFF keeps behavior identical).
    parser.add_argument("--augment-receiver-style", action="store_true",
                        help="Enable train-only receiver-style augmentation (spectral tilt / in-band & OOB scale / noise floor / phase). No target labels used.")
    parser.add_argument("--rx-gain-db-min", type=float, default=-6.0)
    parser.add_argument("--rx-gain-db-max", type=float, default=6.0)
    parser.add_argument("--rx-noise-std-min", type=float, default=0.0)
    parser.add_argument("--rx-noise-std-max", type=float, default=0.01)
    parser.add_argument("--rx-spectral-tilt-db-min", type=float, default=-3.0)
    parser.add_argument("--rx-spectral-tilt-db-max", type=float, default=3.0)
    parser.add_argument("--rx-oob-scale-min", type=float, default=0.5)
    parser.add_argument("--rx-oob-scale-max", type=float, default=2.0)
    parser.add_argument("--rx-inband-scale-min", type=float, default=0.7)
    parser.add_argument("--rx-inband-scale-max", type=float, default=1.5)
    # CFO / peak_offset auxiliary feature (default OFF keeps behavior identical).
    parser.add_argument("--use-cfo-feature", action="store_true",
                        help="Concatenate a CFO proxy (peak_offset / spectral_centroid) to the pooled embedding. No CFO compensation is performed.")
    parser.add_argument("--cfo-feature-type", choices=["peak_offset", "spectral_centroid", "both"], default="both")
    parser.add_argument("--cfo-feature-norm", choices=["train_zscore"], default="train_zscore")
    # Target-unlabeled domain alignment (default OFF; no target labels used in loss).
    parser.add_argument("--use-target-unlabeled", action="store_true",
                        help="Use target-receiver unlabeled windows during training (val split of target manifest).")
    parser.add_argument("--target-manifest", default=None,
                        help="Manifest for target unlabeled data. Defaults to --manifest val split.")
    parser.add_argument("--domain-align-loss", choices=["none", "coral", "coral_im"], default="none")
    parser.add_argument("--domain-align-weight", type=float, default=1.0)
    parser.add_argument("--im-weight", type=float, default=0.1)
    parser.add_argument("--target-loader-ratio", type=int, default=1,
                        help="Number of target batches per source batch when aligning domains.")
    parser.add_argument("--cnn-hidden-dim", type=int, default=25)
    parser.add_argument("--cnn-dropout", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--samples-per-file", type=int, default=128)
    parser.add_argument("--eval-samples-per-file", type=int, default=None)
    parser.add_argument("--sample-rate", type=float, default=1_000_000.0)
    parser.add_argument("--lora-bandwidth", type=float, default=125_000.0)
    parser.add_argument("--spreading-factor", type=int, default=7)
    parser.add_argument("--use-chirp-embedding", action="store_true")
    parser.add_argument("--no-oob", action="store_true")
    parser.add_argument("--use-oob-cross-attention", action="store_true")
    parser.add_argument(
        "--oob-fusion-type",
        choices=["no_oob", "concat_oob", "cross_attn_oob", "gated_oob"],
        default="concat_oob",
    )
    parser.add_argument("--oob-num-heads", type=int, default=4)
    parser.add_argument("--use-multiscale", action="store_true")
    parser.add_argument("--multiscale-ratios", default="1,2,4")
    parser.add_argument("--multiscale-fusion-type", choices=["concat_pool"], default="concat_pool")
    parser.add_argument("--patch-embed-type", choices=["linear", "cnn_stem"], default="linear")
    parser.add_argument("--cnn-stem-dim", type=int, default=32)
    parser.add_argument("--cnn-stem-kernels", default="7,5")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--balanced-batch", action="store_true")
    parser.add_argument("--devices-per-batch", type=int, default=8)
    parser.add_argument("--samples-per-device", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--train-split",
        default="train",
        help="Manifest split name for training rows (default: train).",
    )
    parser.add_argument(
        "--val-split",
        default="val",
        help="Manifest split for early-stopping / best checkpoint (default: val).",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Manifest split for final evaluation (evaluate.py). Defaults to val-split.",
    )
    parser.add_argument(
        "--fold",
        default=None,
        help="LOCO fold id (must match manifest fold column, e.g. 1 or 5m).",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=["acc", "macro_f1"],
        default="acc",
        help="Metric for best.pt selection on val split.",
    )
    parser.add_argument(
        "--loss-type",
        choices=["ce", "focal"],
        default="ce",
        help="Classification loss (focal helps class imbalance / low macro-F1).",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--class-balanced-ce",
        action="store_true",
        help="Inverse-frequency class weights in CE/focal loss.",
    )
    parser.add_argument(
        "--oob-dropout",
        type=float,
        default=0.0,
        help="Train-only: drop OOB branch with this probability (forces IQ path robustness).",
    )
    parser.add_argument(
        "--mixstyle",
        action="store_true",
        help="Train-only MixStyle on encoder tokens (domain style randomization).",
    )
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1)
    parser.add_argument(
        "--use-swa",
        action="store_true",
        help="Stochastic Weight Averaging over last 20%% of epochs.",
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_datasets(args: argparse.Namespace) -> tuple[SigMFIQDataset, SigMFIQDataset]:
    train_split = getattr(args, "train_split", "train")
    val_split = getattr(args, "val_split", "val")
    fold = getattr(args, "fold", None)
    train_rows = load_manifest(
        args.manifest,
        root=args.root,
        split=train_split,
        setup=args.setup,
        fold=fold,
        max_files=args.max_files,
    )
    val_rows = load_manifest(
        args.manifest,
        root=args.root,
        split=val_split,
        setup=args.setup,
        fold=fold,
        max_files=args.max_files,
    )
    if not val_rows:
        raise ValueError(
            f"No val rows for manifest={args.manifest} split={val_split} fold={fold}. "
            "Regenerate manifest with val split or pick a different --val-split."
        )
    val_samples_per_file = args.eval_samples_per_file
    if val_samples_per_file is None:
        val_samples_per_file = max(1, min(args.samples_per_file, 32))
    input_norm = getattr(args, "input_norm", "iq_rms")
    train_ds = SigMFIQDataset(
        train_rows,
        window_size=args.window_size,
        samples_per_file=args.samples_per_file,
        random_windows=True,
        seed=args.seed,
        input_norm=input_norm,
    )
    val_ds = SigMFIQDataset(
        val_rows,
        window_size=args.window_size,
        samples_per_file=val_samples_per_file,
        random_windows=False,
        seed=args.seed,
        input_norm=input_norm,
    )
    return train_ds, val_ds


def make_target_unlabeled_dataset(args: argparse.Namespace) -> SigMFIQDataset:
    """Target-receiver windows for unsupervised alignment (val split, labels not used in loss)."""
    manifest = args.target_manifest or args.manifest
    fold = getattr(args, "fold", None)
    rows = load_manifest(manifest, root=args.root, split="val", setup=args.setup, fold=fold, max_files=args.max_files)
    if not rows:
        raise ValueError(f"No target unlabeled rows (val split) in manifest={manifest}")
    input_norm = getattr(args, "input_norm", "iq_rms")
    return SigMFIQDataset(
        rows,
        window_size=args.window_size,
        samples_per_file=args.samples_per_file,
        random_windows=True,
        seed=args.seed + 17,
        input_norm=input_norm,
    )


def make_loader(dataset: SigMFIQDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if shuffle and args.balanced_batch:
        sampler = DeviceBalancedBatchSampler(
            dataset,
            devices_per_batch=args.devices_per_batch,
            samples_per_device=args.samples_per_device,
            seed=args.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            **loader_kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        **loader_kwargs,
        drop_last=False,
    )


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()


def save_checkpoint(path: str | Path, model: torch.nn.Module, args: argparse.Namespace, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "args": vars(args),
        "domain_fields": DOMAIN_FIELDS,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
