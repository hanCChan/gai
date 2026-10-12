from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfhstu.losses import reconstruction_loss
from rfhstu.models import RFMAE, RFPatchEmbedder
from rfhstu.train_utils import add_common_args, format_metrics, make_datasets, make_loader, resolve_device, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain RF-HSTU with masked RF modeling.")
    add_common_args(parser)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--out-dir", default="runs/pretrain_mae")
    return parser.parse_args()


def run_epoch(model: RFMAE, loader, optimizer, device: torch.device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_count = 0
    for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
        iq = batch["iq"].to(device)
        with torch.set_grad_enabled(train):
            pred, target, mask = model(iq)
            loss = reconstruction_loss(pred, target, mask)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * iq.shape[0]
        total_count += iq.shape[0]
    return total_loss / max(1, total_count)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if args.use_multiscale:
        print("WARNING: --use-multiscale is ignored by pretrain_mae.py; multi-scale is only used in finetune/evaluate.")
    train_ds, val_ds = make_datasets(args)
    embedder = RFPatchEmbedder(
        window_size=args.window_size,
        patch_size=args.patch_size,
        sample_rate=args.sample_rate,
        lora_bandwidth=args.lora_bandwidth,
        spreading_factor=args.spreading_factor,
        use_oob=not args.no_oob and args.oob_fusion_type != "no_oob",
        oob_fusion_type=args.oob_fusion_type,
        use_oob_cross_attention=args.use_oob_cross_attention,
    )
    model = RFMAE(
        embedder,
        dim=args.dim,
        depth=args.depth,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        use_chirp_embedding=args.use_chirp_embedding,
        oob_num_heads=args.oob_num_heads,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, args, shuffle=True)
    val_loader = make_loader(val_ds, args, shuffle=False)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        print(f"epoch={epoch} {format_metrics({'train_loss': train_loss, 'val_loss': val_loss})}")
        save_checkpoint(Path(args.out_dir) / "last.pt", model, args, {"epoch": epoch, "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            save_checkpoint(Path(args.out_dir) / "best.pt", model, args, {"epoch": epoch, "val_loss": val_loss})


if __name__ == "__main__":
    main()
