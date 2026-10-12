from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


FIELDS = [
    "experiment",
    "window_acc",
    "file_acc",
    "macro_f1",
    "eval_mode",
    "file_vote_mode",
    "score_fusion",
    "fusion_alpha",
    "tta_mode",
    "tent_steps",
    "tent_lr",
    "tent_episodic",
    "adapt_mode",
    "adapt_steps",
    "adapt_lr",
    "pseudo_threshold",
    "pseudo_topk_per_class",
    "pseudo_min_per_class",
    "prototype_momentum",
    "adapt_batch_size",
    "cfo_max_z",
    "oob_sim_threshold",
    "pseudo_balance_topk",
    "pseudo_require_cls_proto_agree",
    "cfo_consistency_weight",
    "oob_consistency_weight",
    "bn_modules_adapted",
    "num_pseudo_selected",
    "num_classes_updated",
    "pseudo_class_distribution",
    "num_rejected_by_confidence",
    "num_rejected_by_cls_proto_disagree",
    "num_rejected_by_cfo",
    "num_rejected_by_oob",
    "mean_cfo_z_selected",
    "mean_oob_sim_selected",
    "eval_seed",
    "model_type",
    "cnn_input_type",
    "oob_fusion_type",
    "use_chirp_embedding",
    "use_multiscale",
    "multiscale_ratios",
    "patch_embed_type",
    "cnn_stem_dim",
    "cnn_stem_kernels",
    "use_supcon",
    "supcon_weight",
    "supcon_temperature",
    "use_supcon_proj",
    "supcon_proj_dim",
    "augment_rf",
    "aug_phase_std",
    "aug_amp_std",
    "aug_noise_std",
    "aug_time_shift",
    "balanced_batch",
    "devices_per_batch",
    "samples_per_device",
    "use_hard_margin",
    "hard_margin_weight",
    "hard_margin",
    "use_center_loss",
    "center_loss_weight",
    "input_norm",
    "fft_norm",
    "oob_norm",
    "augment_receiver_style",
    "use_cfo_feature",
    "cfo_feature_type",
    "cfo_feature_norm",
    "use_target_unlabeled",
    "target_manifest",
    "domain_align_loss",
    "domain_align_weight",
    "im_weight",
    "target_loader_ratio",
    "best_epoch",
    "best_val_acc",
    "checkpoint",
    "num_windows",
    "num_files",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize evaluation metrics.json files.")
    parser.add_argument("--root", default="outputs/ablation_cross_day")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rel = metrics_path.parent.relative_to(root)
        row = {"experiment": str(rel).replace("\\", "/")}
        for field in FIELDS:
            if field == "experiment":
                continue
            row[field] = metrics.get(field, "")
        checkpoint = metrics.get("checkpoint", "")
        if checkpoint:
            ckpt_path = Path(checkpoint)
            if not ckpt_path.is_absolute():
                ckpt_path = Path.cwd() / ckpt_path
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    row["best_epoch"] = ckpt.get("epoch", row.get("best_epoch", ""))
                    row["best_val_acc"] = ckpt.get("val_acc", row.get("best_val_acc", ""))
                except Exception as exc:
                    print(f"warning: could not read checkpoint metadata from {ckpt_path}: {exc}")
        rows.append(row)

    out_path = root / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"summary={out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
