from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
