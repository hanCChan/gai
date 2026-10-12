from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "method_family",
    "experiment",
    "window_acc",
    "file_acc",
    "macro_f1",
    "eval_mode",
    "model_type",
    "cnn_input_type",
    "oob_fusion_type",
    "use_oob_cross_attention",
    "use_chirp_embedding",
    "use_multiscale",
    "patch_embed_type",
    "cnn_stem_dim",
    "cnn_stem_kernels",
    "num_windows",
    "num_files",
    "checkpoint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OSU CNN baselines with RF-HSTU results.")
    parser.add_argument("--cnn-root", default="outputs/osu_cnn_day1_day2_spf256")
    parser.add_argument("--rfhstu-root", default="outputs/cross_day_oob_main_spf256")
    parser.add_argument("--out", default="outputs/compare_osu_cnn_vs_rfhstu_spf256.csv")
    return parser.parse_args()


def _read_summary(root: Path) -> list[dict[str, Any]]:
    summary = root / "summary.csv"
    if not summary.exists():
        return []
    with summary.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_metrics(root: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rel = metrics_path.parent.relative_to(root)
        row = {"experiment": str(rel).replace("\\", "/")}
        row.update(metrics)
        rows.append(row)
    return rows


def load_rows(root: Path, method_family: str) -> list[dict[str, Any]]:
    rows = _read_summary(root) or _read_metrics(root)
    out = []
    for row in rows:
        item = {field: row.get(field, "") for field in FIELDS}
        item["method_family"] = method_family
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    rows = [
        *load_rows(Path(args.cnn_root), "osu_cnn"),
        *load_rows(Path(args.rfhstu_root), "rfhstu"),
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"comparison={out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
