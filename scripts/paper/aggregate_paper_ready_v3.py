#!/usr/bin/env python3
"""Aggregate paper_ready_v3 step results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRIC_FIELDS = [
    "step",
    "model_id",
    "seed",
    "model_type",
    "oob_fusion_type",
    "oob_norm",
    "window_acc",
    "window_macro_f1",
    "file_acc",
    "file_macro_f1",
    "checkpoint_metric",
    "checkpoint_epoch",
    "checkpoint_val_acc",
    "checkpoint_val_macro_f1",
    "eval_checkpoint",
    "manifest",
    "metrics_path",
]


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_step_metrics(run_base: Path, step: str) -> list[dict]:
    out_root = run_base / step / "outputs"
    if not out_root.exists():
        return []
    rows: list[dict] = []
    for metrics_path in sorted(out_root.rglob("metrics.json")):
        m = read_metrics(metrics_path)
        rel = metrics_path.parent.relative_to(out_root)
        parts = rel.parts
        model_id = parts[0] if parts else str(rel)
        seed = parts[1].replace("seed_", "") if len(parts) > 1 else ""
        rows.append(
            {
                "step": step,
                "model_id": model_id,
                "seed": seed,
                "model_type": m.get("model_type", ""),
                "oob_fusion_type": m.get("oob_fusion_type", ""),
                "oob_norm": m.get("oob_norm", ""),
                "window_acc": m.get("window_acc", ""),
                "window_macro_f1": m.get("window_macro_f1", m.get("macro_f1", "")),
                "file_acc": m.get("file_acc", ""),
                "file_macro_f1": m.get("file_macro_f1", ""),
                "checkpoint_metric": m.get("checkpoint_metric", ""),
                "checkpoint_epoch": m.get("checkpoint_epoch", ""),
                "checkpoint_val_acc": m.get("checkpoint_val_acc", ""),
                "checkpoint_val_macro_f1": m.get("checkpoint_val_macro_f1", ""),
                "eval_checkpoint": m.get("eval_checkpoint", "best.pt"),
                "manifest": m.get("manifest", ""),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def aggregate_multiseed(rows: list[dict], core_only: bool = False) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        mid = row["model_id"]
        if core_only and not mid.startswith(("A_", "D_", "F_", "H_")):
            continue
        grouped[mid].append(row)
    out = []
    for model_id, items in sorted(grouped.items()):
        for metric in ("file_acc", "file_macro_f1", "window_acc", "window_macro_f1"):
            vals = [float(r[metric]) for r in items if r.get(metric) not in ("", None)]
            if not vals:
                continue
            out.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "n_seeds": len(vals),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": json.dumps(vals),
                }
            )
    return out


def pick_winner(summary: list[dict]) -> str:
    scores: dict[str, list[float]] = defaultdict(list)
    for row in summary:
        if row["metric"] == "file_macro_f1":
            scores[row["model_id"]].append(row["mean"])
    if not scores:
        return "F_cross_attn_chirp_plain"
    return max(scores, key=lambda k: float(np.mean(scores[k])))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-base", required=True)
    parser.add_argument("--step", default="step1")
    parser.add_argument("--out-dir", default="outputs/paper_ready_v3")
    args = parser.parse_args()

    root = Path(args.root)
    run_base = Path(args.run_base)
    if not run_base.is_absolute():
        run_base = root / run_base
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    step_key = args.step if args.step.startswith("step") else f"step{args.step}"
    rows = collect_step_metrics(run_base, step_key)
    out_sub = {
        "step1": "step1_phase7_clean",
        "step2": "step2_recipe",
        "step3": "step3_location_outdoor",
        "step4": "step4_fixed_epoch",
    }.get(step_key, step_key)
    write_csv(out_dir / out_sub / "metrics_all.csv", rows, METRIC_FIELDS)

    summary = aggregate_multiseed(rows, core_only=(step_key == "step1"))
    write_csv(out_dir / out_sub / "multiseed_summary.csv", summary, ["model_id", "metric", "n_seeds", "mean", "std", "values"])

    if step_key == "step1":
        winner = pick_winner(summary)
        (out_dir / "step1_phase7_clean" / "winner.txt").write_text(winner + "\n", encoding="utf-8")
        print(f"Step1 winner (file_macro_f1): {winner}")

    print(f"Aggregated {len(rows)} runs -> {out_dir / out_sub}/")


if __name__ == "__main__":
    main()
