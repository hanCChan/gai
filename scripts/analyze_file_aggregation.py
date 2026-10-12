from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


FIELDS = [
    "file_path",
    "true_label",
    "file_pred",
    "file_correct",
    "num_windows",
    "window_acc_inside_file",
    "mean_top1_confidence",
    "std_top1_confidence",
    "mean_entropy",
    "top1_top2_margin_mean",
    "top1_top2_margin_std",
    "vote_distribution",
    "dominant_pred_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-file window aggregation behavior.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--group", default="file_path")
    return parser.parse_args()


def score_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("score_", "prob_", "logit_", "sim_")
    return [col for col in df.columns if col.startswith(prefixes)]


def entropy(values: list[float]) -> float:
    total = sum(values)
    if total <= 0:
        return float("nan")
    probs = [max(0.0, value / total) for value in values]
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def margin(values: list[float]) -> float:
    ordered = sorted(values, reverse=True)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return float(ordered[0])
    return float(ordered[0] - ordered[1])


def file_pred_from_votes(group: pd.DataFrame) -> int:
    counts = Counter(int(value) for value in group["pred"])
    max_count = max(counts.values())
    candidates = [pred for pred, count in counts.items() if count == max_count]
    if len(candidates) == 1 or "confidence" not in group.columns:
        return int(min(candidates))
    confidence_by_pred = group.groupby("pred")["confidence"].mean().to_dict()
    return int(max(candidates, key=lambda item: (float(confidence_by_pred.get(item, 0.0)), -int(item))))


def analyze_file(file_path: str, group: pd.DataFrame, scores: list[str]) -> dict[str, Any]:
    label = int(group["label"].iloc[0])
    pred = file_pred_from_votes(group)
    correct = int(pred == label)
    counts = Counter(int(value) for value in group["pred"])
    num_windows = len(group)
    vote_dist = {str(key): int(value) for key, value in sorted(counts.items())}
    dominant = max(counts.values()) / max(1, num_windows)
    window_acc = float((group["pred"].astype(int) == label).mean())

    mean_conf = ""
    std_conf = ""
    if "confidence" in group.columns:
        mean_conf = float(group["confidence"].astype(float).mean())
        std_conf = float(group["confidence"].astype(float).std(ddof=0))

    mean_entropy = ""
    margin_mean = ""
    margin_std = ""
    if scores:
        entropies = []
        margins = []
        for _, row in group.iterrows():
            vals = [float(row[col]) for col in scores]
            entropies.append(entropy(vals))
            margins.append(margin(vals))
        mean_entropy = float(pd.Series(entropies).mean())
        margin_mean = float(pd.Series(margins).mean())
        margin_std = float(pd.Series(margins).std(ddof=0))

    return {
        "file_path": file_path,
        "true_label": label,
        "file_pred": pred,
        "file_correct": correct,
        "num_windows": num_windows,
        "window_acc_inside_file": window_acc,
        "mean_top1_confidence": mean_conf,
        "std_top1_confidence": std_conf,
        "mean_entropy": mean_entropy,
        "top1_top2_margin_mean": margin_mean,
        "top1_top2_margin_std": margin_std,
        "vote_distribution": json.dumps(vote_dist, sort_keys=True),
        "dominant_pred_ratio": dominant,
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    if args.group not in df.columns:
        raise ValueError(f"Missing group column {args.group!r} in {args.predictions}")
    scores = score_columns(df)
    rows = [analyze_file(str(path), group, scores) for path, group in df.groupby(args.group, sort=False)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "file_analysis.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"file_analysis={out_path} files={len(rows)} score_columns={len(scores)}")


if __name__ == "__main__":
    main()
