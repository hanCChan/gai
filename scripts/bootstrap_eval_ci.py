from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap file-level accuracy and macro-F1 confidence intervals.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--group", default="file_path")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--format", choices=["long", "wide"], default="long")
    return parser.parse_args()


def macro_f1(labels: list[int], preds: list[int]) -> float:
    scores = []
    for label in sorted(set(labels) | set(preds)):
        tp = sum(1 for y, p in zip(labels, preds) if y == label and p == label)
        fp = sum(1 for y, p in zip(labels, preds) if y != label and p == label)
        fn = sum(1 for y, p in zip(labels, preds) if y == label and p != label)
        if tp + fp + fn == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / max(1, len(scores))


def aggregate_file_predictions(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if "num_windows" in df.columns:
        return df[[group_col, "label", "pred"]].drop_duplicates(group_col).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for file_path, group in df.groupby(group_col, sort=False):
        label = int(group["label"].iloc[0])
        counts = Counter(int(value) for value in group["pred"])
        max_count = max(counts.values())
        candidates = [pred for pred, count in counts.items() if count == max_count]
        if len(candidates) == 1 or "confidence" not in group.columns:
            pred = min(candidates)
        else:
            confidence_by_pred = group.groupby("pred")["confidence"].mean().to_dict()
            pred = max(candidates, key=lambda item: (float(confidence_by_pred.get(item, 0.0)), -int(item)))
        rows.append({group_col: file_path, "label": label, "pred": int(pred)})
    return pd.DataFrame(rows)


def metric_values(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    acc = float((labels == preds).mean()) if labels.size else 0.0
    f1 = macro_f1(labels.astype(int).tolist(), preds.astype(int).tolist())
    return acc, f1


def ci(values: list[float], level: float) -> tuple[float, float]:
    alpha = 1.0 - level
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    if args.group not in df.columns:
        raise ValueError(f"Missing group column {args.group!r} in {args.predictions}")
    file_df = aggregate_file_predictions(df, args.group)
    labels = file_df["label"].to_numpy()
    preds = file_df["pred"].to_numpy()
    point_acc, point_f1 = metric_values(labels, preds)

    rng = np.random.default_rng(args.seed)
    boot_acc: list[float] = []
    boot_f1: list[float] = []
    n_files = len(file_df)
    for _ in range(args.n_bootstrap):
        idx = rng.integers(0, n_files, size=n_files)
        acc, f1 = metric_values(labels[idx], preds[idx])
        boot_acc.append(acc)
        boot_f1.append(f1)

    acc_low, acc_high = ci(boot_acc, args.level)
    f1_low, f1_high = ci(boot_f1, args.level)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        if args.format == "wide":
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "point_file_acc",
                    "file_acc_ci_low",
                    "file_acc_ci_high",
                    "point_macro_f1",
                    "macro_f1_ci_low",
                    "macro_f1_ci_high",
                    "level",
                    "n_files",
                    "n_bootstrap",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "point_file_acc": point_acc,
                    "file_acc_ci_low": acc_low,
                    "file_acc_ci_high": acc_high,
                    "point_macro_f1": point_f1,
                    "macro_f1_ci_low": f1_low,
                    "macro_f1_ci_high": f1_high,
                    "level": args.level,
                    "n_files": n_files,
                    "n_bootstrap": args.n_bootstrap,
                }
            )
        else:
            writer = csv.DictWriter(
                f,
                fieldnames=["metric", "point_estimate", "lower_ci", "upper_ci", "level", "n_files", "n_bootstrap"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "metric": "file_acc",
                    "point_estimate": point_acc,
                    "lower_ci": acc_low,
                    "upper_ci": acc_high,
                    "level": args.level,
                    "n_files": n_files,
                    "n_bootstrap": args.n_bootstrap,
                }
            )
            writer.writerow(
                {
                    "metric": "macro_f1",
                    "point_estimate": point_f1,
                    "lower_ci": f1_low,
                    "upper_ci": f1_high,
                    "level": args.level,
                    "n_files": n_files,
                    "n_bootstrap": args.n_bootstrap,
                }
            )
    print(f"bootstrap_ci={out} n_files={n_files}")


if __name__ == "__main__":
    main()
