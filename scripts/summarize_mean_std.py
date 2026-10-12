from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = ("window_acc", "file_acc", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate summary.csv metrics as mean/std by method and eval mode.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--group-cols", default="method,eval_mode,file_vote_mode")
    return parser.parse_args()


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) / max(1, len(clean))


def std(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if len(clean) <= 1:
        return 0.0
    mu = mean(clean)
    return math.sqrt(sum((value - mu) ** 2 for value in clean) / (len(clean) - 1))


def infer_method(row: dict[str, Any]) -> str:
    if row.get("method"):
        return str(row["method"])
    experiment = str(row.get("experiment", ""))
    if "/" not in experiment:
        return experiment
    first = experiment.split("/", 1)[0]
    if first in {"cnn", "hybrid"}:
        return first
    if first.startswith("seed_") and "/" in experiment:
        parts = experiment.split("/")
        return parts[1] if len(parts) > 1 else first
    return first


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    group_cols = [item.strip() for item in args.group_cols.split(",") if item.strip()]
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "method" in group_cols:
            row["method"] = infer_method(row)
        key = tuple(str(row.get(col, "")) for col in group_cols)
        groups[key].append(row)

    out_rows = []
    for key, items in sorted(groups.items()):
        out = {col: value for col, value in zip(group_cols, key)}
        out["num_runs"] = len(items)
        for metric in METRICS:
            values = [to_float(row.get(metric)) for row in items]
            out[f"mean_{metric}"] = mean(values)
            out[f"std_{metric}"] = std(values)
        out_rows.append(out)

    fieldnames = [
        *group_cols,
        "num_runs",
        "mean_window_acc",
        "std_window_acc",
        "mean_file_acc",
        "std_file_acc",
        "mean_macro_f1",
        "std_macro_f1",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"mean_std={out_path} rows={len(out_rows)}")


if __name__ == "__main__":
    main()
