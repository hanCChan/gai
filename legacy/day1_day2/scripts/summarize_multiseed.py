from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = ("window_acc", "file_acc", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multi-seed mean/std metrics from summary.csv.")
    parser.add_argument("--summary", default="outputs/hybrid_best_multiseed_spf256/summary.csv")
    parser.add_argument("--out", default="outputs/hybrid_best_multiseed_spf256/multiseed_summary.csv")
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


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("eval_mode", ""), row.get("file_vote_mode", ""))].append(row)

    out_rows = []
    for (eval_mode, file_vote_mode), items in sorted(groups.items()):
        out = {
            "eval_mode": eval_mode,
            "file_vote_mode": file_vote_mode,
            "num_runs": len(items),
        }
        for metric in METRICS:
            values = [to_float(row.get(metric)) for row in items]
            out[f"mean_{metric}"] = mean(values)
            out[f"std_{metric}"] = std(values)
        out_rows.append(out)

    fieldnames = [
        "eval_mode",
        "file_vote_mode",
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
    print(f"multiseed_summary={out_path} rows={len(out_rows)}")


if __name__ == "__main__":
    main()
