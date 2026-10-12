from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_A = "outputs/osu_cnn_day1_day2_spf256/osu_cnn_iq/classifier/predictions.csv"
DEFAULT_B = "outputs/cross_day_hybrid_cnnstem_spf256/rfhstu_cnnstem_cross_attn_chirp/classifier/predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two prediction CSV files at file level.")
    parser.add_argument("--a-pred", default=DEFAULT_A)
    parser.add_argument("--b-pred", default=DEFAULT_B)
    parser.add_argument("--a-name", default="osu_cnn_iq_classifier")
    parser.add_argument("--b-name", default="hybrid_cnnstem_cross_attn_chirp_classifier")
    parser.add_argument("--out-dir", default="outputs/analysis_cnn_vs_hybrid")
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _to_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def aggregate_file_level(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["file_path"]].append(row)

    out = {}
    for path, items in grouped.items():
        label = _to_int(items[0], "label")
        pred_counts = Counter(_to_int(row, "pred") for row in items)
        pred = pred_counts.most_common(1)[0][0]
        correct = int(pred == label)
        base = {
            "file_path": path,
            "label": label,
            "pred": pred,
            "correct": correct,
            "num_windows": len(items),
        }
        for key in ["split", "setup", "day", "receiver", "location", "distance", "sf", "config"]:
            if key in items[0]:
                base[key] = items[0][key]
        out[path] = base
    return out


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare(a_rows: dict[str, dict[str, Any]], b_rows: dict[str, dict[str, Any]], a_name: str, b_name: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    paths = sorted(set(a_rows) & set(b_rows))
    detail_rows = []
    buckets = {
        "a_correct_b_wrong": [],
        "b_correct_a_wrong": [],
        "both_wrong": [],
        "both_correct": [],
    }
    for path in paths:
        a = a_rows[path]
        b = b_rows[path]
        row = {
            "file_path": path,
            "label": a["label"],
            f"{a_name}_pred": a["pred"],
            f"{a_name}_correct": a["correct"],
            f"{b_name}_pred": b["pred"],
            f"{b_name}_correct": b["correct"],
            "num_windows_a": a["num_windows"],
            "num_windows_b": b["num_windows"],
        }
        for key in ["split", "setup", "day", "receiver", "location", "distance", "sf", "config"]:
            if key in a:
                row[key] = a[key]
        detail_rows.append(row)
        if a["correct"] and not b["correct"]:
            buckets["a_correct_b_wrong"].append(row)
        elif b["correct"] and not a["correct"]:
            buckets["b_correct_a_wrong"].append(row)
        elif a["correct"] and b["correct"]:
            buckets["both_correct"].append(row)
        else:
            buckets["both_wrong"].append(row)
    return detail_rows, buckets


def per_device_delta(detail_rows: list[dict[str, Any]], a_name: str, b_name: str) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        by_label[_to_int(row, "label")].append(row)
    rows = []
    for label in sorted(by_label):
        items = by_label[label]
        a_acc = sum(_to_int(row, f"{a_name}_correct") for row in items) / max(1, len(items))
        b_acc = sum(_to_int(row, f"{b_name}_correct") for row in items) / max(1, len(items))
        rows.append(
            {
                "label": label,
                "device_name": f"Device{label + 1}",
                "num_files": len(items),
                f"{a_name}_file_acc": a_acc,
                f"{b_name}_file_acc": b_acc,
                "delta_b_minus_a": b_acc - a_acc,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    a = aggregate_file_level(read_csv(args.a_pred))
    b = aggregate_file_level(read_csv(args.b_pred))
    detail_rows, buckets = compare(a, b, args.a_name, args.b_name)
    out_dir = Path(args.out_dir)
    base_fields = list(detail_rows[0].keys()) if detail_rows else [
        "file_path",
        "label",
        f"{args.a_name}_pred",
        f"{args.a_name}_correct",
        f"{args.b_name}_pred",
        f"{args.b_name}_correct",
        "num_windows_a",
        "num_windows_b",
    ]
    write_rows(
        out_dir / "per_device_delta.csv",
        per_device_delta(detail_rows, args.a_name, args.b_name),
        ["label", "device_name", "num_files", f"{args.a_name}_file_acc", f"{args.b_name}_file_acc", "delta_b_minus_a"],
    )
    for name, rows in buckets.items():
        write_rows(out_dir / f"{name}.csv", rows, base_fields)
    print(f"comparison_dir={out_dir} files={len(detail_rows)}")
    for name, rows in buckets.items():
        print(f"{name}={len(rows)}")


if __name__ == "__main__":
    main()
