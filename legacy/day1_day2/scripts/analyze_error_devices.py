from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ERROR_DEVICES = {0, 9, 10, 14, 15, 17, 20, 23, 24}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cross-model file-level error devices.")
    parser.add_argument("--pred-a", default="outputs/osu_cnn_day1_day2_spf256/osu_cnn_iq/classifier/predictions.csv")
    parser.add_argument("--pred-b", default="outputs/cross_day_hybrid_cnnstem_spf256/rfhstu_cnnstem_cross_attn_chirp/classifier/predictions.csv")
    parser.add_argument("--a-name", default="osu_cnn_iq")
    parser.add_argument("--b-name", default="hybrid")
    parser.add_argument("--out-dir", default="outputs/error_device_analysis")
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def aggregate_files(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["file_path"]].append(row)
    out = {}
    for file_path, items in grouped.items():
        label = to_int(items[0].get("label"))
        pred = Counter(to_int(row.get("pred")) for row in items).most_common(1)[0][0]
        record = {
            "file_path": file_path,
            "label": label,
            "device_name": f"Device{label + 1}",
            "pred": pred,
            "pred_device_name": f"Device{pred + 1}",
            "correct": int(pred == label),
            "num_windows": len(items),
        }
        for key in ["split", "setup", "day", "receiver", "location", "distance", "sf", "config"]:
            if key in items[0]:
                record[key] = items[0][key]
        out[file_path] = record
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def confusion_targets(files: dict[str, dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    by_pair: Counter[tuple[int, int]] = Counter()
    for row in files.values():
        label = to_int(row["label"])
        pred = to_int(row["pred"])
        if label in ERROR_DEVICES and pred != label:
            by_pair[(label, pred)] += 1
    out = []
    for (label, pred), count in sorted(by_pair.items()):
        out.append(
            {
                "model": model_name,
                "label": label,
                "device_name": f"Device{label + 1}",
                "pred": pred,
                "pred_device_name": f"Device{pred + 1}",
                "num_files": count,
            }
        )
    return out


def device_rows(records: list[dict[str, Any]], a_name: str, b_name: str) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_label[to_int(row["label"])].append(row)
    out = []
    for label in sorted(by_label):
        items = by_label[label]
        out.append(
            {
                "label": label,
                "device_name": f"Device{label + 1}",
                "num_files": len(items),
                f"{a_name}_preds": ";".join(str(row[f"{a_name}_pred"]) for row in items),
                f"{b_name}_preds": ";".join(str(row[f"{b_name}_pred"]) for row in items),
                "files": ";".join(row["file_path"] for row in items),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    a_files = aggregate_files(read_rows(args.pred_a))
    b_files = aggregate_files(read_rows(args.pred_b))
    common = sorted(set(a_files) & set(b_files))

    comparisons = []
    buckets = {
        "both_wrong": [],
        "cnn_wrong_hybrid_right": [],
        "hybrid_wrong_cnn_right": [],
    }
    for file_path in common:
        a = a_files[file_path]
        b = b_files[file_path]
        row = {
            "file_path": file_path,
            "label": a["label"],
            "device_name": a["device_name"],
            f"{args.a_name}_pred": a["pred"],
            f"{args.a_name}_correct": a["correct"],
            f"{args.b_name}_pred": b["pred"],
            f"{args.b_name}_correct": b["correct"],
        }
        comparisons.append(row)
        if not a["correct"] and not b["correct"]:
            buckets["both_wrong"].append(row)
        elif not a["correct"] and b["correct"]:
            buckets["cnn_wrong_hybrid_right"].append(row)
        elif a["correct"] and not b["correct"]:
            buckets["hybrid_wrong_cnn_right"].append(row)

    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "confusion_targets_by_device.csv",
        [*confusion_targets(a_files, args.a_name), *confusion_targets(b_files, args.b_name)],
        ["model", "label", "device_name", "pred", "pred_device_name", "num_files"],
    )
    summary_fields = ["label", "device_name", "num_files", f"{args.a_name}_preds", f"{args.b_name}_preds", "files"]
    write_csv(out_dir / "both_wrong_by_device.csv", device_rows(buckets["both_wrong"], args.a_name, args.b_name), summary_fields)
    write_csv(
        out_dir / "cnn_wrong_hybrid_right_by_device.csv",
        device_rows(buckets["cnn_wrong_hybrid_right"], args.a_name, args.b_name),
        summary_fields,
    )
    write_csv(
        out_dir / "hybrid_wrong_cnn_right_by_device.csv",
        device_rows(buckets["hybrid_wrong_cnn_right"], args.a_name, args.b_name),
        summary_fields,
    )
    print(f"out_dir={out_dir}")
    for key, rows in buckets.items():
        print(f"{key}={len(rows)}")


if __name__ == "__main__":
    main()
