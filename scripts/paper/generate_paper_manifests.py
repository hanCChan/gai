#!/usr/bin/env python3
"""Generate IoTJ paper manifests: source-only + oracle diagnostic splits."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

EXCLUDED_RAW_DEVICES = {9}
DAY_FIELDNAMES = [
    "path", "relative_path", "device", "label", "day", "receiver", "location",
    "distance", "sf", "scene", "config", "setup", "split", "protocol",
]
RX_FIELDNAMES = DAY_FIELDNAMES


def experiment_device(raw_device: int) -> int:
    if raw_device in EXCLUDED_RAW_DEVICES:
        raise ValueError(f"device {raw_device} excluded")
    return raw_device - sum(1 for x in EXCLUDED_RAW_DEVICES if x < raw_device)


def day_row(day: int, raw_device: int, split: str, protocol: str) -> dict[str, str]:
    device = experiment_device(raw_device)
    rel = f"Diff_Days_Indoor_Setup/Day{day}/Device{raw_device}/IQ_1.dat"
    return {
        "path": f"data/raw/osu_lora/{rel}",
        "relative_path": rel,
        "device": str(device),
        "label": str(device - 1),
        "day": str(day),
        "receiver": "0",
        "location": "0",
        "distance": "0",
        "sf": "0",
        "scene": "indoor",
        "config": "0",
        "setup": "diff_days_indoor",
        "split": split,
        "protocol": protocol,
    }


def rx_row(rx: str, raw_device: int, split: str, protocol: str) -> dict[str, str]:
    device = experiment_device(raw_device)
    rx_id = {"RX1": 1, "RX2": 2}[rx]
    rel = f"Diff_Receivers_Setup/Diff_Receivers_Setup_Indoor_SameTx/{rx}/Device{raw_device}_IQ.dat"
    return {
        "path": f"data/raw/osu_lora/{rel}",
        "relative_path": rel,
        "device": str(device),
        "label": str(device - 1),
        "day": "0",
        "receiver": str(rx_id),
        "location": "0",
        "distance": "0",
        "sf": "0",
        "scene": "indoor",
        "config": "0",
        "setup": "diff_receivers_indoor_sametx",
        "split": split,
        "protocol": protocol,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} rows={len(rows)}")


def day_manifest(days_train: list[int], days_val: list[int], days_test: list[int], protocol: str) -> list[dict]:
    rows = []
    for day in days_train:
        for raw in range(1, 26):
            if raw not in EXCLUDED_RAW_DEVICES:
                rows.append(day_row(day, raw, "train", protocol))
    for day in days_val:
        for raw in range(1, 26):
            if raw not in EXCLUDED_RAW_DEVICES:
                rows.append(day_row(day, raw, "val", protocol))
    for day in days_test:
        for raw in range(1, 26):
            if raw not in EXCLUDED_RAW_DEVICES:
                rows.append(day_row(day, raw, "test", protocol))
    return rows


def rx_manifest(train_rx: str, val_rx: str, test_rx: str, protocol: str) -> list[dict]:
    rows = []
    for raw in range(1, 26):
        if raw in EXCLUDED_RAW_DEVICES:
            continue
        rows.append(rx_row(train_rx, raw, "train", protocol))
        if val_rx:
            rows.append(rx_row(val_rx, raw, "val", protocol))
        if test_rx:
            rows.append(rx_row(test_rx, raw, "test", protocol))
    return rows


def lodo_source_only(test_day: int) -> list[dict]:
    all_days = [1, 2, 3, 4, 5]
    remaining = [d for d in all_days if d != test_day]
    val_day = max(remaining)
    train_days = [d for d in remaining if d != val_day]
    return day_manifest(train_days, [val_day], [test_day], f"lodo_source_only_test_day_{test_day}")


def run_check(root: Path, manifest: Path) -> None:
    subprocess.run(
        [sys.executable, str(root / "scripts" / "check_manifest.py"), "--manifest", str(manifest), "--root", str(root)],
        cwd=root,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    out = root / "data" / "paper"

    # --- Cross-day: Day1-4 -> Day5 ---
    write_csv(
        out / "cross_day_day1to5_source_only.csv",
        day_manifest([1, 2, 3], [4], [5], "cross_day_source_only"),
        DAY_FIELDNAMES,
    )
    write_csv(
        out / "cross_day_day1to5_oracle_target_val.csv",
        day_manifest([1, 2, 3, 4], [5], [5], "oracle_target_val"),
        DAY_FIELDNAMES,
    )

    # --- LODO source-only folds ---
    lodo_dir = out / "lodo_source_only"
    lodo_dir.mkdir(parents=True, exist_ok=True)
    for day in range(1, 6):
        write_csv(lodo_dir / f"test_day_{day}.csv", lodo_source_only(day), DAY_FIELDNAMES)

    # --- Cross-receiver ---
    # Source-only: RX1 train+val (same receiver), RX2 test
    write_csv(
        out / "rx1_to_rx2_source_only.csv",
        rx_manifest("RX1", "RX1", "RX2", "cross_receiver_source_only"),
        RX_FIELDNAMES,
    )
    write_csv(
        out / "rx2_to_rx1_source_only.csv",
        rx_manifest("RX2", "RX2", "RX1", "cross_receiver_source_only"),
        RX_FIELDNAMES,
    )
    # Oracle diagnostic (legacy): target receiver val for checkpoint
    write_csv(
        out / "rx1_to_rx2_oracle_target_val.csv",
        rx_manifest("RX1", "RX2", "RX2", "oracle_target_val"),
        RX_FIELDNAMES,
    )
    write_csv(
        out / "rx2_to_rx1_oracle_target_val.csv",
        rx_manifest("RX2", "RX1", "RX1", "oracle_target_val"),
        RX_FIELDNAMES,
    )
    # Upper bound: same receiver train/test
    write_csv(
        out / "rx1_to_rx1_upper_bound.csv",
        rx_manifest("RX1", "RX1", "RX1", "upper_bound_same_receiver"),
        RX_FIELDNAMES,
    )
    write_csv(
        out / "rx2_to_rx2_upper_bound.csv",
        rx_manifest("RX2", "RX2", "RX2", "upper_bound_same_receiver"),
        RX_FIELDNAMES,
    )

    # Run checks on key manifests
    for name in [
        "cross_day_day1to5_source_only.csv",
        "rx1_to_rx2_source_only.csv",
        "lodo_source_only/test_day_5.csv",
    ]:
        run_check(root, out / name)

    print(f"\nAll paper manifests written under {out}")


if __name__ == "__main__":
    main()
