#!/usr/bin/env python3
"""Generate OSU LoRa day manifests with unavailable raw devices excluded."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

EXCLUDED_RAW_DEVICES = {9}
DEVICE_PATTERN = re.compile(r"Device(\d+)")


def experiment_device(raw_device: int) -> int:
    if raw_device in EXCLUDED_RAW_DEVICES:
        raise ValueError(f"device {raw_device} is excluded")
    return raw_device - sum(1 for excluded in EXCLUDED_RAW_DEVICES if excluded < raw_device)


def raw_device_from_row(row: dict[str, str]) -> int:
    for key in ("relative_path", "path"):
        match = DEVICE_PATTERN.search(row.get(key, ""))
        if match:
            return int(match.group(1))
    return int(row["device"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manifest rows for additional OSU LoRa days.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--days", default="3,4,5", help="Comma-separated day indices.")
    parser.add_argument("--split", default="extra", help="Split label for new day rows.")
    parser.add_argument(
        "--train-days",
        default="1,2,3,4",
        help="If set, generate a clean manifest from scratch with these days as train.",
    )
    parser.add_argument(
        "--val-days",
        default="5",
        help="If set, generate a clean manifest from scratch with these days as val.",
    )
    parser.add_argument(
        "--out",
        default="data/manifest_days_iq1_day1_to_day5.csv",
        help="Output manifest path.",
    )
    parser.add_argument(
        "--base-manifest",
        default="data/manifest_all.csv",
        help="Existing manifest to extend.",
    )
    return parser.parse_args()


def day_rows(day: int, raw_device: int, split: str) -> dict[str, str]:
    device = experiment_device(raw_device)
    rel = f"Diff_Days_Indoor_Setup/Day{day}/Device{raw_device}/IQ_1.dat"
    path = f"data/raw/osu_lora/{rel}"
    return {
        "path": path,
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
    }


def remap_existing_row(row: dict[str, str]) -> dict[str, str] | None:
    raw_device = raw_device_from_row(row)
    if raw_device in EXCLUDED_RAW_DEVICES:
        return None
    out = dict(row)
    device = experiment_device(raw_device)
    out["device"] = str(device)
    out["label"] = str(device - 1)
    return out


def day_rows_for_day(day: int, split: str) -> list[dict[str, str]]:
    rows = []
    for raw_device in range(1, 26):
        if raw_device in EXCLUDED_RAW_DEVICES:
            continue
        rows.append(day_rows(day, raw_device, split))
    return rows


def parse_days(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    base_manifest = root / args.base_manifest
    out_path = root / args.out
    fieldnames = [
        "path",
        "relative_path",
        "device",
        "label",
        "day",
        "receiver",
        "location",
        "distance",
        "sf",
        "scene",
        "config",
        "setup",
        "split",
    ]

    train_days = parse_days(args.train_days)
    val_days = parse_days(args.val_days)

    rows: list[dict[str, str]] = []
    if train_days or val_days:
        for day in train_days:
            rows.extend(day_rows_for_day(day, "train"))
        for day in val_days:
            rows.extend(day_rows_for_day(day, "val"))
    elif base_manifest.exists():
        with base_manifest.open("r", encoding="utf-8-sig", newline="") as f:
            for item in csv.DictReader(f):
                if item.get("setup") == "diff_days_indoor":
                    row = remap_existing_row(item)
                    if row is not None:
                        rows.append(row)

        existing = {(int(r["day"]), int(r["device"])) for r in rows if r.get("day") and r.get("device")}
        for day_str in args.days.split(","):
            day = int(day_str.strip())
            for raw_device in range(1, 26):
                if raw_device in EXCLUDED_RAW_DEVICES:
                    continue
                device = experiment_device(raw_device)
                if (day, device) in existing:
                    continue
                rows.append(day_rows(day, raw_device, args.split))

    rows.sort(key=lambda r: (int(r["day"]), int(r["device"])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
