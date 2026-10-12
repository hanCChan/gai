#!/usr/bin/env python3
"""Drop unavailable devices from manifests and remap labels to 0..N-1."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

EXCLUDED_RAW_DEVICES = {9}
DEVICE_PATTERN = re.compile(r"Device(\d+)")


def remap_device(raw_device: int) -> int:
    if raw_device in EXCLUDED_RAW_DEVICES:
        raise ValueError(f"device {raw_device} is excluded")
    return raw_device - sum(1 for excluded in EXCLUDED_RAW_DEVICES if excluded < raw_device)


def remap_label(raw_device: int) -> int:
    if raw_device in EXCLUDED_RAW_DEVICES:
        raise ValueError(f"device {raw_device} is excluded")
    return remap_device(raw_device) - 1


def raw_device_from_row(row: dict[str, str]) -> int:
    for key in ("relative_path", "path"):
        match = DEVICE_PATTERN.search(row.get(key, ""))
        if match:
            return int(match.group(1))
    return int(row["device"])


def fix_manifest(path: Path, setup: str | None = None) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    kept: list[dict[str, str]] = []
    dropped = 0
    for row in rows:
        raw_device = raw_device_from_row(row)
        if setup is not None and row.get("setup") != setup:
            kept.append(row)
            continue
        if raw_device in EXCLUDED_RAW_DEVICES:
            dropped += 1
            continue
        row = dict(row)
        row["device"] = str(remap_device(raw_device))
        row["label"] = str(remap_label(raw_device))
        kept.append(row)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(kept)

    return dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Exclude unavailable devices from manifest CSV files.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--setup",
        default="diff_days_indoor",
        help="Only remap rows for this setup. Use empty string to remap all rows.",
    )
    args = parser.parse_args()
    root = Path(args.root)
    setup = args.setup or None

    targets = [
        root / "data/manifest_cross_day_day1_day2.csv",
        root / "data/manifest_cross_day_day1_day2.csv.bak",
        root / "data/manifest_cross_day_day1_to_day5.csv",
        root / "data/manifest_days_iq1_day1_to_day5.csv",
        root / "data/manifest_all.csv",
        root / "data/manifest_all.csv.bak",
    ]
    for path in targets:
        if not path.exists():
            continue
        dropped = fix_manifest(path, setup=setup)
        print(f"updated {path} dropped={dropped}")


if __name__ == "__main__":
    main()
