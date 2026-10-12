#!/usr/bin/env python3
"""Generate OSU LoRa cross-receiver manifests (RX1<->RX2, Indoor SameTx).

Excludes raw Device9, remaps to 24 contiguous classes (device 1..24, label 0..23).
Produces two manifests: RX1 train / RX2 val, and RX2 train / RX1 val.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

EXCLUDED_RAW_DEVICES = {9}
SUBSET = "Diff_Receivers_Setup/Diff_Receivers_Setup_Indoor_SameTx"
RX_RECEIVER_ID = {"RX1": 1, "RX2": 2}

FIELDNAMES = [
    "path", "relative_path", "device", "label",
    "day", "receiver", "location", "distance", "sf", "scene", "config",
    "setup", "split",
]


def experiment_device(raw_device: int) -> int:
    if raw_device in EXCLUDED_RAW_DEVICES:
        raise ValueError(f"device {raw_device} is excluded")
    return raw_device - sum(1 for excluded in EXCLUDED_RAW_DEVICES if excluded < raw_device)


def make_row(rx: str, raw_device: int, split: str) -> dict[str, str]:
    device = experiment_device(raw_device)
    rel = f"{SUBSET}/{rx}/Device{raw_device}_IQ.dat"
    return {
        "path": f"data/raw/osu_lora/{rel}",
        "relative_path": rel,
        "device": str(device),
        "label": str(device - 1),
        "day": "0",
        "receiver": str(RX_RECEIVER_ID[rx]),
        "location": "0",
        "distance": "0",
        "sf": "0",
        "scene": "indoor",
        "config": "0",
        "setup": "diff_receivers_indoor_sametx",
        "split": split,
    }


def build(train_rx: str, val_rx: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_device in range(1, 26):
        if raw_device in EXCLUDED_RAW_DEVICES:
            continue
        rows.append(make_row(train_rx, raw_device, "train"))
        rows.append(make_row(val_rx, raw_device, "val"))
    rows.sort(key=lambda r: (r["split"] != "train", int(r["device"])))
    return rows


def write_manifest(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    print(f"wrote {out_path} rows={len(rows)} train={n_train} val={n_val}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cross-receiver manifests.")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    write_manifest(build("RX1", "RX2"), root / "data" / "manifest_rx1_to_rx2.csv")
    write_manifest(build("RX2", "RX1"), root / "data" / "manifest_rx2_to_rx1.csv")


if __name__ == "__main__":
    main()
