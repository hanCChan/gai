#!/usr/bin/env python3
"""Audit all paper manifests: files, devices, windows, domain distribution."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WINDOW_SIZE = 8192
COMPLEX_DTYPE_SIZE = 8  # complex64


def count_windows(dat_path: Path) -> int:
    if not dat_path.exists():
        return 0
    nbytes = dat_path.stat().st_size
    samples = nbytes // COMPLEX_DTYPE_SIZE
    return max(0, samples // WINDOW_SIZE)


def audit_manifest(manifest: Path, root: Path) -> list[dict]:
    rows_out = []
    with manifest.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        by_split: dict[str, list[dict]] = defaultdict(list)
        for row in reader:
            by_split[row["split"]].append(row)

    for split, rows in sorted(by_split.items()):
        devices = {int(r["device"]) for r in rows}
        files = len(rows)
        total_windows = 0
        domain_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in rows:
            p = Path(r["path"])
            if not p.is_absolute():
                p = root / p
            total_windows += count_windows(p)
            for field in ("day", "receiver", "config", "location", "distance"):
                if field in r and r[field] not in ("", None):
                    domain_counts[field][str(r[field])] += 1
        rows_out.append(
            {
                "manifest": str(manifest.relative_to(root)),
                "protocol": rows[0].get("protocol", ""),
                "split": split,
                "num_files": files,
                "num_devices": len(devices),
                "total_windows_available": total_windows,
                "day_distribution": json.dumps(domain_counts.get("day", {})),
                "receiver_distribution": json.dumps(domain_counts.get("receiver", {})),
                "config_distribution": json.dumps(domain_counts.get("config", {})),
                "location_distribution": json.dumps(domain_counts.get("location", {})),
                "distance_distribution": json.dumps(domain_counts.get("distance", {})),
            }
        )
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="outputs/paper_ready/manifest_audit.csv")
    args = parser.parse_args()
    root = Path(args.root)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    manifests = sorted((root / "data" / "paper").rglob("*.csv"))
    if not manifests:
        print("No manifests in data/paper/. Run generate_paper_manifests.py first.")
        return

    all_rows = []
    for m in manifests:
        all_rows.extend(audit_manifest(m, root))

    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {out} rows={len(all_rows)}")


if __name__ == "__main__":
    main()
