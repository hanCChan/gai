#!/usr/bin/env python3
"""Shared helpers for OSU LoRa P1 manifest generation."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

EXCLUDED_RAW_DEVICES = {9}
RAW_DEVICE_IQ_PATTERN = re.compile(r"IQ_(\d+)\.dat")
RAW_DEVICE_DIR_PATTERN = re.compile(r"Device(\d+)")


def experiment_device(raw_device: int) -> int:
    if raw_device in EXCLUDED_RAW_DEVICES:
        raise ValueError(f"raw device {raw_device} is excluded")
    return raw_device - sum(1 for excluded in EXCLUDED_RAW_DEVICES if excluded < raw_device)


def experiment_label(raw_device: int) -> int:
    return experiment_device(raw_device) - 1


def raw_device_from_path(path: str) -> int | None:
    match = RAW_DEVICE_IQ_PATTERN.search(path)
    if match:
        return int(match.group(1))
    match = RAW_DEVICE_DIR_PATTERN.search(path)
    if match:
        return int(match.group(1))
    return None


def check_file(path: Path) -> tuple[bool, str | None]:
    if not path.exists():
        return False, f"missing file: {path}"
    size = path.stat().st_size
    if size <= 0:
        return False, f"empty file: {path} size={size}"
    return True, None


def write_manifest(rows: list[dict[str, str]], out_path: Path, fieldnames: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} rows={len(rows)}")


def run_manifest_check(root: Path, manifest: Path) -> int:
    cmd = [
        sys.executable,
        str(root / "scripts" / "check_manifest.py"),
        "--manifest",
        str(manifest.relative_to(root) if manifest.is_relative_to(root) else manifest),
        "--root",
        str(root),
        "--min-dat-bytes",
        "1",
    ]
    print(f"running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=root, check=False).returncode
