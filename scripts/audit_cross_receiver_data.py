#!/usr/bin/env python3
"""Task 1: audit cross-receiver raw data.

For RX1/RX2 under Diff_Receivers_Setup_Indoor_SameTx, list every raw DeviceX:
file presence, byte size, #complex samples, #windows, sigmf-meta presence.
Uses the same 24-class mapping (exclude raw Device9, device 1..24, label 0..23).

Output: outputs/cross_receiver_analysis/data_audit.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path("/data1/hcc/llm4RF")
SUBSET = ROOT / "data/raw/osu_lora/Diff_Receivers_Setup/Diff_Receivers_Setup_Indoor_SameTx"
RECEIVERS = ["RX1", "RX2"]
EXCLUDED_RAW_DEVICES = {9}
WINDOW_SIZE = 8192
BYTES_PER_SAMPLE = np.dtype(np.complex64).itemsize  # 8
OUT = ROOT / "outputs/cross_receiver_analysis/data_audit.csv"


def experiment_device(raw_device: int) -> int:
    return raw_device - sum(1 for ex in EXCLUDED_RAW_DEVICES if ex < raw_device)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rx in RECEIVERS:
        for raw_device in range(1, 26):
            excluded = raw_device in EXCLUDED_RAW_DEVICES
            dat = SUBSET / rx / f"Device{raw_device}_IQ.dat"
            meta = SUBSET / rx / f"Device{raw_device}_IQ.sigmf-meta"
            dat_exists = dat.exists()
            dat_bytes = dat.stat().st_size if dat_exists else 0
            n_samples = dat_bytes // BYTES_PER_SAMPLE
            n_windows = n_samples // WINDOW_SIZE
            row = {
                "receiver": rx,
                "raw_device": raw_device,
                "excluded": int(excluded),
                "exp_device": "" if excluded else experiment_device(raw_device),
                "label": "" if excluded else experiment_device(raw_device) - 1,
                "dat_exists": int(dat_exists),
                "dat_bytes": dat_bytes,
                "n_complex_samples": n_samples,
                "n_windows_8192": n_windows,
                "meta_exists": int(meta.exists()),
                "dat_path": str(dat.relative_to(ROOT)) if dat_exists else "",
            }
            rows.append(row)

    fieldnames = [
        "receiver", "raw_device", "excluded", "exp_device", "label",
        "dat_exists", "dat_bytes", "n_complex_samples", "n_windows_8192",
        "meta_exists", "dat_path",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    used = [r for r in rows if not r["excluded"]]
    n_files_per_dev = 1  # one .dat per (receiver, device) in this subset
    print(f"wrote {OUT} rows={len(rows)}")
    print(f"used (non-excluded) entries={len(used)} -> RX1+RX2 = {len(used)} files, {len(used)//2} devices x 2 receivers")
    print(f"files per (receiver,device) = {n_files_per_dev}")
    miss = [r for r in used if not r["dat_exists"]]
    print(f"missing .dat among used: {len(miss)}")
    if used:
        nw = [r["n_windows_8192"] for r in used]
        print(f"n_windows_8192: min={min(nw)} max={max(nw)} (window=8192, sr=1MHz => ~{min(nw)*8192/1e6:.2f}s min per file)")


if __name__ == "__main__":
    main()
