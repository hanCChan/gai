#!/usr/bin/env python3
"""Generate OSU LoRa Diff_Configurations_Setup manifests (24-class, exclude raw Device9)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from manifest_utils import (
    EXCLUDED_RAW_DEVICES,
    check_file,
    experiment_device,
    experiment_label,
    run_manifest_check,
    write_manifest,
)

# Val: high device ids from ONE source domain only; train keeps all 24 classes.
VAL_DEVICE_MIN = 19


def split_for_train_val(
    device: int,
    domain_split: str,
    domain_id: int,
    val_source_domain: int,
) -> str:
    """Within LOCO source domains, hold out devices>=19 from one source domain for val."""
    if domain_split == "test":
        return "test"
    if domain_split == "train" and domain_id == val_source_domain and device >= VAL_DEVICE_MIN:
        return "val"
    return "train"


CONFIG_SF = {1: 7, 2: 8, 3: 11, 4: 12}
SETUP = "diff_configurations"
REL_PREFIX = "Diff_Configurations_Setup"
FIELDNAMES = [
    "path",
    "relative_path",
    "split",
    "raw_device",
    "device",
    "label",
    "setup",
    "config",
    "sf",
    "scene",
    "file_index",
    "fold",
    "held_out_config",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Diff_Configurations manifests.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--data-root",
        default="data/raw/osu_lora",
        help="Root directory containing Diff_Configurations_Setup/",
    )
    parser.add_argument(
        "--out-all",
        default="data/manifest_configs_all.csv",
        help="Full inventory manifest (split=all).",
    )
    parser.add_argument(
        "--out-leave-one",
        default="data/manifest_configs_leave_one_config.csv",
        help="Leave-one-config-out manifest with fold column.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Do not run check_manifest.py after writing.",
    )
    return parser.parse_args()


def make_row(
    root: Path,
    data_root: Path,
    config: int,
    raw_device: int,
    split: str,
    fold: str = "",
    held_out_config: str = "",
) -> tuple[dict[str, str] | None, str | None]:
    if raw_device in EXCLUDED_RAW_DEVICES:
        return None, None

    rel = f"{REL_PREFIX}/Config{config}/IQ_{raw_device}.dat"
    path = data_root / rel
    ok, warning = check_file(path)
    if not ok:
        return None, warning

    device = experiment_device(raw_device)
    label = experiment_label(raw_device)
    rel_path = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    return (
        {
            "path": rel_path.replace("\\", "/"),
            "relative_path": rel,
            "split": split,
            "raw_device": str(raw_device),
            "device": str(device),
            "label": str(label),
            "setup": SETUP,
            "config": str(config),
            "sf": str(CONFIG_SF[config]),
            "scene": "indoor",
            "file_index": "1",
            "fold": fold,
            "held_out_config": held_out_config,
        },
        None,
    )


def build_all_rows(root: Path, data_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    for config in (1, 2, 3, 4):
        for raw_device in range(1, 26):
            row, warning = make_row(root, data_root, config, raw_device, split="all")
            if warning:
                warnings.append(warning)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda r: (int(r["config"]), int(r["device"])))
    return rows, warnings


def build_leave_one_rows(root: Path, data_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    for held_out in (1, 2, 3, 4):
        source_configs = [c for c in (1, 2, 3, 4) if c != held_out]
        val_source = max(source_configs)
        for config in (1, 2, 3, 4):
            domain_split = "test" if config == held_out else "train"
            for raw_device in range(1, 26):
                if raw_device in EXCLUDED_RAW_DEVICES:
                    continue
                device = experiment_device(raw_device)
                split = split_for_train_val(device, domain_split, config, val_source)
                row, warning = make_row(
                    root,
                    data_root,
                    config,
                    raw_device,
                    split=split,
                    fold=str(held_out),
                    held_out_config=str(held_out),
                )
                if warning:
                    warnings.append(warning)
                if row is not None:
                    rows.append(row)
    rows.sort(key=lambda r: (int(r["fold"]), r["split"] != "train", int(r["config"]), int(r["device"])))
    return rows, warnings


def print_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    unique = sorted(set(warnings))
    print(f"warnings={len(unique)}", file=sys.stderr)
    for warning in unique[:50]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if len(unique) > 50:
        print(f"WARNING: ... {len(unique) - 50} more", file=sys.stderr)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    data_root = (root / args.data_root).resolve()

    all_rows, all_warnings = build_all_rows(root, data_root)
    loo_rows, loo_warnings = build_leave_one_rows(root, data_root)
    print_warnings(all_warnings + loo_warnings)

    out_all = root / args.out_all
    out_loo = root / args.out_leave_one
    write_manifest(all_rows, out_all, FIELDNAMES)
    write_manifest(loo_rows, out_loo, FIELDNAMES)

    if args.skip_check:
        return 0

    rc_all = run_manifest_check(root, out_all)
    rc_loo = run_manifest_check(root, out_loo)
    return max(rc_all, rc_loo)


if __name__ == "__main__":
    raise SystemExit(main())
