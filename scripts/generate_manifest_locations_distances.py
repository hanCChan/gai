#!/usr/bin/env python3
"""Generate OSU LoRa Diff_Locations / Diff_Distances manifests (24-class, exclude raw Device9)."""

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

LOCATION_SCENE = {1: "room", 2: "office", 3: "outdoor"}
DISTANCE_METERS = {"5m": 5, "10m": 10, "15m": 15, "20m": 20}
VAL_DEVICE_MIN = 19


def split_for_train_val(
    device: int,
    domain_split: str,
    domain_id: int | str,
    val_source_domain: int | str,
) -> str:
    """Hold out devices>=19 from one source domain for val; train keeps all 24 classes."""
    if domain_split == "test":
        return "test"
    if domain_split == "train" and str(domain_id) == str(val_source_domain) and device >= VAL_DEVICE_MIN:
        return "val"
    return "train"


FIELDNAMES = [
    "path",
    "relative_path",
    "split",
    "raw_device",
    "device",
    "label",
    "setup",
    "location",
    "distance",
    "scene",
    "file_index",
    "fold",
    "held_out_location",
    "held_out_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate location/distance manifests.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--data-root",
        default="data/raw/osu_lora",
        help="Root directory containing Diff_Locations_Setup/ and Diff_Distances_Setup/",
    )
    parser.add_argument("--out-locations-all", default="data/manifest_locations_all.csv")
    parser.add_argument(
        "--out-locations-leave-one",
        default="data/manifest_locations_leave_one_location.csv",
    )
    parser.add_argument("--out-distances-all", default="data/manifest_distances_all.csv")
    parser.add_argument(
        "--out-distances-leave-one",
        default="data/manifest_distances_leave_one_distance.csv",
    )
    parser.add_argument("--skip-check", action="store_true")
    return parser.parse_args()


def location_row(
    root: Path,
    data_root: Path,
    location: int,
    raw_device: int,
    split: str,
    fold: str = "",
    held_out_location: str = "",
) -> tuple[dict[str, str] | None, str | None]:
    if raw_device in EXCLUDED_RAW_DEVICES:
        return None, None

    rel = f"Diff_Locations_Setup/Location{location}/IQ_{raw_device}.dat"
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
            "setup": "diff_locations",
            "location": str(location),
            "distance": "0",
            "scene": LOCATION_SCENE[location],
            "file_index": "1",
            "fold": fold,
            "held_out_location": held_out_location,
            "held_out_distance": "",
        },
        None,
    )


def distance_row(
    root: Path,
    data_root: Path,
    distance: str,
    raw_device: int,
    split: str,
    fold: str = "",
    held_out_distance: str = "",
) -> tuple[dict[str, str] | None, str | None]:
    if raw_device in EXCLUDED_RAW_DEVICES:
        return None, None

    rel = f"Diff_Distances_Setup/{distance}/IQ_{raw_device}.dat"
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
            "setup": "diff_distances",
            "location": "0",
            "distance": str(DISTANCE_METERS[distance]),
            "scene": "indoor",
            "file_index": "1",
            "fold": fold,
            "held_out_location": "",
            "held_out_distance": held_out_distance,
        },
        None,
    )


def build_location_manifests(
    root: Path, data_root: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    all_rows: list[dict[str, str]] = []
    loo_rows: list[dict[str, str]] = []
    warnings: list[str] = []

    for location in (1, 2, 3):
        for raw_device in range(1, 26):
            row, warning = location_row(root, data_root, location, raw_device, split="all")
            if warning:
                warnings.append(warning)
            if row is not None:
                all_rows.append(row)

    for held_out in (1, 2, 3):
        source_locs = [loc for loc in (1, 2, 3) if loc != held_out]
        val_source = max(source_locs)
        for location in (1, 2, 3):
            domain_split = "test" if location == held_out else "train"
            for raw_device in range(1, 26):
                if raw_device in EXCLUDED_RAW_DEVICES:
                    continue
                device = experiment_device(raw_device)
                split = split_for_train_val(device, domain_split, location, val_source)
                row, warning = location_row(
                    root,
                    data_root,
                    location,
                    raw_device,
                    split=split,
                    fold=str(held_out),
                    held_out_location=str(held_out),
                )
                if warning:
                    warnings.append(warning)
                if row is not None:
                    loo_rows.append(row)

    all_rows.sort(key=lambda r: (int(r["location"]), int(r["device"])))
    loo_rows.sort(
        key=lambda r: (int(r["fold"]), r["split"] != "train", int(r["location"]), int(r["device"]))
    )
    return all_rows, loo_rows, warnings


def build_distance_manifests(
    root: Path, data_root: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    all_rows: list[dict[str, str]] = []
    loo_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    distances = ("5m", "10m", "15m", "20m")

    for distance in distances:
        for raw_device in range(1, 26):
            row, warning = distance_row(root, data_root, distance, raw_device, split="all")
            if warning:
                warnings.append(warning)
            if row is not None:
                all_rows.append(row)

    for held_out in distances:
        source_dists = [d for d in distances if d != held_out]
        val_source = source_dists[-1]
        for distance in distances:
            domain_split = "test" if distance == held_out else "train"
            for raw_device in range(1, 26):
                if raw_device in EXCLUDED_RAW_DEVICES:
                    continue
                device = experiment_device(raw_device)
                split = split_for_train_val(device, domain_split, distance, val_source)
                row, warning = distance_row(
                    root,
                    data_root,
                    distance,
                    raw_device,
                    split=split,
                    fold=held_out,
                    held_out_distance=held_out,
                )
                if warning:
                    warnings.append(warning)
                if row is not None:
                    loo_rows.append(row)

    all_rows.sort(key=lambda r: (int(r["distance"]), int(r["device"])))
    loo_rows.sort(
        key=lambda r: (
            distances.index(r["fold"]) if r["fold"] in distances else 99,
            r["split"] != "train",
            int(r["distance"]),
            int(r["device"]),
        )
    )
    return all_rows, loo_rows, warnings


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

    loc_all, loc_loo, loc_warnings = build_location_manifests(root, data_root)
    dist_all, dist_loo, dist_warnings = build_distance_manifests(root, data_root)
    print_warnings(loc_warnings + dist_warnings)

    outputs = [
        (loc_all, root / args.out_locations_all),
        (loc_loo, root / args.out_locations_leave_one),
        (dist_all, root / args.out_distances_all),
        (dist_loo, root / args.out_distances_leave_one),
    ]
    for rows, out_path in outputs:
        write_manifest(rows, out_path, FIELDNAMES)

    if args.skip_check:
        return 0

    rc = 0
    for _, out_path in outputs:
        rc = max(rc, run_manifest_check(root, out_path))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
