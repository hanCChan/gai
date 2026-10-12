from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DOMAIN_FIELDS = ("day", "receiver", "location", "distance", "sf", "config")


@dataclass(frozen=True)
class Row:
    path: Path
    device: int
    label: int
    setup: str
    split: str
    domains: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check OSU LoRa manifest integrity.")
    parser.add_argument("--manifest", default="data/manifest_all.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--min-dat-bytes", type=int, default=1)
    return parser.parse_args()


def to_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def load_rows(manifest: Path, root: Path) -> list[Row]:
    rows: list[Row] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            path = Path(item["path"])
            if not path.is_absolute():
                path = root / path
            rows.append(
                Row(
                    path=path,
                    device=to_int(item.get("device")),
                    label=to_int(item.get("label")),
                    setup=item.get("setup", "unknown"),
                    split=item.get("split", "unknown"),
                    domains={field: to_int(item.get(field)) for field in DOMAIN_FIELDS},
                )
            )
    return rows


def format_missing(values: set[int], expected: set[int]) -> str:
    missing = sorted(expected - values)
    return str(missing)


def check_files(rows: list[Row], min_dat_bytes: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for row in rows:
        if not row.path.exists():
            errors.append(f"missing dat: {row.path}")
            continue
        size = row.path.stat().st_size
        if size < min_dat_bytes:
            errors.append(f"abnormal dat size: {row.path} size={size}")
        meta = row.path.with_suffix(".sigmf-meta")
        if not meta.exists():
            warnings.append(f"missing sigmf-meta: {meta}")
    return errors, warnings


def print_setup_summary(rows: list[Row]) -> None:
    by_setup: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_setup[row.setup].append(row)

    for setup in sorted(by_setup):
        setup_rows = by_setup[setup]
        expected_devices = {row.device for row in setup_rows}
        print(f"\n[{setup}]")
        print(f"rows={len(setup_rows)} devices={len(expected_devices)} device_ids={sorted(expected_devices)}")

        for field in DOMAIN_FIELDS:
            by_value: dict[int, set[int]] = defaultdict(set)
            by_value_split: dict[tuple[int, str], set[int]] = defaultdict(set)
            for row in setup_rows:
                value = row.domains[field]
                by_value[value].add(row.device)
                by_value_split[(value, row.split)].add(row.device)
            if len(by_value) <= 1 and next(iter(by_value.keys()), 0) == 0:
                continue
            print(f"{field}:")
            for value in sorted(by_value):
                devices = by_value[value]
                missing = sorted(expected_devices - devices)
                suffix = f", missing={missing}" if missing else ""
                print(f"  {field}={value}: {len(devices)} devices{suffix}")
                for split in sorted({row.split for row in setup_rows}):
                    split_devices = by_value_split.get((value, split), set())
                    if split_devices:
                        split_missing = sorted(expected_devices - split_devices)
                        split_suffix = f", missing={split_missing}" if split_missing else ""
                        print(f"    split={split}: {len(split_devices)} devices{split_suffix}")


def print_label_summary(rows: list[Row]) -> None:
    labels = {row.label for row in rows}
    label_min = min(labels)
    label_max = max(labels)
    expected_labels = set(range(label_min, label_max + 1))
    print("")
    print(f"label_range={label_min}-{label_max}")
    print(f"num_classes={label_max + 1}")
    print(f"missing_labels_global={sorted(expected_labels - labels)}")

    by_split: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_split[row.split].add(row.label)
    parts = []
    for split in sorted(by_split):
        parts.append(f"{split}={sorted(expected_labels - by_split[split])}")
    print("missing_labels_by_split: " + ", ".join(parts))

    mismatches = []
    for row in rows:
        expected = row.device - 1
        if row.label != expected:
            mismatches.append((row.device, row.label, expected))
    if mismatches:
        print("label_mapping_warnings:")
        for device, label, expected in mismatches[:20]:
            print(f"  Device{device}: label={label}, expected={expected}")
        if len(mismatches) > 20:
            print(f"  ... {len(mismatches) - 20} more")


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest)
    root = Path(args.root)
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest}")
        return 2

    rows = load_rows(manifest, root)
    if not rows:
        print("ERROR: manifest has no rows")
        return 2

    print(f"manifest={manifest}")
    print(f"root={root.resolve()}")
    print(f"total_rows={len(rows)}")

    errors, warnings = check_files(rows, args.min_dat_bytes)
    print_setup_summary(rows)
    print_label_summary(rows)

    if warnings:
        print("")
        print(f"warnings={len(warnings)}")
        for warning in warnings[:50]:
            print(f"WARNING: {warning}")
        if len(warnings) > 50:
            print(f"WARNING: ... {len(warnings) - 50} more")

    if errors:
        print("")
        print(f"errors={len(errors)}")
        for error in errors[:50]:
            print(f"ERROR: {error}")
        if len(errors) > 50:
            print(f"ERROR: ... {len(errors) - 50} more")
        return 1

    print("")
    print("manifest_check=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

