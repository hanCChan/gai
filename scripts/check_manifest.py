from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_RAW_DEVICES = {9}
EXPECTED_NUM_CLASSES = 24
EXPECTED_LABELS = set(range(EXPECTED_NUM_CLASSES))
EXPECTED_DEVICES = set(range(1, EXPECTED_NUM_CLASSES + 1))
RAW_DEVICE_IQ_PATTERN = re.compile(r"IQ_(\d+)\.dat")
RAW_DEVICE_DIR_PATTERN = re.compile(r"Device(\d+)")

DOMAIN_FIELDS = ("day", "receiver", "location", "distance", "sf", "config")
GROUP_COUNT_FIELDS = ("setup", "config", "location", "distance")


@dataclass(frozen=True)
class Row:
    path: Path
    raw_path: str
    raw_device: int | None
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
    parser.add_argument(
        "--expected-num-classes",
        type=int,
        default=EXPECTED_NUM_CLASSES,
        help="Expected number of classes (default: 24 -> labels 0..23).",
    )
    return parser.parse_args()


def to_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def infer_raw_device(item: dict[str, str]) -> int | None:
    raw_value = item.get("raw_device")
    if raw_value not in (None, ""):
        return int(raw_value)
    for key in ("relative_path", "path"):
        text = item.get(key, "")
        match = RAW_DEVICE_DIR_PATTERN.search(text)
        if match:
            return int(match.group(1))
        match = RAW_DEVICE_IQ_PATTERN.search(text)
        if match:
            return int(match.group(1))
    return None


def load_rows(manifest: Path, root: Path) -> list[Row]:
    rows: list[Row] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            raw_path = item["path"]
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            rows.append(
                Row(
                    path=path,
                    raw_path=raw_path,
                    raw_device=infer_raw_device(item),
                    device=to_int(item.get("device")),
                    label=to_int(item.get("label")),
                    setup=item.get("setup", "unknown"),
                    split=item.get("split", "unknown"),
                    domains={field: to_int(item.get(field)) for field in DOMAIN_FIELDS},
                )
            )
    return rows


def check_files(rows: list[Row], min_dat_bytes: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for row in rows:
        if not row.path.exists():
            errors.append(f"missing dat: {row.path}")
            continue
        size = row.path.stat().st_size
        if size <= 0:
            errors.append(f"empty dat file: {row.path} size={size}")
        elif size < min_dat_bytes:
            errors.append(f"abnormal dat size: {row.path} size={size} min={min_dat_bytes}")
        meta = row.path.with_suffix(".sigmf-meta")
        if not meta.exists():
            warnings.append(f"missing sigmf-meta: {meta}")
    return errors, warnings


def check_raw_device_rules(rows: list[Row]) -> list[str]:
    errors: list[str] = []
    mapping: dict[int, tuple[int, int]] = {}

    for row in rows:
        if row.raw_device is None:
            continue
        if row.raw_device in EXCLUDED_RAW_DEVICES:
            errors.append(
                f"raw Device{row.raw_device} must be excluded but appears in manifest: {row.raw_path}"
            )
            continue

        expected_device = row.raw_device - sum(
            1 for excluded in EXCLUDED_RAW_DEVICES if excluded < row.raw_device
        )
        expected_label = expected_device - 1
        if row.device != expected_device or row.label != expected_label:
            errors.append(
                f"raw Device{row.raw_device} maps to device={row.device}, label={row.label}, "
                f"expected device={expected_device}, label={expected_label} ({row.raw_path})"
            )

        prior = mapping.get(row.raw_device)
        if prior is None:
            mapping[row.raw_device] = (row.device, row.label)
        elif prior != (row.device, row.label):
            errors.append(
                f"inconsistent mapping for raw Device{row.raw_device}: "
                f"saw device/label={prior}, also device/label={(row.device, row.label)}"
            )

    return errors


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
                print(f"  {field}={value}: class_count={len(devices)}{suffix}")
                for split in sorted({row.split for row in setup_rows}):
                    split_devices = by_value_split.get((value, split), set())
                    if split_devices:
                        split_missing = sorted(expected_devices - split_devices)
                        split_suffix = f", missing={split_missing}" if split_missing else ""
                        print(
                            f"    split={split}: class_count={len(split_devices)}{split_suffix}"
                        )


def print_group_class_counts(rows: list[Row]) -> None:
    grouped: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for row in rows:
        for field in ("config", "location", "distance"):
            value = row.domains.get(field, 0)
            if value == 0 and field == "distance":
                continue
            if value == 0:
                continue
            key = (row.setup, field, str(value))
            grouped[key].add(row.device)

    if not grouped:
        return

    print("\nclass_count_by_setup_domain:")
    for key in sorted(grouped):
        setup, field, value = key
        devices = grouped[key]
        missing = sorted(EXPECTED_DEVICES - devices)
        suffix = f", missing_devices={missing}" if missing else ""
        print(f"  setup={setup} {field}={value}: class_count={len(devices)}{suffix}")


def check_label_summary(rows: list[Row], expected_num_classes: int) -> list[str]:
    errors: list[str] = []
    expected_labels = set(range(expected_num_classes))
    expected_devices = set(range(1, expected_num_classes + 1))

    labels = {row.label for row in rows}
    label_min = min(labels) if labels else -1
    label_max = max(labels) if labels else -1

    print("")
    print(f"label_range={label_min}-{label_max}")
    print(f"num_classes={expected_num_classes}")
    print(f"expected_label_range=0-{expected_num_classes - 1}")

    out_of_range = sorted(label for label in labels if label < 0 or label >= expected_num_classes)
    if out_of_range:
        errors.append(f"labels outside 0..{expected_num_classes - 1}: {out_of_range}")

    missing_global = sorted(expected_labels - labels)
    print(f"missing_labels={missing_global}")
    if missing_global:
        errors.append(f"missing labels: {missing_global}")

    if label_min != 0:
        errors.append(f"labels must start at 0, got {label_min}")
    if label_max != expected_num_classes - 1 and labels:
        errors.append(f"labels must end at {expected_num_classes - 1}, got {label_max}")

    by_split: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_split[row.split].add(row.label)
    parts = []
    for split in sorted(by_split):
        missing_split = sorted(expected_labels - by_split[split])
        parts.append(f"{split}={missing_split}")
    print("missing_labels_by_split: " + ", ".join(parts))

    devices = {row.device for row in rows}
    missing_devices = sorted(expected_devices - devices)
    extra_devices = sorted(devices - expected_devices)
    print(f"device_range=1-{max(devices) if devices else 0}")
    print(f"missing_experiment_devices={missing_devices}")
    print(f"extra_experiment_devices={extra_devices}")
    if missing_devices or extra_devices:
        errors.append(
            f"experiment device ids must be contiguous 1..{expected_num_classes}, "
            f"missing={missing_devices}, extra={extra_devices}"
        )

    mismatches = []
    for row in rows:
        expected = row.device - 1
        if row.label != expected:
            mismatches.append((row.device, row.label, expected, row.raw_path))
    if mismatches:
        print("label_mapping_warnings:")
        for device, label, expected, raw_path in mismatches[:20]:
            print(f"  device={device}: label={label}, expected={expected}, path={raw_path}")
        if len(mismatches) > 20:
            print(f"  ... {len(mismatches) - 20} more")
        errors.append(f"device/label mapping mismatches: {len(mismatches)}")

    return errors


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
    print_group_class_counts(rows)
    errors.extend(check_raw_device_rules(rows))
    errors.extend(check_label_summary(rows, args.expected_num_classes))

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
