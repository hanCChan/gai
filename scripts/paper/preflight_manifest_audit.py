#!/usr/bin/env python3
"""Preflight manifest/split audit for paper experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

NUM_CLASSES = 24
ALL_LABELS = set(range(NUM_CLASSES))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def label_set(rows: list[dict[str, str]]) -> set[int]:
    return {int(r["label"]) for r in rows}


def domain_values(rows: list[dict[str, str]], field: str) -> set[str]:
    return {str(r[field]) for r in rows if field in r and r[field] not in ("", None)}


def paths(rows: list[dict[str, str]]) -> set[str]:
    return {r["path"] for r in rows}


def path_hash(sorted_paths: list[str]) -> str:
    payload = "\n".join(sorted_paths).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def missing_labels(labels: set[int]) -> str:
    missing = sorted(ALL_LABELS - labels)
    return ",".join(str(x) for x in missing)


def overlap(a: set[str], b: set[str]) -> int:
    return len(a & b)


def audit_fold(
    manifest: Path,
    fold: str | None,
    protocol: str,
    held_out_field: str | None,
    held_out_value: str | None,
    source_domain_field: str | None,
) -> tuple[list[dict], list[dict], bool]:
    """Return (detail_rows, hash_rows, pass_flag)."""
    root_rows = read_manifest(manifest)
    if fold is None or fold == "":
        fold_rows = root_rows
        fold_key = "all"
    else:
        fold_rows = [r for r in root_rows if str(r.get("fold", "")) == str(fold)]
        fold_key = str(fold)

    detail_rows: list[dict] = []
    hash_rows: list[dict] = []
    fold_pass = True

    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in fold_rows:
        by_split[r["split"]].append(r)

    train_rows = by_split.get("train", [])
    val_rows = by_split.get("val", [])
    test_rows = by_split.get("test", [])

    train_paths = paths(train_rows)
    val_paths = paths(val_rows)
    test_paths = paths(test_rows)

    train_labels = label_set(train_rows)
    val_labels = label_set(val_rows)
    test_labels = label_set(test_rows)

    train_missing = ALL_LABELS - train_labels
    if train_missing:
        fold_pass = False

    # val must be source domain only
    val_from_source_ok = True
    val_domain_note = ""
    if held_out_field and held_out_value:
        val_domains = domain_values(val_rows, held_out_field)
        if held_out_value in val_domains or (val_domains & {held_out_value}):
            val_from_source_ok = False
        val_domain_note = json.dumps(sorted(val_domains))
        if not val_from_source_ok:
            fold_pass = False

    # test must be held-out/target only
    test_target_ok = True
    test_domain_note = ""
    if held_out_field and held_out_value:
        test_domains = domain_values(test_rows, held_out_field)
        if test_domains != {held_out_value}:
            test_target_ok = False
        test_domain_note = json.dumps(sorted(test_domains))
        if not test_target_ok:
            fold_pass = False

    # cross_day special case
    if protocol == "cross_day_source_only":
        val_days = domain_values(val_rows, "day")
        test_days = domain_values(test_rows, "day")
        train_days = domain_values(train_rows, "day")
        val_from_source_ok = val_days == {"4"} and val_days.isdisjoint({"5"})
        test_target_ok = test_days == {"5"}
        val_domain_note = json.dumps({"day": sorted(val_days)})
        test_domain_note = json.dumps({"day": sorted(test_days)})
        if not val_from_source_ok or not test_target_ok:
            fold_pass = False

    path_overlap_tv = overlap(train_paths, val_paths)
    path_overlap_tt = overlap(train_paths, test_paths)
    path_overlap_vt = overlap(val_paths, test_paths)
    if path_overlap_tv or path_overlap_tt or path_overlap_vt:
        fold_pass = False

    test_hash = path_hash(sorted(test_paths)) if test_paths else ""
    hash_rows.append(
        {
            "manifest": str(manifest),
            "protocol": protocol,
            "fold": fold_key,
            "held_out_field": held_out_field or "",
            "held_out_value": held_out_value or "",
            "num_test_files": len(test_rows),
            "test_path_hash": test_hash,
        }
    )

    for split, rows in sorted(by_split.items()):
        labels = label_set(rows)
        detail_rows.append(
            {
                "manifest": str(manifest),
                "protocol": protocol,
                "fold": fold_key,
                "split": split,
                "num_files": len(rows),
                "num_labels": len(labels),
                "label_coverage": json.dumps(sorted(labels)),
                "missing_labels": missing_labels(labels),
                "train_has_all_24": split != "train" or not train_missing,
                "train_missing_labels": missing_labels(train_labels) if split == "train" else "",
                "val_from_source_only": val_from_source_ok if split == "val" else "",
                "test_target_only": test_target_ok if split == "test" else "",
                "val_domains": val_domain_note if split == "val" else "",
                "test_domains": test_domain_note if split == "test" else "",
                "train_days": json.dumps(sorted(domain_values(train_rows, "day"))) if protocol.startswith("cross_day") and split == "train" else "",
                "path_overlap_train_val": path_overlap_tv if split == "train" else "",
                "path_overlap_train_test": path_overlap_tt if split == "train" else "",
                "path_overlap_val_test": path_overlap_vt if split == "val" else "",
                "audit_pass": fold_pass,
            }
        )

    return detail_rows, hash_rows, fold_pass


def discover_folds(manifest: Path) -> list[str | None]:
    rows = read_manifest(manifest)
    folds = sorted({str(r.get("fold", "")) for r in rows if r.get("fold", "") not in ("", None)})
    if not folds:
        return [None]
    return folds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="outputs/paper_ready_v2")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "manifest": root / "data/paper/cross_day_day1to5_source_only.csv",
            "protocol": "cross_day_source_only",
            "held_out_field": "day",
            "held_out_value": "5",
        },
        {
            "manifest": root / "data/manifest_configs_leave_one_config.csv",
            "protocol": "config_loco",
            "held_out_field": "config",
            "held_out_value": None,  # per fold
        },
        {
            "manifest": root / "data/manifest_locations_leave_one_location.csv",
            "protocol": "location_loco",
            "held_out_field": "location",
            "held_out_value": None,
        },
        {
            "manifest": root / "data/manifest_distances_leave_one_distance.csv",
            "protocol": "distance_loco",
            "held_out_field": "distance",
            "held_out_value": None,
        },
    ]

    all_detail: list[dict] = []
    all_hash: list[dict] = []
    global_pass = True

    for spec in specs:
        manifest: Path = spec["manifest"]
        if not manifest.exists():
            print(f"MISSING manifest: {manifest}", file=sys.stderr)
            global_pass = False
            continue
        for fold in discover_folds(manifest):
            held_out = spec["held_out_value"]
            if held_out is None and fold is not None:
                held_out = str(fold).replace("m", "") if spec["protocol"] == "distance_loco" else str(fold)
                # distance fold keys are like 5m but field values are 5
                if spec["protocol"] == "distance_loco":
                    held_out = str(fold).replace("m", "")
            detail, hashes, ok = audit_fold(
                manifest,
                fold,
                spec["protocol"],
                spec["held_out_field"],
                held_out,
                spec["held_out_field"],
            )
            all_detail.extend(detail)
            all_hash.extend(hashes)
            if not ok:
                global_pass = False
                print(f"FAIL {manifest.name} fold={fold}", file=sys.stderr)

    # Phase4 hash uniqueness check
    for protocol in ("config_loco", "location_loco", "distance_loco"):
        proto_hashes = [r for r in all_hash if r["protocol"] == protocol]
        hash_vals = [r["test_path_hash"] for r in proto_hashes]
        unique = len(set(hash_vals))
        if unique != len(hash_vals) and len(hash_vals) > 1:
            print(f"WARN {protocol}: test hashes not all unique ({unique}/{len(hash_vals)})", file=sys.stderr)
            global_pass = False

    detail_path = out_dir / "preflight_manifest_audit.csv"
    hash_path = out_dir / "phase4_fold_hash_audit.csv"
    manifest_audit_path = out_dir / "manifest_audit.csv"

    if all_detail:
        fields = list(all_detail[0].keys())
        with detail_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_detail)
        with manifest_audit_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_detail)

    if all_hash:
        fields = list(all_hash[0].keys())
        with hash_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_hash)

    print(f"wrote {detail_path} rows={len(all_detail)}")
    print(f"wrote {hash_path} rows={len(all_hash)}")
    print(f"PREFLIGHT_PASS={global_pass}")
    return 0 if global_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
