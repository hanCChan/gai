#!/usr/bin/env python3
"""LODO result diagnosis: day-wise comparison table and weak-day confusion analysis."""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "lodo_day1to5"
SUMMARY = OUT / "summary.csv"
DAYS = ["1", "2", "3", "4", "5"]
METRICS = ["window_acc", "file_acc", "macro_f1"]

# (column_prefix, method, eval_mode, file_vote_mode)
CONFIGS = [
    ("cnn_clf_ml", "cnn", "classifier", "mean_logits"),
    ("cnn_proto_mp", "cnn", "prototype", "mean_prob"),
    ("hyb_clf_ml", "hybrid", "classifier", "mean_logits"),
    ("hyb_clf_cw", "hybrid", "classifier", "confidence_weighted"),
    ("hyb_proto_mp", "hybrid", "prototype", "mean_prob"),
]


def parse_experiment(exp: str) -> tuple[str, str]:
    parts = exp.split("/")
    method = parts[0]
    day = parts[1].replace("test_day_", "")
    return method, day


def load_summary() -> dict[tuple[str, str, str, str], dict]:
    idx: dict[tuple[str, str, str, str], dict] = {}
    with SUMMARY.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            method, day = parse_experiment(row["experiment"])
            idx[(method, day, row["eval_mode"], row["file_vote_mode"])] = row
    return idx


def task1_daywise(idx) -> None:
    def get(method, day, mode, vote, metric):
        row = idx.get((method, day, mode, vote))
        return float(row[metric]) if row else float("nan")

    fieldnames = ["test_day"]
    for prefix, *_ in CONFIGS:
        for m in METRICS:
            fieldnames.append(f"{prefix}_{m}")
    for m in METRICS:
        fieldnames.append(f"diff_clf_{m}")  # hyb_clf_ml - cnn_clf_ml
    for m in METRICS:
        fieldnames.append(f"diff_proto_{m}")  # hyb_proto_mp - cnn_proto_mp

    out_rows = []
    for day in DAYS:
        row = {"test_day": day}
        for prefix, method, mode, vote in CONFIGS:
            for m in METRICS:
                row[f"{prefix}_{m}"] = round(get(method, day, mode, vote, m), 4)
        for m in METRICS:
            row[f"diff_clf_{m}"] = round(row[f"hyb_clf_ml_{m}"] - row[f"cnn_clf_ml_{m}"], 4)
            row[f"diff_proto_{m}"] = round(row[f"hyb_proto_mp_{m}"] - row[f"cnn_proto_mp_{m}"], 4)
        out_rows.append(row)

    # mean / std summary rows
    def agg(col, fn):
        vals = [r[col] for r in out_rows]
        n = len(vals)
        mu = sum(vals) / n
        if fn == "mean":
            return round(mu, 4)
        var = sum((v - mu) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        return round(var ** 0.5, 4)

    for tag, fn in (("mean", "mean"), ("std", "std")):
        srow = {"test_day": tag}
        for col in fieldnames[1:]:
            srow[col] = agg(col, fn)
        out_rows.append(srow)

    out_path = OUT / "daywise_comparison.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[task1] wrote {out_path} rows={len(out_rows)}")


def task2_confusion(weak_days: list[str]) -> None:
    for day in weak_days:
        src = OUT / "hybrid" / f"test_day_{day}" / "classifier_mean_logits"
        dst = OUT / "error_analysis" / f"test_day_{day}"
        dst.mkdir(parents=True, exist_ok=True)

        # device name map from per_device_accuracy
        label_to_name: dict[int, str] = {}
        with (src / "per_device_accuracy.csv").open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                label_to_name[int(r["label"])] = r["device_name"]

        # copy ready-made artifacts
        shutil.copy(src / "confusion_matrix.csv", dst / "confusion_matrix.csv")
        shutil.copy(src / "per_device_accuracy.csv", dst / "per_device_accuracy.csv")

        # wrong_files from file_predictions (correct == 0)
        wrong = []
        with (src / "file_predictions.csv").open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["correct"] in ("0", "0.0", "False"):
                    wrong.append(r)
        with (dst / "wrong_files.csv").open("w", encoding="utf-8", newline="") as f:
            cols = ["file_path", "label", "true_device", "pred", "pred_device", "num_windows", "confidence"]
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in wrong:
                lab, pred = int(r["label"]), int(r["pred"])
                writer.writerow({
                    "file_path": r["file_path"],
                    "label": lab,
                    "true_device": label_to_name.get(lab, str(lab)),
                    "pred": pred,
                    "pred_device": label_to_name.get(pred, str(pred)),
                    "num_windows": r["num_windows"],
                    "confidence": r["confidence"],
                })

        # top_confusion_pairs from window-level confusion_matrix off-diagonal
        pairs = []
        with (src / "confusion_matrix.csv").open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            pred_cols = [c for c in reader.fieldnames if c.startswith("pred_")]
            for r in reader:
                true_lab = int(r["label"])
                for c in pred_cols:
                    pred_lab = int(c.replace("pred_", ""))
                    count = int(r[c])
                    if pred_lab != true_lab and count > 0:
                        pairs.append((true_lab, pred_lab, count))
        pairs.sort(key=lambda x: x[2], reverse=True)
        with (dst / "top_confusion_pairs.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true_label", "true_device", "pred_label", "pred_device", "window_count"])
            for tl, pl, c in pairs[:30]:
                writer.writerow([tl, label_to_name.get(tl, str(tl)), pl, label_to_name.get(pl, str(pl)), c])

        print(f"[task2] test_day_{day}: wrong_files={len(wrong)} confusion_pairs={len(pairs)} -> {dst}")


def main() -> None:
    idx = load_summary()
    task1_daywise(idx)
    task2_confusion(["1", "2"])


if __name__ == "__main__":
    main()
