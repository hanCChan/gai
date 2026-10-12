#!/usr/bin/env python3
"""Aggregate Phase4/7 results into outputs/paper_ready_v2/."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_FIELDS = [
    "phase",
    "experiment",
    "seed",
    "model_type",
    "oob_fusion_type",
    "window_acc",
    "window_macro_f1",
    "file_acc",
    "file_macro_f1",
    "n_windows",
    "n_files",
    "checkpoint_metric",
    "checkpoint_epoch",
    "checkpoint_val_acc",
    "checkpoint_val_macro_f1",
    "eval_checkpoint",
    "manifest",
    "metrics_path",
]


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_phase(out_dir: Path) -> str:
    parts = out_dir.parts
    for p in parts:
        if p.startswith("phase"):
            return p.split("_")[0] + "_" + p.split("_")[1] if len(p.split("_")) > 1 else p
    return "unknown"


def collect_metrics(root: Path, phase_dirs: list[Path]) -> list[dict]:
    rows = []
    for phase_base in phase_dirs:
        if not phase_base.exists():
            continue
        phase = phase_base.name
        for metrics_path in sorted(phase_base.rglob("metrics.json")):
            m = read_metrics(metrics_path)
            rel = metrics_path.parent.relative_to(phase_base / "outputs")
            exp = str(rel).replace("\\", "/")
            seed = ""
            if "_seed_" in exp:
                seed = exp.rsplit("_seed_", 1)[-1]
            rows.append(
                {
                    "phase": phase,
                    "experiment": exp,
                    "seed": seed,
                    "model_type": m.get("model_type", ""),
                    "oob_fusion_type": m.get("oob_fusion_type", ""),
                    "window_acc": m.get("window_acc", ""),
                    "window_macro_f1": m.get("window_macro_f1", m.get("macro_f1", "")),
                    "file_acc": m.get("file_acc", ""),
                    "file_macro_f1": m.get("file_macro_f1", ""),
                    "n_windows": m.get("num_windows", ""),
                    "n_files": m.get("num_files", ""),
                    "checkpoint_metric": m.get("checkpoint_metric", ""),
                    "checkpoint_epoch": m.get("checkpoint_epoch", ""),
                    "checkpoint_val_acc": m.get("checkpoint_val_acc", ""),
                    "checkpoint_val_macro_f1": m.get("checkpoint_val_macro_f1", ""),
                    "eval_checkpoint": m.get("eval_checkpoint", ""),
                    "manifest": m.get("manifest", ""),
                    "metrics_path": str(metrics_path),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def aggregate_multiseed(rows: list[dict], prefixes: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for prefix in prefixes:
            if row["experiment"].startswith(prefix):
                base = prefix
                grouped[base].append(row)
                break
    out = []
    for base, items in sorted(grouped.items()):
        for metric in ("window_acc", "window_macro_f1", "file_acc", "file_macro_f1"):
            vals = [float(r[metric]) for r in items if r.get(metric) not in ("", None)]
            if vals:
                out.append(
                    {
                        "model": base,
                        "metric": metric,
                        "n_seeds": len(vals),
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "values": json.dumps(vals),
                    }
                )
    return out


def deployment_summary(rows: list[dict]) -> list[dict]:
    dep = [r for r in rows if "phase4" in r["phase"]]
    return dep


def phase4_fold_check(phase4_base: Path, out_path: Path) -> bool:
    """Verify per-fold metrics differ; write hash audit."""
    hash_rows = []
    metric_by_fold: dict[str, list[float]] = defaultdict(list)
    for metrics_path in sorted((phase4_base / "outputs").rglob("metrics.json")):
        m = read_metrics(metrics_path)
        exp = metrics_path.parent.name
        fp = phase4_base / "outputs" / exp / "file_predictions.csv"
        paths = []
        if fp.exists():
            with fp.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                paths = sorted({row["file_path"] for row in reader})
        h = hashlib.sha256("\n".join(paths).encode()).hexdigest()[:16]
        hash_rows.append({"experiment": exp, "file_acc": m.get("file_acc", ""), "test_path_hash": h, "n_files": len(paths)})
        task = exp.rsplit("_fold_", 1)[0] if "_fold_" in exp else exp
        metric_by_fold[task].append(float(m.get("file_acc", 0)))

    write_csv(out_path, hash_rows, ["experiment", "file_acc", "test_path_hash", "n_files"])

    ok = True
    for task, accs in metric_by_fold.items():
        if len(accs) > 1 and len(set(round(a, 6) for a in accs)) == 1:
            print(f"WARN: {task} all folds have identical file_acc={accs[0]:.4f}")
            ok = False
    return ok


def plot_confusion(cm_path: Path, out_path: Path) -> None:
    if not cm_path.exists():
        return
    matrix = []
    with cm_path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            matrix.append([int(x) for x in row[1:]])
    if not matrix:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(out_path.stem)
    fig.colorbar(im, ax=ax, fraction=0.046)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pred_hist(pred_path: Path, out_path: Path) -> None:
    if not pred_path.exists():
        return
    preds = []
    with pred_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds.append(int(row["pred"]))
    if not preds:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(preds, bins=range(0, 25), align="left", rwidth=0.8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Count")
    ax.set_title(out_path.stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--phase4-base", default="")
    parser.add_argument("--phase7-base", default="")
    parser.add_argument("--out-dir", default="outputs/paper_ready_v2")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = root / "outputs" / "paper_runs"
    phase4 = Path(args.phase4_base) if args.phase4_base else max(runs.glob("phase4_deployment_*"), default=None, key=lambda p: p.stat().st_mtime)
    phase7 = Path(args.phase7_base) if args.phase7_base else max(runs.glob("phase7_domain_robust_*"), default=None, key=lambda p: p.stat().st_mtime)

    phase_dirs = [p for p in [phase4, phase7] if p is not None and p.exists()]
    rows = collect_metrics(root, phase_dirs)
    write_csv(out_dir / "paper_metrics_clean.csv", rows, METRIC_FIELDS)

    if phase4 and phase4.exists():
        dep = deployment_summary(rows)
        write_csv(out_dir / "deployment_shift_fixed_summary.csv", dep, METRIC_FIELDS)
        ok = phase4_fold_check(phase4, out_dir / "phase4_results_hash_audit.csv")
        if not ok:
            print("WARN: Phase4 fold metrics may still be buggy")

    if phase7 and phase7.exists():
        p7 = [r for r in rows if "phase7" in r["phase"]]
        write_csv(out_dir / "phase7_domain_robust_ablation.csv", p7, METRIC_FIELDS)
        ms = aggregate_multiseed(
            p7,
            ["M1_cnn_iq", "M4_concat_oob", "M5a_gated_plain", "M6_gated_oob_dropout", "M7_gated_full_robust"],
        )
        write_csv(out_dir / "phase7_key_models_multiseed.csv", ms, ["model", "metric", "n_seeds", "mean", "std", "values"])

    fig_cm = out_dir / "figures" / "confusion_matrices"
    fig_hist = out_dir / "figures" / "prediction_histograms"
    for phase_base in phase_dirs:
        for metrics_path in (phase_base / "outputs").rglob("metrics.json"):
            exp = metrics_path.parent
            plot_confusion(exp / "confusion_matrix.csv", fig_cm / f"{phase_base.name}_{exp.name}.png")
            plot_pred_hist(exp / "predictions.csv", fig_hist / f"{phase_base.name}_{exp.name}.png")

    readme = out_dir / "README_paper_results_v2.md"
    readme.write_text(
        f"""# Paper Results v2

Generated from:
- Phase4: `{phase4}`
- Phase7: `{phase7}`

## Files
- `preflight_manifest_audit.csv` — manifest split audit (run preflight first)
- `phase4_fold_hash_audit.csv` — test path hashes per LOCO fold
- `paper_metrics_clean.csv` — unified metrics (window/file acc & macro-F1)
- `deployment_shift_fixed_summary.csv` — Phase4 deployment results
- `phase7_domain_robust_ablation.csv` — ablation matrix
- `phase7_key_models_multiseed.csv` — 3-seed aggregates for key models

## Manifest fix (2026-06-24)
Deployment LOCO val split revised: devices 19–24 val only from ONE source domain;
train retains all 24 classes across remaining source domains.
""",
        encoding="utf-8",
    )
    print(f"paper_ready_v2 updated: {out_dir}")


if __name__ == "__main__":
    main()
