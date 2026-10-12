#!/usr/bin/env python3
"""Step1.5 statistical analysis for paper_ready_v3 Step1 (no training)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CORE_PAIR = ("F_cross_attn_chirp_plain", "A_cnn_iq")
COLLAPSE_MODELS = ("D_concat_oob_plain", "H_gated_chirp_plain")
F_MODEL = "F_cross_attn_chirp_plain"
SEEDS = list(range(5))


def load_file_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def predictions_by_file(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        key = row.get("file_path") or row.get("path") or row["label"]
        out[key] = row
    return out


def file_acc(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    correct = sum(int(r["correct"]) for r in rows)
    return correct / len(rows)


def mcnemar(b: int, c: int) -> dict:
    """b=F-only correct, c=A-only correct."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0, "statistic": 0.0}
    stat = (abs(b - c) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2.0))
    return {"b": b, "c": c, "n_discordant": n, "p_value": p, "statistic": stat}


def paired_bootstrap_diff(
    f_rows: list[dict],
    a_rows: list[dict],
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    f_map = predictions_by_file(f_rows)
    a_map = predictions_by_file(a_rows)
    keys = sorted(set(f_map) & set(a_map))
    if not keys:
        return {"n_files": 0, "mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    f_correct = np.array([int(f_map[k]["correct"]) for k in keys], dtype=float)
    a_correct = np.array([int(a_map[k]["correct"]) for k in keys], dtype=float)
    diffs = np.empty(n_boot, dtype=float)
    n = len(keys)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = f_correct[idx].mean() - a_correct[idx].mean()
    return {
        "n_files": n,
        "mean_diff": float(f_correct.mean() - a_correct.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_bootstrap": n_boot,
    }


def prediction_histogram(rows: list[dict]) -> dict:
    preds = [int(r["pred"]) for r in rows]
    counts = Counter(preds)
    n = len(preds)
    top = counts.most_common(3)
    return {
        "n_files": n,
        "n_unique_preds": len(counts),
        "top_predictions": top,
        "mode_fraction": top[0][1] / n if n else 0.0,
        "counts": dict(sorted(counts.items())),
    }


def per_device_accuracy(rows: list[dict]) -> list[dict]:
    by_dev: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        dev = row.get("device") or row.get("label")
        by_dev[str(dev)].append(int(row["correct"]))
    out = []
    for dev in sorted(by_dev, key=lambda x: int(x) if x.isdigit() else x):
        vals = by_dev[dev]
        out.append({"device": dev, "n_files": len(vals), "accuracy": sum(vals) / len(vals)})
    return out


def prediction_overlap(f_rows: list[dict], a_rows: list[dict]) -> dict:
    f_map = predictions_by_file(f_rows)
    a_map = predictions_by_file(a_rows)
    keys = sorted(set(f_map) & set(a_map))
    same_pred = both_correct = f_only = a_only = both_wrong = 0
    for k in keys:
        fp = int(f_map[k]["pred"])
        ap = int(a_map[k]["pred"])
        fc = int(f_map[k]["correct"])
        ac = int(a_map[k]["correct"])
        if fp == ap:
            same_pred += 1
        if fc and ac:
            both_correct += 1
        elif fc and not ac:
            f_only += 1
        elif ac and not fc:
            a_only += 1
        else:
            both_wrong += 1
    n = len(keys)
    return {
        "n_files": n,
        "same_prediction": same_pred,
        "same_prediction_rate": same_pred / n if n else 0.0,
        "both_correct": both_correct,
        "f_only_correct": f_only,
        "a_only_correct": a_only,
        "both_wrong": both_wrong,
    }


def compare_best_last(
    best_rows: list[dict],
    last_rows: list[dict],
) -> dict:
    best_map = predictions_by_file(best_rows)
    last_map = predictions_by_file(last_rows)
    keys = sorted(set(best_map) & set(last_map))
    best_acc = file_acc(best_rows)
    last_acc = file_acc(last_rows)
    agree = sum(int(best_map[k]["pred"]) == int(last_map[k]["pred"]) for k in keys)
    return {
        "n_files": len(keys),
        "best_file_acc": best_acc,
        "last_file_acc": last_acc,
        "acc_delta_last_minus_best": last_acc - best_acc,
        "same_prediction_rate": agree / len(keys) if keys else 0.0,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def render_report(summary: dict) -> str:
    lines = [
        "# Step1.5 Statistical Report",
        "",
        "## F vs A (cross-day Day5 test, 5 seeds)",
        "",
    ]
    fa = summary["f_vs_a"]
    lines.append(
        f"- F File-Acc: **{fa['f_mean_acc']*100:.1f} ± {fa['f_std_acc']*100:.1f}%** "
        f"(n={fa['n_seeds']} seeds)"
    )
    lines.append(
        f"- A File-Acc: **{fa['a_mean_acc']*100:.1f} ± {fa['a_std_acc']*100:.1f}%**"
    )
    lines.append(f"- Mean paired gain (F−A): **{fa['mean_paired_gain_pp']:+.1f} pp**")
    lines.append(f"- Seed wins: F better **{fa['f_wins']}**, tie **{fa['ties']}**, A better **{fa['a_wins']}**")
    lines.append("")
    lines.append("### Paired bootstrap CI (File-Acc difference F−A)")
    lines.append("")
    for row in fa["bootstrap_per_seed"]:
        lines.append(
            f"- seed {row['seed']}: mean diff {row['mean_diff']*100:+.1f} pp, "
            f"95% CI [{row['ci_low']*100:+.1f}, {row['ci_high']*100:+.1f}] pp (n={row['n_files']} files)"
        )
    pooled = fa["bootstrap_pooled"]
    lines.append(
        f"- pooled (descriptive): mean diff {pooled['mean_diff']*100:+.1f} pp, "
        f"95% CI [{pooled['ci_low']*100:+.1f}, {pooled['ci_high']*100:+.1f}] pp "
        f"(n={pooled['n_files']} file-seed pairs)"
    )
    lines.append("")
    lines.append("### McNemar (per seed, appendix)")
    lines.append("")
    for row in fa["mcnemar_per_seed"]:
        lines.append(
            f"- seed {row['seed']}: F-only={row['b']}, A-only={row['c']}, "
            f"p={row['p_value']:.3f}"
        )
    lines.append("")
    lines.append("## F best.pt vs last.pt (eval-only robustness check)")
    lines.append("")
    bl = summary.get("best_vs_last", {})
    if not bl.get("per_seed"):
        lines.append("_Run `bash scripts/paper/run_step1_5_f_last_eval.sh` first._")
    else:
        for row in bl["per_seed"]:
            lines.append(
                f"- seed {row['seed']}: best={row['best_file_acc']*100:.1f}%, "
                f"last={row['last_file_acc']*100:.1f}%, "
                f"Δ={row['acc_delta_last_minus_best']*100:+.1f} pp"
            )
        lines.append(
            f"- mean Δ(last−best): **{bl['mean_delta_pp']:+.1f} pp** "
            f"(std {bl['std_delta_pp']:.1f} pp)"
        )
    lines.append("")
    lines.append("## D/H collapse diagnostic")
    lines.append("")
    for model_id, seeds in summary.get("collapse_hist", {}).items():
        lines.append(f"### {model_id}")
        for seed, hist in seeds.items():
            lines.append(
                f"- seed {seed}: unique_preds={hist['n_unique_preds']}, "
                f"mode_frac={hist['mode_fraction']:.2f}, top={hist['top_predictions']}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step1.5 statistics for paper_ready_v3 Step1")
    parser.add_argument(
        "--step1-dir",
        default="outputs/paper_ready_v3/step1_phase7_clean",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: {step1-dir}/statistics",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    step1_dir = Path(args.step1_dir)
    if not step1_dir.is_absolute():
        step1_dir = ROOT / step1_dir
    out_dir = Path(args.out_dir) if args.out_dir else step1_dir / "statistics"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = step1_dir / "outputs"
    f_model, a_model = CORE_PAIR

    seed_summary_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    mcnemar_rows: list[dict] = []
    overlap_rows: list[dict] = []
    per_device_rows: list[dict] = []
    histogram_rows: list[dict] = []

    f_accs: list[float] = []
    a_accs: list[float] = []
    paired_gains: list[float] = []
    f_wins = ties = a_wins = 0

    pooled_f: list[dict] = []
    pooled_a: list[dict] = []

    for seed in SEEDS:
        f_path = outputs / f_model / f"seed_{seed}" / "file_predictions.csv"
        a_path = outputs / a_model / f"seed_{seed}" / "file_predictions.csv"
        if not f_path.exists() or not a_path.exists():
            print(f"WARN: missing predictions for seed {seed}")
            continue

        f_rows = load_file_predictions(f_path)
        a_rows = load_file_predictions(a_path)
        f_acc = file_acc(f_rows)
        a_acc = file_acc(a_rows)
        f_accs.append(f_acc)
        a_accs.append(a_acc)
        gain = f_acc - a_acc
        paired_gains.append(gain)
        if gain > 1e-9:
            f_wins += 1
        elif gain < -1e-9:
            a_wins += 1
        else:
            ties += 1

        seed_summary_rows.append(
            {
                "seed": seed,
                "f_file_acc": f_acc,
                "a_file_acc": a_acc,
                "paired_gain": gain,
                "winner": "F" if gain > 1e-9 else ("A" if gain < -1e-9 else "tie"),
            }
        )

        boot = paired_bootstrap_diff(f_rows, a_rows, n_boot=args.n_bootstrap, seed=seed)
        boot["seed"] = seed
        bootstrap_rows.append(boot)

        ov = prediction_overlap(f_rows, a_rows)
        ov["seed"] = seed
        overlap_rows.append(ov)

        f_map = predictions_by_file(f_rows)
        a_map = predictions_by_file(a_rows)
        keys = sorted(set(f_map) & set(a_map))
        b = sum(int(f_map[k]["correct"]) and not int(a_map[k]["correct"]) for k in keys)
        c = sum(int(a_map[k]["correct"]) and not int(f_map[k]["correct"]) for k in keys)
        mc = mcnemar(b, c)
        mc["seed"] = seed
        mcnemar_rows.append(mc)

        for model_id, rows in ((f_model, f_rows), (a_model, a_rows)):
            for dev_row in per_device_accuracy(rows):
                per_device_rows.append({"model_id": model_id, "seed": seed, **dev_row})
            hist = prediction_histogram(rows)
            histogram_rows.append({"model_id": model_id, "seed": seed, **hist})

        for row in f_rows:
            pooled_f.append({**row, "_seed": seed})
        for row in a_rows:
            pooled_a.append({**row, "_seed": seed})

    # Pooled descriptive bootstrap: treat each (file, seed) as independent pair
    if pooled_f and pooled_a:
        pf = {f"{r.get('file_path')}|{r['_seed']}": r for r in pooled_f}
        pa = {f"{r.get('file_path')}|{r['_seed']}": r for r in pooled_a}
        keys = sorted(set(pf) & set(pa))
        rng = np.random.default_rng(42)
        f_c = np.array([int(pf[k]["correct"]) for k in keys], dtype=float)
        a_c = np.array([int(pa[k]["correct"]) for k in keys], dtype=float)
        diffs = []
        n = len(keys)
        for _ in range(args.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            diffs.append(f_c[idx].mean() - a_c[idx].mean())
        bootstrap_pooled = {
            "n_files": n,
            "mean_diff": float(f_c.mean() - a_c.mean()),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "n_bootstrap": args.n_bootstrap,
        }
    else:
        bootstrap_pooled = {"n_files": 0}

    collapse_hist: dict[str, dict] = {}
    for model_id in COLLAPSE_MODELS:
        collapse_hist[model_id] = {}
        for seed in SEEDS:
            p = outputs / model_id / f"seed_{seed}" / "file_predictions.csv"
            if not p.exists():
                continue
            hist = prediction_histogram(load_file_predictions(p))
            collapse_hist[model_id][str(seed)] = hist

    # best vs last
    last_root = out_dir / "F_last_eval"
    best_last_rows: list[dict] = []
    for seed in SEEDS:
        best_path = outputs / F_MODEL / f"seed_{seed}" / "file_predictions.csv"
        last_path = last_root / f"seed_{seed}" / "file_predictions.csv"
        if not last_path.exists():
            continue
        cmp = compare_best_last(load_file_predictions(best_path), load_file_predictions(last_path))
        cmp["seed"] = seed
        best_last_rows.append(cmp)

    best_vs_last_summary: dict = {"per_seed": best_last_rows}
    if best_last_rows:
        deltas = [r["acc_delta_last_minus_best"] for r in best_last_rows]
        best_vs_last_summary["mean_delta_pp"] = float(np.mean(deltas) * 100)
        best_vs_last_summary["std_delta_pp"] = float(np.std(deltas) * 100)

    summary = {
        "f_vs_a": {
            "f_mean_acc": float(np.mean(f_accs)) if f_accs else float("nan"),
            "f_std_acc": float(np.std(f_accs)) if f_accs else float("nan"),
            "a_mean_acc": float(np.mean(a_accs)) if a_accs else float("nan"),
            "a_std_acc": float(np.std(a_accs)) if a_accs else float("nan"),
            "mean_paired_gain_pp": float(np.mean(paired_gains) * 100) if paired_gains else float("nan"),
            "f_wins": f_wins,
            "ties": ties,
            "a_wins": a_wins,
            "n_seeds": len(f_accs),
            "bootstrap_per_seed": bootstrap_rows,
            "bootstrap_pooled": bootstrap_pooled,
            "mcnemar_per_seed": mcnemar_rows,
        },
        "best_vs_last": best_vs_last_summary,
        "collapse_hist": collapse_hist,
    }

    write_csv(out_dir / "seed_paired_summary.csv", seed_summary_rows, ["seed", "f_file_acc", "a_file_acc", "paired_gain", "winner"])
    write_csv(
        out_dir / "bootstrap_ci.csv",
        bootstrap_rows,
        ["seed", "n_files", "mean_diff", "ci_low", "ci_high", "n_bootstrap"],
    )
    write_csv(
        out_dir / "mcnemar_per_seed.csv",
        mcnemar_rows,
        ["seed", "b", "c", "n_discordant", "p_value", "statistic"],
    )
    write_csv(
        out_dir / "prediction_overlap.csv",
        overlap_rows,
        ["seed", "n_files", "same_prediction", "same_prediction_rate", "both_correct", "f_only_correct", "a_only_correct", "both_wrong"],
    )
    write_csv(out_dir / "per_device_accuracy.csv", per_device_rows, ["model_id", "seed", "device", "n_files", "accuracy"])
    write_csv(
        out_dir / "prediction_histogram.csv",
        [
            {
                "model_id": r["model_id"],
                "seed": r["seed"],
                "n_files": r["n_files"],
                "n_unique_preds": r["n_unique_preds"],
                "mode_fraction": r["mode_fraction"],
                "top_predictions": json.dumps(r["top_predictions"]),
            }
            for r in histogram_rows
        ],
        ["model_id", "seed", "n_files", "n_unique_preds", "mode_fraction", "top_predictions"],
    )
    write_csv(
        out_dir / "best_vs_last.csv",
        best_last_rows,
        ["seed", "n_files", "best_file_acc", "last_file_acc", "acc_delta_last_minus_best", "same_prediction_rate"],
    )

    (out_dir / "step1_5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "STEP1_STAT_REPORT.md").write_text(render_report(summary), encoding="utf-8")

    print(f"Wrote statistics -> {out_dir}")
    print(f"  F vs A: {f_wins} wins, {ties} ties, {a_wins} losses over {len(f_accs)} seeds")
    if f_accs:
        print(f"  Mean paired gain: {np.mean(paired_gains)*100:+.1f} pp")
    if not best_last_rows:
        print("  NOTE: run scripts/paper/run_step1_5_f_last_eval.sh for best vs last check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
