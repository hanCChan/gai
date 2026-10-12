from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_A = "outputs/osu_cnn_day1_day2_spf256/osu_cnn_iq/classifier/predictions.csv"
DEFAULT_B_PRIMARY = "outputs/hybrid_center_loss_spf256/center_w001/classifier/predictions.csv"
DEFAULT_B_FALLBACK = "outputs/hybrid_regularization_spf256/label_smoothing_005_weight_decay_5e4/classifier/predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired file-level comparison between two prediction CSV files.")
    parser.add_argument("--a-pred", default=DEFAULT_A)
    parser.add_argument("--b-pred", default=None)
    parser.add_argument("--a-name", default="OSU-CNN-IQ")
    parser.add_argument("--b-name", default="Hybrid")
    parser.add_argument("--out", default="outputs/stat_analysis/paired_cnn_vs_hybrid.csv")
    parser.add_argument("--diff-out", default="outputs/stat_analysis/paired_cnn_vs_hybrid_diff_files.csv")
    parser.add_argument("--group", default="file_path")
    return parser.parse_args()


def resolve_b_path(value: str | None) -> str:
    if value:
        return value
    primary = Path(DEFAULT_B_PRIMARY)
    if primary.exists():
        return str(primary)
    return DEFAULT_B_FALLBACK


def aggregate_file_predictions(path: str, group_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r} in {path}")
    if "num_windows" in df.columns:
        return df[[group_col, "label", "pred"]].drop_duplicates(group_col).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for file_path, group in df.groupby(group_col, sort=False):
        label = int(group["label"].iloc[0])
        counts = Counter(int(value) for value in group["pred"])
        max_count = max(counts.values())
        candidates = [pred for pred, count in counts.items() if count == max_count]
        if len(candidates) == 1 or "confidence" not in group.columns:
            pred = min(candidates)
        else:
            confidence_by_pred = group.groupby("pred")["confidence"].mean().to_dict()
            pred = max(candidates, key=lambda item: (float(confidence_by_pred.get(item, 0.0)), -int(item)))
        rows.append({group_col: file_path, "label": label, "pred": int(pred)})
    return pd.DataFrame(rows)


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, 2.0 * cdf)


def chi_square_pvalue_cc(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    stat = (abs(b - c) - 1.0) ** 2 / n
    return math.erfc(math.sqrt(stat / 2.0))


def main() -> None:
    args = parse_args()
    b_pred = resolve_b_path(args.b_pred)
    a = aggregate_file_predictions(args.a_pred, args.group).rename(columns={"pred": "pred_a", "label": "label_a"})
    b = aggregate_file_predictions(b_pred, args.group).rename(columns={"pred": "pred_b", "label": "label_b"})
    joined = a.merge(b, on=args.group, how="inner")
    if joined.empty:
        raise ValueError("No overlapping files between prediction CSVs.")
    joined["label"] = joined["label_a"]
    joined["a_correct"] = joined["pred_a"] == joined["label"]
    joined["b_correct"] = joined["pred_b"] == joined["label"]

    both_correct = int((joined["a_correct"] & joined["b_correct"]).sum())
    both_wrong = int((~joined["a_correct"] & ~joined["b_correct"]).sum())
    a_correct_b_wrong = int((joined["a_correct"] & ~joined["b_correct"]).sum())
    a_wrong_b_correct = int((~joined["a_correct"] & joined["b_correct"]).sum())
    total = int(len(joined))
    acc_a = float(joined["a_correct"].mean())
    acc_b = float(joined["b_correct"].mean())
    exact_p = exact_mcnemar_pvalue(a_correct_b_wrong, a_wrong_b_correct)
    chi_p = chi_square_pvalue_cc(a_correct_b_wrong, a_wrong_b_correct)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "a_name",
                "b_name",
                "a_pred",
                "b_pred",
                "both_correct",
                "both_wrong",
                "a_correct_b_wrong",
                "a_wrong_b_correct",
                "total_files",
                "acc_a",
                "acc_b",
                "mcnemar_p_value",
                "mcnemar_exact_p",
                "mcnemar_chi_square_cc_p",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "a_name": args.a_name,
                "b_name": args.b_name,
                "a_pred": args.a_pred,
                "b_pred": b_pred,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "a_correct_b_wrong": a_correct_b_wrong,
                "a_wrong_b_correct": a_wrong_b_correct,
                "total_files": total,
                "acc_a": acc_a,
                "acc_b": acc_b,
                "mcnemar_p_value": exact_p,
                "mcnemar_exact_p": exact_p,
                "mcnemar_chi_square_cc_p": chi_p,
            }
        )

    diff = joined[joined["a_correct"] != joined["b_correct"]].copy()
    diff["case"] = diff.apply(lambda row: "a_correct_b_wrong" if row["a_correct"] else "a_wrong_b_correct", axis=1)
    diff_out = Path(args.diff_out)
    diff_out.parent.mkdir(parents=True, exist_ok=True)
    diff[[args.group, "label", "pred_a", "pred_b", "a_correct", "b_correct", "case"]].to_csv(diff_out, index=False)
    print(f"paired_summary={out}")
    print(f"paired_diff_files={diff_out}")


if __name__ == "__main__":
    main()
