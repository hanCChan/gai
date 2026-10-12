from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize eval-seed sensitivity mean/std by evaluation mode.")
    parser.add_argument("--summary", default="outputs/eval_seed_sensitivity_hybrid/summary.csv")
    parser.add_argument("--out", default="outputs/eval_seed_sensitivity_hybrid/seed_sensitivity_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary)
    group_cols = ["eval_mode", "file_vote_mode"]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        eval_mode, file_vote_mode = keys
        rows.append(
            {
                "eval_mode": eval_mode,
                "file_vote_mode": file_vote_mode,
                "mean_window_acc": group["window_acc"].mean(),
                "std_window_acc": group["window_acc"].std(ddof=1),
                "mean_file_acc": group["file_acc"].mean(),
                "std_file_acc": group["file_acc"].std(ddof=1),
                "mean_macro_f1": group["macro_f1"].mean(),
                "std_macro_f1": group["macro_f1"].std(ddof=1),
                "num_runs": len(group),
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"seed_sensitivity_summary={out} rows={len(rows)}")


if __name__ == "__main__":
    main()
