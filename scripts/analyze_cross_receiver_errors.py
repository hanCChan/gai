#!/usr/bin/env python3
"""Task 3: cross-receiver error analysis (Hybrid main line, classifier mean_logits).

Reads predictions.csv (window level) and file_predictions.csv (file level) from
outputs/cross_receiver/{dir}/hybrid/classifier_mean_logits/.

Outputs into outputs/cross_receiver_analysis/:
  {dir}_confusion_matrix.csv   (window-level 24x24, rows=true label, cols=pred)
  {dir}_wrong_files.csv        (file-level misclassifications)
  top_confusion_pairs.csv      (aggregated across both directions)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data1/hcc/llm4RF")
OUTDIR = ROOT / "outputs/cross_receiver_analysis"
NUM_CLASSES = 24
MODEL = "hybrid"
EVALCFG = "classifier_mean_logits"
DIRECTIONS = ["rx1_to_rx2", "rx2_to_rx1"]


def pred_dir(direction: str) -> Path:
    return ROOT / "outputs/cross_receiver" / direction / MODEL / EVALCFG


def confusion_matrix(df: pd.DataFrame) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for lab, pred in zip(df["label"].astype(int), df["pred"].astype(int)):
        if 0 <= lab < NUM_CLASSES and 0 <= pred < NUM_CLASSES:
            cm[lab, pred] += 1
    return cm


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_pairs: dict[tuple[int, int], int] = {}

    for direction in DIRECTIONS:
        d = pred_dir(direction)
        win = pd.read_csv(d / "predictions.csv")
        files = pd.read_csv(d / "file_predictions.csv")

        # window-level confusion matrix
        cm = confusion_matrix(win)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{i}" for i in range(NUM_CLASSES)],
            columns=[f"pred_{i}" for i in range(NUM_CLASSES)],
        )
        cm_df.to_csv(OUTDIR / f"{direction}_confusion_matrix.csv")

        # file-level wrong files
        wrong = files[files["correct"].astype(int) == 0].copy()
        wrong["true_device"] = wrong["label"].astype(int) + 1
        wrong["pred_device"] = wrong["pred"].astype(int) + 1
        wrong["direction"] = direction
        cols = ["direction", "file_path", "receiver", "label", "pred",
                "true_device", "pred_device", "confidence", "num_windows"]
        cols = [c for c in cols if c in wrong.columns]
        wrong[cols].to_csv(OUTDIR / f"{direction}_wrong_files.csv", index=False)

        # accumulate off-diagonal window confusions
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if i != j and cm[i, j] > 0:
                    all_pairs[(i, j)] = all_pairs.get((i, j), 0) + int(cm[i, j])

        n_wrong_files = int((files["correct"].astype(int) == 0).sum())
        print(f"{direction}: window_rows={len(win)} files={len(files)} "
              f"wrong_files={n_wrong_files}/{len(files)} file_acc={1 - n_wrong_files/len(files):.4f}")

    # top confusion pairs aggregated across both directions
    pair_rows = [
        {"true_label": i, "pred_label": j, "true_device": i + 1, "pred_device": j + 1, "window_count": c}
        for (i, j), c in all_pairs.items()
    ]
    pairs_df = pd.DataFrame(pair_rows).sort_values("window_count", ascending=False).reset_index(drop=True)
    pairs_df.to_csv(OUTDIR / "top_confusion_pairs.csv", index=False)
    print(f"wrote top_confusion_pairs.csv pairs={len(pairs_df)}")
    print("top 10 confusion pairs (true->pred, window_count):")
    for _, r in pairs_df.head(10).iterrows():
        print(f"  {int(r.true_label):2d} -> {int(r.pred_label):2d}   {int(r.window_count)}")


if __name__ == "__main__":
    main()
