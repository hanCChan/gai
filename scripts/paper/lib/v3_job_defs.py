"""Job definitions for paper_ready_v3 clean experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    job_id: str
    train_args: list[str]
    oob_fusion_type: str = ""
    oob_norm: str = ""
    use_chirp: bool = False
    model_type: str = "rf_hstu"


CROSS_DAY_MANIFEST = "data/paper/cross_day_day1to5_source_only.csv"
LOCATION_MANIFEST = "data/manifest_locations_leave_one_location.csv"
LOCATION_FOLD = "3"

# Cross-day plain protocol (Step1)
CROSS_DAY_COMMON = [
    "--manifest", CROSS_DAY_MANIFEST,
    "--batch-size", "128",
    "--samples-per-file", "256",
    "--eval-samples-per-file", "256",
    "--dim", "64",
    "--depth", "2",
    "--device", "cuda",
    "--train-split", "train",
    "--val-split", "val",
    "--eval-split", "test",
    "--input-norm", "iq_rms",
    "--fft-norm", "log_zscore",
]

CROSS_DAY_TRAIN_ONLY = [
    "--epochs", "80",
    "--lr", "3e-3",
    "--loss-type", "ce",
    "--checkpoint-metric", "acc",
    "--weight-decay", "5e-4",
    "--label-smoothing", "0",
]

# Backward-compatible aliases
CROSS_DAY_TRAIN_BASE = CROSS_DAY_COMMON + CROSS_DAY_TRAIN_ONLY
CROSS_DAY_EVAL_BASE = CROSS_DAY_COMMON

# Location outdoor (Step3) — align with Phase4 deployment
LOCATION_TRAIN_BASE = [
    "--manifest", LOCATION_MANIFEST,
    "--fold", LOCATION_FOLD,
    "--epochs", "80",
    "--batch-size", "256",
    "--lr", "3e-3",
    "--samples-per-file", "256",
    "--eval-samples-per-file", "256",
    "--dim", "64",
    "--depth", "2",
    "--device", "cuda",
    "--train-split", "train",
    "--val-split", "val",
    "--eval-split", "test",
    "--input-norm", "iq_rms",
    "--fft-norm", "log_zscore",
    "--loss-type", "ce",
    "--checkpoint-metric", "acc",
    "--weight-decay", "5e-4",
    "--label-smoothing", "0.05",
]

LOCATION_EVAL_BASE = [
    "--manifest", LOCATION_MANIFEST,
    "--fold", LOCATION_FOLD,
    "--batch-size", "256",
    "--samples-per-file", "256",
    "--eval-samples-per-file", "256",
    "--dim", "64",
    "--depth", "2",
    "--device", "cuda",
    "--train-split", "train",
    "--val-split", "val",
    "--eval-split", "test",
    "--input-norm", "iq_rms",
    "--fft-norm", "log_zscore",
]

STEM = ["--model-type", "rf_hstu", "--patch-embed-type", "cnn_stem", "--cnn-stem-dim", "32"]

STEP1_CORE: list[ModelSpec] = [
    ModelSpec(
        "A_cnn_iq",
        ["--model-type", "osu_cnn", "--cnn-input-type", "iq"],
        oob_fusion_type="n/a",
        oob_norm="n/a",
        model_type="osu_cnn",
    ),
    ModelSpec(
        "D_concat_oob_plain",
        [*STEM, "--oob-fusion-type", "concat_oob", "--oob-norm", "zscore"],
        oob_fusion_type="concat_oob",
        oob_norm="zscore",
    ),
    ModelSpec(
        "F_cross_attn_chirp_plain",
        [
            *STEM,
            "--oob-fusion-type", "cross_attn_oob",
            "--use-oob-cross-attention",
            "--use-chirp-embedding",
            "--oob-norm", "zscore",
        ],
        oob_fusion_type="cross_attn_oob",
        oob_norm="zscore",
        use_chirp=True,
    ),
    ModelSpec(
        "H_gated_chirp_plain",
        [
            *STEM,
            "--oob-fusion-type", "gated_oob",
            "--use-chirp-embedding",
            "--oob-norm", "zscore",
        ],
        oob_fusion_type="gated_oob",
        oob_norm="zscore",
        use_chirp=True,
    ),
]

STEP1_DIAG: list[ModelSpec] = [
    ModelSpec(
        "B_linear_no_oob",
        [
            "--model-type", "rf_hstu",
            "--patch-embed-type", "linear",
            "--no-oob",
            "--oob-fusion-type", "no_oob",
            "--oob-norm", "none",
        ],
        oob_fusion_type="no_oob",
        oob_norm="none",
    ),
    ModelSpec(
        "C_cnn_stem_chirp_no_oob",
        [
            *STEM,
            "--no-oob",
            "--oob-fusion-type", "no_oob",
            "--use-chirp-embedding",
            "--oob-norm", "none",
        ],
        oob_fusion_type="no_oob",
        oob_norm="none",
        use_chirp=True,
    ),
]

STEP3_LOCATION: list[ModelSpec] = [
    ModelSpec(
        "L0_cnn",
        ["--model-type", "osu_cnn", "--cnn-input-type", "iq", "--oob-norm", "none"],
        oob_fusion_type="n/a",
        oob_norm="none",
        model_type="osu_cnn",
    ),
    ModelSpec(
        "L1_cross_attn_chirp_phase4",
        [
            *STEM,
            "--oob-fusion-type", "cross_attn_oob",
            "--use-oob-cross-attention",
            "--use-chirp-embedding",
            "--oob-norm", "ratio",
        ],
        oob_fusion_type="cross_attn_oob",
        oob_norm="ratio",
        use_chirp=True,
    ),
    ModelSpec(
        "L2_concat_oob",
        [*STEM, "--oob-fusion-type", "concat_oob", "--oob-norm", "ratio"],
        oob_fusion_type="concat_oob",
        oob_norm="ratio",
    ),
    ModelSpec(
        "L3_gated_chirp",
        [
            *STEM,
            "--oob-fusion-type", "gated_oob",
            "--use-chirp-embedding",
            "--oob-norm", "ratio",
        ],
        oob_fusion_type="gated_oob",
        oob_norm="ratio",
        use_chirp=True,
    ),
    ModelSpec(
        "L4_no_oob_chirp",
        [
            *STEM,
            "--no-oob",
            "--oob-fusion-type", "no_oob",
            "--use-chirp-embedding",
            "--oob-norm", "none",
        ],
        oob_fusion_type="no_oob",
        oob_norm="none",
        use_chirp=True,
    ),
]

# L5 added dynamically: winner + oob dropout


@dataclass
class RecipeSpec:
    suffix: str
    extra_train: list[str]
    checkpoint_name: str = "best.pt"
    loss_type: str = "ce"
    checkpoint_metric: str = "acc"


STEP2_RECIPES: list[RecipeSpec] = [
    RecipeSpec("R0_CE_valacc", []),
    RecipeSpec("R1_CE_valmacrof1", ["--checkpoint-metric", "macro_f1"]),
    RecipeSpec("R2_classbalanced_valmacrof1", ["--class-balanced-ce", "--checkpoint-metric", "macro_f1"]),
    RecipeSpec("R3_focal_valmacrof1", ["--loss-type", "focal", "--checkpoint-metric", "macro_f1"]),
    RecipeSpec("R4_CE_valacc_SWA", ["--use-swa"]),
]

# Base model args for recipe ablation (winner defaults to F)
RECIPE_BASE_SPEC = STEP1_CORE[2]  # F_cross_attn_chirp_plain

# ---------------------------------------------------------------------------
# Phase5-clean: strict source-only cross-receiver stress test
# ---------------------------------------------------------------------------
PHASE5_DIRECTIONS: list[tuple[str, str]] = [
    ("rx1_to_rx2", "data/paper/rx1_to_rx2_source_only.csv"),
    ("rx2_to_rx1", "data/paper/rx2_to_rx1_source_only.csv"),
]

PHASE5_SEEDS = [0, 1, 2]

PHASE5_COMMON_SUFFIX = [
    "--batch-size", "128",
    "--samples-per-file", "256",
    "--eval-samples-per-file", "256",
    "--dim", "64",
    "--depth", "2",
    "--device", "cuda",
    "--train-split", "train",
    "--val-split", "val",
    "--eval-split", "test",
    "--input-norm", "iq_rms",
    "--fft-norm", "log_zscore",
]

PHASE5_TRAIN_ONLY = CROSS_DAY_TRAIN_ONLY

PHASE5_CNN = ModelSpec(
    "A_cnn_iq",
    ["--model-type", "osu_cnn", "--cnn-input-type", "iq"],
    oob_fusion_type="n/a",
    oob_norm="n/a",
    model_type="osu_cnn",
)

PHASE5_F_RATIO = ModelSpec(
    "F_cross_attn_chirp_plain",
    [
        *STEM,
        "--oob-fusion-type", "cross_attn_oob",
        "--use-oob-cross-attention",
        "--use-chirp-embedding",
        "--oob-norm", "ratio",
    ],
    oob_fusion_type="cross_attn_oob",
    oob_norm="ratio",
    use_chirp=True,
)

PHASE5_MODELS: list[ModelSpec] = [PHASE5_CNN, PHASE5_F_RATIO]

# Step1b: chirp / fusion ablation (cross-day, zscore, 3 seeds)
STEP1B_MODELS: list[ModelSpec] = [
    ModelSpec(
        "F_cross_attn_no_chirp_plain",
        [
            *STEM,
            "--oob-fusion-type", "cross_attn_oob",
            "--use-oob-cross-attention",
            "--oob-norm", "zscore",
        ],
        oob_fusion_type="cross_attn_oob",
        oob_norm="zscore",
        use_chirp=False,
    ),
    ModelSpec(
        "D_concat_chirp_plain",
        [
            *STEM,
            "--oob-fusion-type", "concat_oob",
            "--use-chirp-embedding",
            "--oob-norm", "zscore",
        ],
        oob_fusion_type="concat_oob",
        oob_norm="zscore",
        use_chirp=True,
    ),
]
