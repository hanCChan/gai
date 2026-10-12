#!/usr/bin/env python3
"""Generate paper_ready_v3 Step1 jobs with separated train/eval commands."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper" / "lib"))

from v3_job_defs import (  # noqa: E402
    CROSS_DAY_COMMON,
    CROSS_DAY_TRAIN_ONLY,
    PHASE5_COMMON_SUFFIX,
    PHASE5_DIRECTIONS,
    PHASE5_MODELS,
    PHASE5_SEEDS,
    PHASE5_TRAIN_ONLY,
    STEP1_CORE,
    STEP1_DIAG,
    STEP1B_MODELS,
    ModelSpec,
)

PREVIEW_FIELDS = [
    "model_id",
    "seed",
    "job_uid",
    "train_cmd",
    "eval_cmd",
    "out_dir",
    "checkpoint_path",
    "model_type",
    "oob_fusion_type",
    "oob_norm",
    "use_chirp",
    "loss_type",
    "checkpoint_metric",
    "lr",
    "batch_size",
    "epochs",
]

EVAL_ONLY_ARGS = [
    "--mode", "classifier",
    "--file-vote-mode", "mean_logits",
]

FORBIDDEN_IN_EVAL = [
    "--loss-type",
    "--checkpoint-metric",
    "--class-balanced-ce",
    "--label-smoothing",
    "--weight-decay",
    "--use-swa",
    "--mixstyle",
    "--oob-dropout",
    "--focal-gamma",
    "--epochs",
    "--lr",
]

FORBIDDEN_IN_CNN_TRAIN = [
    "--oob-fusion-type",
    "--oob-norm",
    "--use-chirp-embedding",
]

SAMPLE_MODELS = [
    "A_cnn_iq",
    "D_concat_oob_plain",
    "F_cross_attn_chirp_plain",
    "H_gated_chirp_plain",
]

EXPECTED_MODEL_COUNTS = {
    "A_cnn_iq": 5,
    "D_concat_oob_plain": 5,
    "F_cross_attn_chirp_plain": 5,
    "H_gated_chirp_plain": 5,
    "B_linear_no_oob": 3,
    "C_cnn_stem_chirp_no_oob": 1,
}

ARCH_CHECKS: dict[str, list[str]] = {
    "A_cnn_iq": ["--model-type osu_cnn", "--cnn-input-type iq"],
    "D_concat_oob_plain": [
        "--model-type rf_hstu",
        "--patch-embed-type cnn_stem",
        "--cnn-stem-dim 32",
        "--oob-fusion-type concat_oob",
        "--oob-norm zscore",
    ],
    "F_cross_attn_chirp_plain": [
        "--model-type rf_hstu",
        "--patch-embed-type cnn_stem",
        "--cnn-stem-dim 32",
        "--oob-fusion-type cross_attn_oob",
        "--use-oob-cross-attention",
        "--use-chirp-embedding",
        "--oob-norm zscore",
    ],
    "H_gated_chirp_plain": [
        "--model-type rf_hstu",
        "--patch-embed-type cnn_stem",
        "--cnn-stem-dim 32",
        "--oob-fusion-type gated_oob",
        "--oob-norm zscore",
        "--use-chirp-embedding",
    ],
}

PHASE5_ARCH_CHECKS: dict[str, list[str]] = {
    "A_cnn_iq": ARCH_CHECKS["A_cnn_iq"],
    "F_cross_attn_chirp_plain": [
        "--model-type rf_hstu",
        "--patch-embed-type cnn_stem",
        "--cnn-stem-dim 32",
        "--oob-fusion-type cross_attn_oob",
        "--use-oob-cross-attention",
        "--use-chirp-embedding",
        "--oob-norm ratio",
    ],
}

STEP1_F_OOB_NORM = "zscore"
PHASE5_F_OOB_NORM = "ratio"

STEP1B_SEEDS = [0, 1, 2]

STEP1B_ARCH_CHECKS: dict[str, list[str]] = {
    "F_cross_attn_no_chirp_plain": [
        "--model-type rf_hstu",
        "--patch-embed-type cnn_stem",
        "--cnn-stem-dim 32",
        "--oob-fusion-type cross_attn_oob",
        "--use-oob-cross-attention",
        "--oob-norm zscore",
    ],
    "D_concat_chirp_plain": [
        "--model-type rf_hstu",
        "--patch-embed-type cnn_stem",
        "--cnn-stem-dim 32",
        "--oob-fusion-type concat_oob",
        "--oob-norm zscore",
        "--use-chirp-embedding",
    ],
}

STEP1B_SAMPLE_MODELS = [
    "F_cross_attn_no_chirp_plain",
    "D_concat_chirp_plain",
]


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:
        return "nogit"


def arg_value(args: list[str], flag: str, default: str = "") -> str:
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    return default


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def build_train_cmd(
    py: str,
    common_args: list[str],
    model_args: list[str],
    train_only_args: list[str],
    seed: int,
    run_dir: str,
) -> str:
    return shell_join(
        [
            py,
            "scripts/finetune.py",
            *common_args,
            *model_args,
            *train_only_args,
            "--seed",
            str(seed),
            "--out-dir",
            run_dir,
        ]
    )


def build_eval_cmd(
    py: str,
    common_args: list[str],
    model_args: list[str],
    eval_only_args: list[str],
    seed: int,
    checkpoint: str,
    out_dir: str,
) -> str:
    return shell_join(
        [
            py,
            "scripts/evaluate.py",
            *common_args,
            *model_args,
            *eval_only_args,
            "--seed",
            str(seed),
            "--checkpoint",
            checkpoint,
            "--out-dir",
            out_dir,
        ]
    )


def build_full_cmd(root: Path, py: str, train_cmd: str, eval_cmd: str) -> str:
    return " && ".join(
        [
            f"cd {shlex.quote(root.as_posix())}",
            f"export PYTHONPATH={shlex.quote((root / 'src').as_posix())}:$PYTHONPATH",
            train_cmd,
            eval_cmd,
        ]
    )


def add_model_jobs(
    rows: list[dict],
    jobs: list[tuple[str, str, str]],
    *,
    py: str,
    root: Path,
    runs_dir: Path,
    outs_dir: Path,
    spec: ModelSpec,
    seeds: list[int],
    common_args: list[str],
    train_only_args: list[str],
    eval_only_args: list[str],
) -> None:
    model_id = spec.job_id
    lr = arg_value(train_only_args, "--lr", "3e-3")
    batch_size = arg_value(common_args, "--batch-size", "128")
    epochs = arg_value(train_only_args, "--epochs", "80")
    loss_type = arg_value(train_only_args, "--loss-type", "ce")
    checkpoint_metric = arg_value(train_only_args, "--checkpoint-metric", "acc")

    for seed in seeds:
        job_uid = f"{model_id}_seed_{seed}"
        run_dir = runs_dir / model_id / f"seed_{seed}"
        out_dir = outs_dir / model_id / f"seed_{seed}"
        checkpoint_path = run_dir / "best.pt"
        marker = out_dir / "file_predictions.csv"

        train_cmd = build_train_cmd(
            str(py), common_args, spec.train_args, train_only_args, seed, run_dir.as_posix()
        )
        eval_cmd = build_eval_cmd(
            str(py),
            common_args,
            spec.train_args,
            eval_only_args,
            seed,
            checkpoint_path.as_posix(),
            out_dir.as_posix(),
        )
        full_cmd = build_full_cmd(root, py, train_cmd, eval_cmd)
        jobs.append((job_uid, full_cmd, marker.as_posix()))

        rows.append(
            {
                "model_id": model_id,
                "seed": str(seed),
                "job_uid": job_uid,
                "train_cmd": train_cmd,
                "eval_cmd": eval_cmd,
                "out_dir": out_dir.as_posix(),
                "checkpoint_path": checkpoint_path.as_posix(),
                "model_type": spec.model_type,
                "oob_fusion_type": spec.oob_fusion_type,
                "oob_norm": spec.oob_norm,
                "use_chirp": spec.use_chirp,
                "loss_type": loss_type,
                "checkpoint_metric": checkpoint_metric,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
            }
        )


def generate_step1(
    root: Path,
    py: str,
    runs_dir: Path,
    outs_dir: Path,
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    rows: list[dict] = []
    jobs: list[tuple[str, str, str]] = []

    for spec in STEP1_CORE:
        add_model_jobs(
            rows,
            jobs,
            py=py,
            root=root,
            runs_dir=runs_dir,
            outs_dir=outs_dir,
            spec=spec,
            seeds=list(range(5)),
            common_args=CROSS_DAY_COMMON,
            train_only_args=CROSS_DAY_TRAIN_ONLY,
            eval_only_args=EVAL_ONLY_ARGS,
        )

    add_model_jobs(
        rows,
        jobs,
        py=py,
        root=root,
        runs_dir=runs_dir,
        outs_dir=outs_dir,
        spec=STEP1_DIAG[0],
        seeds=list(range(3)),
        common_args=CROSS_DAY_COMMON,
        train_only_args=CROSS_DAY_TRAIN_ONLY,
        eval_only_args=EVAL_ONLY_ARGS,
    )

    add_model_jobs(
        rows,
        jobs,
        py=py,
        root=root,
        runs_dir=runs_dir,
        outs_dir=outs_dir,
        spec=STEP1_DIAG[1],
        seeds=[0],
        common_args=CROSS_DAY_COMMON,
        train_only_args=CROSS_DAY_TRAIN_ONLY,
        eval_only_args=EVAL_ONLY_ARGS,
    )

    return rows, jobs


def add_phase5_jobs(
    rows: list[dict],
    jobs: list[tuple[str, str, str]],
    *,
    py: str,
    root: Path,
    runs_dir: Path,
    outs_dir: Path,
    spec: ModelSpec,
    direction: str,
    manifest: str,
    seeds: list[int],
    train_only_args: list[str],
    eval_only_args: list[str],
) -> None:
    model_id = spec.job_id
    common_args = ["--manifest", manifest, *PHASE5_COMMON_SUFFIX]
    lr = arg_value(train_only_args, "--lr", "3e-3")
    batch_size = arg_value(common_args, "--batch-size", "128")
    epochs = arg_value(train_only_args, "--epochs", "80")
    loss_type = arg_value(train_only_args, "--loss-type", "ce")
    checkpoint_metric = arg_value(train_only_args, "--checkpoint-metric", "acc")

    for seed in seeds:
        job_uid = f"{model_id}_{direction}_seed_{seed}"
        run_dir = runs_dir / model_id / direction / f"seed_{seed}"
        out_dir = outs_dir / model_id / direction / f"seed_{seed}"
        checkpoint_path = run_dir / "best.pt"
        marker = out_dir / "file_predictions.csv"

        train_cmd = build_train_cmd(
            str(py), common_args, spec.train_args, train_only_args, seed, run_dir.as_posix()
        )
        eval_cmd = build_eval_cmd(
            str(py),
            common_args,
            spec.train_args,
            eval_only_args,
            seed,
            checkpoint_path.as_posix(),
            out_dir.as_posix(),
        )
        full_cmd = build_full_cmd(root, py, train_cmd, eval_cmd)
        jobs.append((job_uid, full_cmd, marker.as_posix()))

        rows.append(
            {
                "model_id": model_id,
                "direction": direction,
                "seed": str(seed),
                "job_uid": job_uid,
                "train_cmd": train_cmd,
                "eval_cmd": eval_cmd,
                "out_dir": out_dir.as_posix(),
                "checkpoint_path": checkpoint_path.as_posix(),
                "manifest": manifest,
                "model_type": spec.model_type,
                "oob_fusion_type": spec.oob_fusion_type,
                "oob_norm": spec.oob_norm,
                "use_chirp": spec.use_chirp,
                "loss_type": loss_type,
                "checkpoint_metric": checkpoint_metric,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
            }
        )


def generate_phase5_clean(
    root: Path,
    py: str,
    runs_dir: Path,
    outs_dir: Path,
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    rows: list[dict] = []
    jobs: list[tuple[str, str, str]] = []

    for direction, manifest in PHASE5_DIRECTIONS:
        for spec in PHASE5_MODELS:
            add_phase5_jobs(
                rows,
                jobs,
                py=py,
                root=root,
                runs_dir=runs_dir,
                outs_dir=outs_dir,
                spec=spec,
                direction=direction,
                manifest=manifest,
                seeds=PHASE5_SEEDS,
                train_only_args=PHASE5_TRAIN_ONLY,
                eval_only_args=EVAL_ONLY_ARGS,
            )

    return rows, jobs


def generate_step1b(
    root: Path,
    py: str,
    runs_dir: Path,
    outs_dir: Path,
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    rows: list[dict] = []
    jobs: list[tuple[str, str, str]] = []

    for spec in STEP1B_MODELS:
        add_model_jobs(
            rows,
            jobs,
            py=py,
            root=root,
            runs_dir=runs_dir,
            outs_dir=outs_dir,
            spec=spec,
            seeds=STEP1B_SEEDS,
            common_args=CROSS_DAY_COMMON,
            train_only_args=CROSS_DAY_TRAIN_ONLY,
            eval_only_args=EVAL_ONLY_ARGS,
        )

    return rows, jobs


def validate_step1b_rows(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_out: set[str] = set()
    seen_uid: set[str] = set()

    if len(rows) != 6:
        errors.append(f"expected 6 step1b jobs, got {len(rows)}")

    from collections import Counter

    model_counts = Counter(r["model_id"] for r in rows)
    for model_id in STEP1B_SAMPLE_MODELS:
        if model_counts.get(model_id) != 3:
            errors.append(
                f"{model_id} count expected 3, got {model_counts.get(model_id)}"
            )

    for row in rows:
        model_id = row["model_id"]
        seed = row["seed"]
        job_uid = row["job_uid"]
        train_cmd = row["train_cmd"]
        eval_cmd = row["eval_cmd"]
        out_dir = row["out_dir"]

        if job_uid != f"{model_id}_seed_{seed}":
            errors.append(f"{job_uid}: job_uid must equal {{model_id}}_seed_{{seed}}")

        if f"/{model_id}/seed_{seed}" not in out_dir:
            errors.append(f"{job_uid}: out_dir must contain model_id and seed")
        if out_dir in seen_out:
            errors.append(f"{job_uid}: duplicate out_dir {out_dir}")
        seen_out.add(out_dir)

        if job_uid in seen_uid:
            errors.append(f"{job_uid}: duplicate job_uid")
        seen_uid.add(job_uid)

        for tok in FORBIDDEN_IN_EVAL:
            if tok in eval_cmd:
                errors.append(f"{job_uid}: eval_cmd contains forbidden {tok}")

        for flag in STEP1B_ARCH_CHECKS.get(model_id, []):
            if flag not in train_cmd:
                errors.append(f"{job_uid}: train_cmd missing {flag}")
            if flag not in eval_cmd:
                errors.append(f"{job_uid}: eval_cmd missing {flag}")

        if model_id == "F_cross_attn_no_chirp_plain":
            if "--use-chirp-embedding" in train_cmd or "--use-chirp-embedding" in eval_cmd:
                errors.append(f"{job_uid}: F_no_chirp must not use chirp embedding")

        if "--oob-norm ratio" in train_cmd or "--oob-norm ratio" in eval_cmd:
            errors.append(f"{job_uid}: Step1b must use zscore, not ratio")

    return errors


def validate_phase5_rows(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_out: set[str] = set()
    seen_uid: set[str] = set()

    if len(rows) != 12:
        errors.append(f"expected 12 phase5_clean jobs, got {len(rows)}")

    for row in rows:
        model_id = row["model_id"]
        direction = row["direction"]
        seed = row["seed"]
        job_uid = row["job_uid"]
        train_cmd = row["train_cmd"]
        eval_cmd = row["eval_cmd"]
        out_dir = row["out_dir"]
        ckpt = row["checkpoint_path"]
        manifest = row["manifest"]

        expected_uid = f"{model_id}_{direction}_seed_{seed}"
        if job_uid != expected_uid:
            errors.append(f"{job_uid}: job_uid must equal {{model_id}}_{{direction}}_seed_{{seed}}")

        if f"/{model_id}/{direction}/seed_{seed}" not in out_dir:
            errors.append(f"{job_uid}: out_dir must contain model_id, direction, seed")
        if out_dir in seen_out:
            errors.append(f"{job_uid}: duplicate out_dir {out_dir}")
        seen_out.add(out_dir)

        if job_uid in seen_uid:
            errors.append(f"{job_uid}: duplicate job_uid")
        seen_uid.add(job_uid)

        if manifest not in train_cmd or manifest not in eval_cmd:
            errors.append(f"{job_uid}: manifest missing from train/eval cmd")

        if "--eval-split test" not in train_cmd:
            errors.append(f"{job_uid}: must use eval-split test")

        for tok in FORBIDDEN_IN_EVAL:
            if tok in eval_cmd:
                errors.append(f"{job_uid}: eval_cmd contains forbidden {tok}")

        if model_id == "A_cnn_iq":
            for tok in FORBIDDEN_IN_CNN_TRAIN:
                if tok in train_cmd:
                    errors.append(f"{job_uid}: CNN train_cmd contains forbidden {tok}")

        for flag in PHASE5_ARCH_CHECKS.get(model_id, []):
            if flag not in train_cmd:
                errors.append(f"{job_uid}: train_cmd missing {flag}")
            if flag not in eval_cmd:
                errors.append(f"{job_uid}: eval_cmd missing {flag}")

        if model_id == "F_cross_attn_chirp_plain":
            if "--oob-norm zscore" in train_cmd or "--oob-norm zscore" in eval_cmd:
                errors.append(f"{job_uid}: Phase5 F must use ratio, not zscore")

    return errors


def validate_rows(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_out: set[str] = set()
    seen_uid: set[str] = set()

    from collections import Counter

    model_counts = Counter(r["model_id"] for r in rows)
    for model_id, expected in EXPECTED_MODEL_COUNTS.items():
        if model_counts.get(model_id) != expected:
            errors.append(
                f"{model_id} count expected {expected}, got {model_counts.get(model_id)}"
            )

    for row in rows:
        model_id = row["model_id"]
        seed = row["seed"]
        job_uid = row["job_uid"]
        train_cmd = row["train_cmd"]
        eval_cmd = row["eval_cmd"]
        out_dir = row["out_dir"]
        ckpt = row["checkpoint_path"]

        if job_uid != f"{model_id}_seed_{seed}":
            errors.append(f"{job_uid}: job_uid must equal {{model_id}}_seed_{{seed}}")

        if f"/{model_id}/seed_{seed}" not in out_dir:
            errors.append(f"{job_uid}: out_dir must contain model_id and seed")
        if out_dir in seen_out:
            errors.append(f"{job_uid}: duplicate out_dir {out_dir}")
        seen_out.add(out_dir)

        if job_uid in seen_uid:
            errors.append(f"{job_uid}: duplicate job_uid")
        seen_uid.add(job_uid)

        if not ckpt.endswith(".pt"):
            errors.append(f"{job_uid}: checkpoint_path must end with .pt")

        if "--checkpoint " not in eval_cmd:
            errors.append(f"{job_uid}: eval missing --checkpoint")

        for tok in FORBIDDEN_IN_EVAL:
            if tok in eval_cmd:
                errors.append(f"{job_uid}: eval_cmd contains forbidden {tok}")

        if model_id == "A_cnn_iq":
            for tok in FORBIDDEN_IN_CNN_TRAIN:
                if tok in train_cmd:
                    errors.append(f"{job_uid}: CNN train_cmd contains forbidden {tok}")

        for flag in ARCH_CHECKS.get(model_id, []):
            if flag not in train_cmd:
                errors.append(f"{job_uid}: train_cmd missing {flag}")
            if flag not in eval_cmd:
                errors.append(f"{job_uid}: eval_cmd missing {flag}")

        if model_id.startswith(("D_", "F_", "H_")):
            if "--oob-norm ratio" in train_cmd or "--oob-norm ratio" in eval_cmd:
                errors.append(f"{job_uid}: Step1 must use zscore, not ratio")

    if len(seen_out) != len(rows):
        errors.append("out_dir not unique across all rows")
    if len(seen_uid) != len(rows):
        errors.append("job_uid not unique across all rows")

    return errors


def write_preview_tsv(rows: list[dict], path: Path, fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or PREVIEW_FIELDS
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_jobs_tsv(jobs: list[tuple[str, str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for job_uid, cmd, marker in jobs:
            f.write(f"{job_uid}\t{cmd}\t{marker}\n")


def write_run_script(out_dir: Path, root: Path, step: str) -> Path:
    if step == "phase5_clean":
        script = out_dir / "run_phase5_clean_cross_receiver.sh"
        jobs_rel = "outputs/paper_ready_v3/phase5_clean_cross_receiver/step0_audit/jobs.tsv"
        log_dir = "outputs/paper_ready_v3/phase5_clean_cross_receiver/logs/train_jobs"
        title = "Phase5-clean cross-receiver"
    elif step == "step1b":
        script = out_dir / "run_step1b_chirp_fusion_ablation.sh"
        jobs_rel = "outputs/paper_ready_v3/step1b_chirp_fusion_ablation/step0_audit/jobs.tsv"
        log_dir = "outputs/paper_ready_v3/step1b_chirp_fusion_ablation/logs/train_jobs"
        title = "Step1b chirp/fusion ablation"
    else:
        script = out_dir / "run_step1_phase7_clean.sh"
        jobs_rel = "outputs/paper_ready_v3/step0_audit/jobs.tsv"
        log_dir = "outputs/paper_ready_v3/step1_phase7_clean/logs/train_jobs"
        title = "Step1"

    content = f"""#!/usr/bin/env bash
# Auto-generated by generate_paper_v3_jobs.py — do not edit by hand.
set -euo pipefail

ROOT=${{ROOT:-{root}}}
PY=${{PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}}
export GPUS=${{GPUS:-1,2,3,4,5,6}}

cd "${{ROOT}}"
export PYTHONPATH="${{ROOT}}/src:${{PYTHONPATH:-}}"
source scripts/paper/lib/paper_env.sh
source scripts/paper/lib/job_helpers.sh

JOBS_FILE="${{ROOT}}/{jobs_rel}"
LOG_DIR="${{ROOT}}/{log_dir}"
mkdir -p "${{LOG_DIR}}"

echo "Launching {title} from ${{JOBS_FILE}} on GPUs ${{GPUS}}"
mgpu_run_jobs "${{JOBS_FILE}}" "${{LOG_DIR}}"
"""
    script.write_text(content, encoding="utf-8")
    script.chmod(0o755)
    return script


PREVIEW_FIELDS_PHASE5 = PREVIEW_FIELDS + ["direction", "manifest"]


def print_sample_step1b_commands(rows: list[dict]) -> None:
    print("\n=== Step1b sample commands (seed=0) ===")
    for model_id in STEP1B_SAMPLE_MODELS:
        match = next(
            (r for r in rows if r["model_id"] == model_id and r["seed"] == "0"),
            None,
        )
        if match is None:
            print(f"\n--- {model_id} seed 0: NOT FOUND ---")
            continue
        print(f"\n--- {model_id} ({match['job_uid']}) ---")
        print(f"TRAIN:\n  {match['train_cmd']}")
        print(f"EVAL:\n  {match['eval_cmd']}")
        has_chirp = "--use-chirp-embedding" in match["train_cmd"]
        print(f"use_chirp: {has_chirp}")


def print_sample_phase5_commands(rows: list[dict]) -> None:
    print("\n=== Phase5-clean sample commands (seed=0) ===")
    for direction in ("rx1_to_rx2", "rx2_to_rx1"):
        for model_id in ("A_cnn_iq", "F_cross_attn_chirp_plain"):
            match = next(
                (
                    r
                    for r in rows
                    if r["model_id"] == model_id
                    and r["direction"] == direction
                    and r["seed"] == "0"
                ),
                None,
            )
            if match is None:
                print(f"\n--- {model_id} {direction} seed 0: NOT FOUND ---")
                continue
            print(f"\n--- {model_id} {direction} ({match['job_uid']}) ---")
            print(f"TRAIN:\n  {match['train_cmd']}")
            print(f"EVAL:\n  {match['eval_cmd']}")


def print_sample_commands(rows: list[dict]) -> None:
    print("\n=== Sample commands for manual review (seed=0) ===")
    for model_id in SAMPLE_MODELS:
        match = next(
            (r for r in rows if r["model_id"] == model_id and r["seed"] == "0"),
            None,
        )
        if match is None:
            print(f"\n--- {model_id} seed 0: NOT FOUND ---")
            continue
        print(f"\n--- {model_id} ({match['job_uid']}) ---")
        print(f"TRAIN:\n  {match['train_cmd']}")
        print(f"EVAL:\n  {match['eval_cmd']}")
        print(f"out_dir: {match['out_dir']}")
        print(f"checkpoint_path: {match['checkpoint_path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper_ready_v3 jobs")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--py",
        default="/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python",
    )
    parser.add_argument(
        "--step",
        choices=["step1", "phase5_clean", "step1b"],
        default="step1",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Audit dir for jobs_preview.tsv and jobs.tsv",
    )
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--outputs-dir", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate preview/jobs files only; do not launch training",
    )
    args = parser.parse_args()

    root = Path(args.root)

    if args.step == "phase5_clean":
        base = root / "outputs/paper_ready_v3/phase5_clean_cross_receiver"
        out_dir = Path(args.out_dir) if args.out_dir else base / "step0_audit"
        runs_dir = Path(args.runs_dir) if args.runs_dir else base / "runs"
        outs_dir = Path(args.outputs_dir) if args.outputs_dir else base / "outputs"
        rows, jobs = generate_phase5_clean(root, args.py, runs_dir, outs_dir)
        errors = validate_phase5_rows(rows)
        preview_fields = PREVIEW_FIELDS_PHASE5
        sample_fn = print_sample_phase5_commands
    elif args.step == "step1b":
        base = root / "outputs/paper_ready_v3/step1b_chirp_fusion_ablation"
        out_dir = Path(args.out_dir) if args.out_dir else base / "step0_audit"
        runs_dir = Path(args.runs_dir) if args.runs_dir else base / "runs"
        outs_dir = Path(args.outputs_dir) if args.outputs_dir else base / "outputs"
        rows, jobs = generate_step1b(root, args.py, runs_dir, outs_dir)
        errors = validate_step1b_rows(rows)
        preview_fields = PREVIEW_FIELDS
        sample_fn = print_sample_step1b_commands
    else:
        out_dir = Path(args.out_dir) if args.out_dir else root / "outputs/paper_ready_v3/step0_audit"
        runs_dir = (
            Path(args.runs_dir)
            if args.runs_dir
            else root / "outputs/paper_ready_v3/step1_phase7_clean/runs"
        )
        outs_dir = (
            Path(args.outputs_dir)
            if args.outputs_dir
            else root / "outputs/paper_ready_v3/step1_phase7_clean/outputs"
        )
        rows, jobs = generate_step1(root, args.py, runs_dir, outs_dir)
        errors = validate_rows(rows)
        preview_fields = PREVIEW_FIELDS
        sample_fn = print_sample_commands

    if not out_dir.is_absolute():
        out_dir = root / out_dir
    if not runs_dir.is_absolute():
        runs_dir = root / runs_dir
    if not outs_dir.is_absolute():
        outs_dir = root / outs_dir

    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    write_preview_tsv(rows, out_dir / "jobs_preview.tsv", preview_fields)
    write_jobs_tsv(jobs, out_dir / "jobs.tsv")
    run_script = write_run_script(out_dir, root, args.step)

    meta = {
        "commit": git_commit(root),
        "step": args.step,
        "n_jobs": len(jobs),
        "dry_run": args.dry_run,
        "runs_dir": runs_dir.as_posix(),
        "outputs_dir": outs_dir.as_posix(),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"generated {len(jobs)} {args.step} jobs")
    print(f"preview -> {out_dir / 'jobs_preview.tsv'}")
    print(f"jobs    -> {out_dir / 'jobs.tsv'}")
    print(f"runner  -> {run_script}")
    sample_fn(rows)

    if args.dry_run:
        print("\nDRY_RUN=1 (files written, training NOT launched)")
    else:
        print(f"\nNOTE: pass --dry-run to skip training; launch via {run_script.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
