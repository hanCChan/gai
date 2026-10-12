#!/usr/bin/env bash
# Quick cross-receiver rerun: CNN iq_rms vs Hybrid oob_ratio, both directions, 30 epochs.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
RUN_ROOT=${RUN_ROOT:-runs/cross_receiver_quick_30ep}
OUT_ROOT=${OUT_ROOT:-outputs/cross_receiver_quick_30ep}

EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs "${OUT_ROOT}"
LOG="logs/cross_receiver_quick_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

COMMON_TRAIN=(
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --samples-per-file 256
  --eval-samples-per-file 256
  --num-workers "${NUM_WORKERS}"
  --dim 64
  --depth 2
  --device cuda
  --label-smoothing 0.05
  --weight-decay 5e-4
)
HYBRID_ARCH=(
  --patch-embed-type cnn_stem
  --cnn-stem-dim 32
  --oob-fusion-type cross_attn_oob
  --use-oob-cross-attention
  --use-chirp-embedding
)
DIRECTIONS=(
  "rx1_to_rx2:data/manifest_rx1_to_rx2.csv"
  "rx2_to_rx1:data/manifest_rx2_to_rx1.csv"
)

run_step() {
  local name="$1"; shift
  echo; echo "==> $(date --iso-8601=seconds) ${name}"
  "$@"
}

train_and_eval() {
  local direction="$1" manifest="$2" name="$3"
  shift 3
  local ckpt_dir="${RUN_ROOT}/${direction}/${name}"
  local out_dir="${OUT_ROOT}/${direction}/${name}"

  run_step "train ${direction}/${name}" \
    "${PY}" scripts/finetune.py --manifest "${manifest}" "$@" \
      "${COMMON_TRAIN[@]}" --out-dir "${ckpt_dir}"

  run_step "eval ${direction}/${name}" \
    "${PY}" scripts/evaluate.py --manifest "${manifest}" \
      --checkpoint "${ckpt_dir}/best.pt" \
      --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
      --samples-per-file 256 --eval-samples-per-file 256 \
      --mode classifier --file-vote-mode mean_logits \
      --device cuda --out-dir "${out_dir}/classifier_mean_logits"
}

paired_compare() {
  local direction="$1"
  run_step "paired ${direction}" \
    "${PY}" scripts/paired_compare_models.py \
      --a-pred "${OUT_ROOT}/${direction}/cnn_iq_rms/classifier_mean_logits/predictions.csv" \
      --b-pred "${OUT_ROOT}/${direction}/hybrid_oob_ratio/classifier_mean_logits/predictions.csv" \
      --a-name CNN-iq_rms \
      --b-name Hybrid-oob_ratio \
      --out "${OUT_ROOT}/paired_${direction}.csv" \
      --diff-out "${OUT_ROOT}/paired_${direction}_diff_files.csv"
}

{
  echo "cross_receiver_quick started at $(date --iso-8601=seconds)"
  echo "gpu_id=${GPU_ID} epochs=${EPOCHS} batch_size=${BATCH_SIZE}"

  for dir_entry in "${DIRECTIONS[@]}"; do
    direction="${dir_entry%%:*}"
    manifest="${dir_entry#*:}"
    run_step "check manifest ${direction}" "${PY}" scripts/check_manifest.py --manifest "${manifest}"

    train_and_eval "${direction}" "${manifest}" cnn_iq_rms \
      --model-type osu_cnn --cnn-input-type iq \
      --input-norm iq_rms --fft-norm none --oob-norm none

    train_and_eval "${direction}" "${manifest}" hybrid_oob_ratio \
      "${HYBRID_ARCH[@]}" \
      --input-norm iq_rms --fft-norm log_zscore --oob-norm ratio
  done

  run_step "summarize" "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"
  paired_compare rx1_to_rx2
  paired_compare rx2_to_rx1

  echo
  echo "cross_receiver_quick finished at $(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"
