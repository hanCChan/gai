#!/usr/bin/env bash
# Cross-receiver RFFI: CNN-IQ baseline vs center_none Hybrid, both directions (RX1<->RX2).
# Indoor SameTx, 24 clean classes. No center loss / multiscale / supcon / hard margin / adversarial / TTA.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
RUN_ROOT=${RUN_ROOT:-runs/cross_receiver}
OUT_ROOT=${OUT_ROOT:-outputs/cross_receiver}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs "${OUT_ROOT}"
LOG="logs/cross_receiver_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_step() {
  local name="$1"
  shift
  echo
  echo "==> $(date --iso-8601=seconds) ${name}"
  "$@"
}

eval_model() {
  local method="$1"
  local direction="$2"
  local manifest="$3"
  local checkpoint="$4"
  local out_root="${OUT_ROOT}/${direction}/${method}"
  for vote in mean_logits confidence_weighted; do
    run_step "eval ${direction} ${method} classifier ${vote}" \
      "${PY}" scripts/evaluate.py \
        --manifest "${manifest}" \
        --checkpoint "${checkpoint}" \
        --samples-per-file 256 \
        --eval-samples-per-file 256 \
        --mode classifier \
        --file-vote-mode "${vote}" \
        --device cuda \
        --out-dir "${out_root}/classifier_${vote}"
  done
  run_step "eval ${direction} ${method} prototype mean_prob" \
    "${PY}" scripts/evaluate.py \
      --manifest "${manifest}" \
      --checkpoint "${checkpoint}" \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --mode prototype \
      --file-vote-mode mean_prob \
      --device cuda \
      --out-dir "${out_root}/prototype_mean_prob"
}

train_direction() {
  local direction="$1"
  local manifest="$2"
  run_step "check manifest ${direction}" "${PY}" scripts/check_manifest.py --manifest "${manifest}"

  run_step "train cnn ${direction}" \
    "${PY}" scripts/finetune.py \
      --manifest "${manifest}" \
      --model-type osu_cnn \
      --cnn-input-type iq \
      --epochs 80 \
      --batch-size 16 \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --device cuda \
      --label-smoothing 0.05 \
      --weight-decay 5e-4 \
      --out-dir "${RUN_ROOT}/${direction}/cnn"
  eval_model cnn "${direction}" "${manifest}" "${RUN_ROOT}/${direction}/cnn/best.pt"

  # Hybrid center_none main line: do NOT pass --use-center-loss
  run_step "train hybrid ${direction}" \
    "${PY}" scripts/finetune.py \
      --manifest "${manifest}" \
      --epochs 80 \
      --batch-size 16 \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --dim 64 \
      --depth 2 \
      --device cuda \
      --label-smoothing 0.05 \
      --weight-decay 5e-4 \
      --patch-embed-type cnn_stem \
      --cnn-stem-dim 32 \
      --oob-fusion-type cross_attn_oob \
      --use-oob-cross-attention \
      --use-chirp-embedding \
      --out-dir "${RUN_ROOT}/${direction}/hybrid"
  eval_model hybrid "${direction}" "${manifest}" "${RUN_ROOT}/${direction}/hybrid/best.pt"
}

paired_compare() {
  local direction="$1"
  run_step "paired ${direction}" \
    "${PY}" scripts/paired_compare_models.py \
      --a-pred "${OUT_ROOT}/${direction}/cnn/classifier_mean_logits/predictions.csv" \
      --b-pred "${OUT_ROOT}/${direction}/hybrid/classifier_mean_logits/predictions.csv" \
      --a-name CNN-IQ \
      --b-name Hybrid \
      --out "${OUT_ROOT}/paired_${direction}.csv" \
      --diff-out "${OUT_ROOT}/paired_${direction}_diff_files.csv"
}

{
  echo "cross_receiver started at $(date --iso-8601=seconds)"
  echo "gpu_id=${GPU_ID} center_loss=none main_line=center_none"
  train_direction rx1_to_rx2 data/manifest_rx1_to_rx2.csv
  train_direction rx2_to_rx1 data/manifest_rx2_to_rx1.csv

  run_step "summarize cross_receiver" "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"
  paired_compare rx1_to_rx2
  paired_compare rx2_to_rx1

  echo
  echo "cross_receiver finished at $(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"
