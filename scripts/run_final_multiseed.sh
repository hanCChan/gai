#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
MANIFEST=${MANIFEST:-data/manifest_cross_day_day1_to_day5.csv}
CENTER_LOSS_WEIGHT=${CENTER_LOSS_WEIGHT:-0.01}
RUN_ROOT=${RUN_ROOT:-runs/final_multiseed}
OUT_ROOT=${OUT_ROOT:-outputs/final_multiseed}
SEEDS=${SEEDS:-0,1,2}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs
LOG="logs/final_multiseed_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_step() {
  local name="$1"
  shift
  echo
  echo "==> $(date --iso-8601=seconds) ${name}"
  "$@"
}

eval_final() {
  local method="$1"
  local seed="$2"
  local checkpoint="$3"
  local out_root="${OUT_ROOT}/seed_${seed}/${method}"
  for vote in mean_logits mean_prob; do
    run_step "eval seed${seed} ${method} classifier ${vote}" \
      "${PY}" scripts/evaluate.py \
        --manifest "${MANIFEST}" \
        --checkpoint "${checkpoint}" \
        --samples-per-file 256 \
        --eval-samples-per-file 256 \
        --mode classifier \
        --file-vote-mode "${vote}" \
        --device cuda \
        --out-dir "${out_root}/classifier_${vote}"
  done
  for vote in mean_prob confidence_weighted; do
    run_step "eval seed${seed} ${method} prototype ${vote}" \
      "${PY}" scripts/evaluate.py \
        --manifest "${MANIFEST}" \
        --checkpoint "${checkpoint}" \
        --samples-per-file 256 \
        --eval-samples-per-file 256 \
        --mode prototype \
        --file-vote-mode "${vote}" \
        --device cuda \
        --out-dir "${out_root}/prototype_${vote}"
  done
}

run_seed() {
  local seed="$1"
  run_step "train seed${seed} cnn" \
    "${PY}" scripts/finetune.py \
      --manifest "${MANIFEST}" \
      --model-type osu_cnn \
      --cnn-input-type iq \
      --epochs 80 \
      --batch-size 16 \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --device cuda \
      --seed "${seed}" \
      --out-dir "${RUN_ROOT}/seed_${seed}/cnn"
  eval_final cnn "${seed}" "${RUN_ROOT}/seed_${seed}/cnn/best.pt"

  run_step "train seed${seed} hybrid" \
    "${PY}" scripts/finetune.py \
      --manifest "${MANIFEST}" \
      --epochs 80 \
      --batch-size 16 \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --dim 64 \
      --depth 2 \
      --device cuda \
      --seed "${seed}" \
      --label-smoothing 0.05 \
      --weight-decay 5e-4 \
      --patch-embed-type cnn_stem \
      --cnn-stem-dim 32 \
      --oob-fusion-type cross_attn_oob \
      --use-oob-cross-attention \
      --use-chirp-embedding \
      --use-center-loss \
      --center-loss-weight "${CENTER_LOSS_WEIGHT}" \
      --out-dir "${RUN_ROOT}/seed_${seed}/hybrid"
  eval_final hybrid "${seed}" "${RUN_ROOT}/seed_${seed}/hybrid/best.pt"
}

{
  echo "final_multiseed started at $(date --iso-8601=seconds)"
  echo "manifest=${MANIFEST}"
  echo "center_loss_weight=${CENTER_LOSS_WEIGHT}"
  IFS=',' read -r -a seed_list <<< "${SEEDS}"
  for seed in "${seed_list[@]}"; do
    run_seed "${seed}"
  done
  run_step "summarize final multiseed" "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"
  run_step "mean/std final multiseed" "${PY}" scripts/summarize_mean_std.py --summary "${OUT_ROOT}/summary.csv" --out "${OUT_ROOT}/mean_std.csv"
  echo
  echo "final_multiseed finished at $(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"
