#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
CENTER_LOSS_WEIGHT=${CENTER_LOSS_WEIGHT:-0.01}
RUN_ROOT=${RUN_ROOT:-runs/lodo_day1to5}
OUT_ROOT=${OUT_ROOT:-outputs/lodo_day1to5}
MANIFEST_ROOT=${MANIFEST_ROOT:-data/lodo_day1to5}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs "${MANIFEST_ROOT}"
LOG="logs/lodo_day1to5_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_step() {
  local name="$1"
  shift
  echo
  echo "==> $(date --iso-8601=seconds) ${name}"
  "$@"
}

eval_all() {
  local method="$1"
  local day="$2"
  local manifest="$3"
  local checkpoint="$4"
  local out_root="${OUT_ROOT}/${method}/test_day_${day}"
  for vote in mean_logits mean_prob confidence_weighted; do
    run_step "eval ${method} day${day} classifier ${vote}" \
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
  for vote in mean_prob confidence_weighted; do
    run_step "eval ${method} day${day} prototype ${vote}" \
      "${PY}" scripts/evaluate.py \
        --manifest "${manifest}" \
        --checkpoint "${checkpoint}" \
        --samples-per-file 256 \
        --eval-samples-per-file 256 \
        --mode prototype \
        --file-vote-mode "${vote}" \
        --device cuda \
        --out-dir "${out_root}/prototype_${vote}"
  done
}

train_fold() {
  local day="$1"
  local train_days="$2"
  local manifest="${MANIFEST_ROOT}/test_day_${day}.csv"
  run_step "generate manifest test_day_${day}" \
    "${PY}" scripts/generate_manifest_days.py \
      --train-days "${train_days}" \
      --val-days "${day}" \
      --out "${manifest}"
  run_step "check manifest test_day_${day}" "${PY}" scripts/check_manifest.py --manifest "${manifest}"

  run_step "train cnn test_day_${day}" \
    "${PY}" scripts/finetune.py \
      --manifest "${manifest}" \
      --model-type osu_cnn \
      --cnn-input-type iq \
      --epochs 80 \
      --batch-size 16 \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --device cuda \
      --out-dir "${RUN_ROOT}/cnn/test_day_${day}"
  eval_all cnn "${day}" "${manifest}" "${RUN_ROOT}/cnn/test_day_${day}/best.pt"

  local center_args=()
  if awk "BEGIN{exit !(${CENTER_LOSS_WEIGHT} > 0)}"; then
    center_args=(--use-center-loss --center-loss-weight "${CENTER_LOSS_WEIGHT}")
    echo "center loss enabled: weight=${CENTER_LOSS_WEIGHT}"
  else
    echo "center loss disabled (CENTER_LOSS_WEIGHT=${CENTER_LOSS_WEIGHT}); center_none main line"
  fi

  run_step "train hybrid test_day_${day}" \
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
      ${center_args[@]+"${center_args[@]}"} \
      --out-dir "${RUN_ROOT}/hybrid/test_day_${day}"
  eval_all hybrid "${day}" "${manifest}" "${RUN_ROOT}/hybrid/test_day_${day}/best.pt"
}

{
  echo "lodo_day1to5 started at $(date --iso-8601=seconds)"
  echo "gpu_id=${GPU_ID}"
  echo "center_loss_weight=${CENTER_LOSS_WEIGHT}"
  train_fold 1 "2,3,4,5"
  train_fold 2 "1,3,4,5"
  train_fold 3 "1,2,4,5"
  train_fold 4 "1,2,3,5"
  train_fold 5 "1,2,3,4"
  run_step "summarize lodo" "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"
  run_step "mean/std lodo" "${PY}" scripts/summarize_mean_std.py --summary "${OUT_ROOT}/summary.csv" --out "${OUT_ROOT}/lodo_mean_std.csv"
  echo
  echo "lodo_day1to5 finished at $(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"
