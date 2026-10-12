#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
MANIFEST=${MANIFEST:-data/manifest_cross_day_day1_to_day5.csv}
RUN_ROOT=${RUN_ROOT:-runs/center_sweep_day1to5}
OUT_ROOT=${OUT_ROOT:-outputs/center_sweep_day1to5}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs
LOG="logs/center_loss_sweep_day1to5_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

run_step() {
  local name="$1"
  shift
  echo
  echo "==> $(date --iso-8601=seconds) ${name}"
  "$@"
}

eval_checkpoint() {
  local name="$1"
  local run_dir="${RUN_ROOT}/${name}"
  local out_root="${OUT_ROOT}/${name}"
  for vote in mean_logits mean_prob confidence_weighted; do
    run_step "evaluate ${name} classifier ${vote}" \
      "${PY}" scripts/evaluate.py \
        --manifest "${MANIFEST}" \
        --checkpoint "${run_dir}/best.pt" \
        --samples-per-file 256 \
        --eval-samples-per-file 256 \
        --mode classifier \
        --file-vote-mode "${vote}" \
        --device cuda \
        --out-dir "${out_root}/classifier_${vote}"
  done
  for vote in mean_prob confidence_weighted; do
    run_step "evaluate ${name} prototype ${vote}" \
      "${PY}" scripts/evaluate.py \
        --manifest "${MANIFEST}" \
        --checkpoint "${run_dir}/best.pt" \
        --samples-per-file 256 \
        --eval-samples-per-file 256 \
        --mode prototype \
        --file-vote-mode "${vote}" \
        --device cuda \
        --out-dir "${out_root}/prototype_${vote}"
  done
}

train_experiment() {
  local name="$1"
  shift
  run_step "finetune ${name}" \
    "${PY}" scripts/finetune.py \
      --manifest "${MANIFEST}" \
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
      --out-dir "${RUN_ROOT}/${name}" \
      "$@"
  eval_checkpoint "${name}"
}

{
  echo "center_loss_sweep_day1to5 started at $(date --iso-8601=seconds)"
  echo "root=${ROOT}"
  echo "python=${PY}"
  echo "gpu_id=${GPU_ID}"
  echo "manifest=${MANIFEST}"

  run_step "manifest check" "${PY}" scripts/check_manifest.py --manifest "${MANIFEST}"
  train_experiment center_none
  train_experiment center_w0001 --use-center-loss --center-loss-weight 0.001
  train_experiment center_w0005 --use-center-loss --center-loss-weight 0.005
  train_experiment center_w001 --use-center-loss --center-loss-weight 0.01

  run_step "summarize center sweep" "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"

  echo
  echo "center_loss_sweep_day1to5 finished at $(date --iso-8601=seconds)"
  echo "summary=${OUT_ROOT}/summary.csv"
} 2>&1 | tee "${LOG}"
