#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-4}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs

LOG="logs/single_gpu_compare_${RUN_TAG}_gpu${GPU_ID}.log"
MANIFEST="${MANIFEST:-data/manifest_cross_day_day1_to_day5.csv}"
RUN_ROOT="runs/single_gpu_compare_${RUN_TAG}"
OUT_ROOT="outputs/single_gpu_compare_${RUN_TAG}"

run_step() {
  local name="$1"
  shift
  echo
  echo "==> $(date --iso-8601=seconds) ${name}"
  "$@"
}

{
  echo "single_gpu_compare started at $(date --iso-8601=seconds)"
  echo "root=${ROOT}"
  echo "python=${PY}"
  echo "gpu_id=${GPU_ID}"
  echo "run_tag=${RUN_TAG}"
  echo "manifest=${MANIFEST}"

  run_step "cuda check" "${PY}" - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0=", torch.cuda.get_device_name(0))
PY

  run_step "finetune osu_cnn_iq" \
    "${PY}" scripts/finetune.py \
      --manifest "${MANIFEST}" \
      --model-type osu_cnn \
      --cnn-input-type iq \
      --epochs 80 \
      --batch-size 16 \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --device cuda \
      --out-dir "${RUN_ROOT}/osu_cnn_iq"

  run_step "evaluate osu_cnn_iq classifier" \
    "${PY}" scripts/evaluate.py \
      --manifest "${MANIFEST}" \
      --checkpoint "${RUN_ROOT}/osu_cnn_iq/best.pt" \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --mode classifier \
      --device cuda \
      --out-dir "${OUT_ROOT}/osu_cnn_iq/classifier"

  run_step "evaluate osu_cnn_iq prototype" \
    "${PY}" scripts/evaluate.py \
      --manifest "${MANIFEST}" \
      --checkpoint "${RUN_ROOT}/osu_cnn_iq/best.pt" \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --mode prototype \
      --file-vote-mode mean_prob \
      --device cuda \
      --out-dir "${OUT_ROOT}/osu_cnn_iq/prototype_mean_prob"

  run_step "finetune hybrid_cnnstem_cross_attn_chirp" \
    "${PY}" scripts/finetune.py \
      --manifest "${MANIFEST}" \
      --epochs 80 \
      --batch-size 16 \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --dim 64 \
      --depth 2 \
      --label-smoothing 0.05 \
      --weight-decay 5e-4 \
      --patch-embed-type cnn_stem \
      --cnn-stem-dim 32 \
      --oob-fusion-type cross_attn_oob \
      --use-oob-cross-attention \
      --use-chirp-embedding \
      --device cuda \
      --out-dir "${RUN_ROOT}/hybrid_cnnstem_cross_attn_chirp"

  run_step "evaluate hybrid classifier" \
    "${PY}" scripts/evaluate.py \
      --manifest "${MANIFEST}" \
      --checkpoint "${RUN_ROOT}/hybrid_cnnstem_cross_attn_chirp/best.pt" \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --mode classifier \
      --device cuda \
      --out-dir "${OUT_ROOT}/hybrid_cnnstem_cross_attn_chirp/classifier"

  run_step "evaluate hybrid prototype" \
    "${PY}" scripts/evaluate.py \
      --manifest "${MANIFEST}" \
      --checkpoint "${RUN_ROOT}/hybrid_cnnstem_cross_attn_chirp/best.pt" \
      --samples-per-file 256 \
      --eval-samples-per-file 256 \
      --mode prototype \
      --file-vote-mode mean_prob \
      --device cuda \
      --out-dir "${OUT_ROOT}/hybrid_cnnstem_cross_attn_chirp/prototype_mean_prob"

  run_step "summarize results" \
    "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"

  run_step "compare classifier predictions" \
    "${PY}" scripts/compare_predictions.py \
      --a-pred "${OUT_ROOT}/osu_cnn_iq/classifier/predictions.csv" \
      --b-pred "${OUT_ROOT}/hybrid_cnnstem_cross_attn_chirp/classifier/predictions.csv" \
      --a-name osu_cnn_iq_classifier \
      --b-name hybrid_classifier \
      --out-dir "${OUT_ROOT}/analysis_classifier"

  echo
  echo "single_gpu_compare finished at $(date --iso-8601=seconds)"
  echo "summary=${OUT_ROOT}/summary.csv"
} 2>&1 | tee "${LOG}"
