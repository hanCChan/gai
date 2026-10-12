#!/usr/bin/env bash
# Cross-receiver source-only improvement sweep (QUICK, 30 epochs).
# Base = best normalization (iq_rms + fft log_zscore + oob ratio).
# Tests receiver-style augmentation and CFO auxiliary feature.
# No backbone change, NO center loss / SupCon / multi-scale / hard margin / TTA / adversarial.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
RUN_ROOT=${RUN_ROOT:-runs/cross_receiver_aug_cfo_sweep}
OUT_ROOT=${OUT_ROOT:-outputs/cross_receiver_aug_cfo_sweep}

EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
PIN_MEMORY=${PIN_MEMORY:-1}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs "${OUT_ROOT}"
LOG="logs/cross_receiver_aug_cfo_sweep_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

LOADER_ARGS=( --num-workers "${NUM_WORKERS}" )
if [[ "${PIN_MEMORY}" == "1" ]] && "${PY}" scripts/finetune.py --help 2>/dev/null | grep -q -- '--pin-memory'; then
  LOADER_ARGS+=( --pin-memory )
  PIN_STATUS="enabled"
elif [[ "${PIN_MEMORY}" == "1" ]]; then
  PIN_STATUS="requested-but-unsupported(skipped)"
else
  PIN_STATUS="disabled"
fi

COMMON_TRAIN=(
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --samples-per-file 256
  --eval-samples-per-file 256
  --dim 64
  --depth 2
  --device cuda
  --label-smoothing 0.05
  --weight-decay 5e-4
  --input-norm iq_rms
  --fft-norm log_zscore
  --oob-norm ratio
  "${LOADER_ARGS[@]}"
)
EVAL_COMMON=(
  --batch-size "${BATCH_SIZE}"
  --samples-per-file 256
  --eval-samples-per-file 256
  --device cuda
  "${LOADER_ARGS[@]}"
)

HYBRID_ARCH=(
  --patch-embed-type cnn_stem
  --cnn-stem-dim 32
  --oob-fusion-type cross_attn_oob
  --use-oob-cross-attention
  --use-chirp-embedding
)
RX_AUG=( --augment-receiver-style )
CFO_FEAT=( --use-cfo-feature --cfo-feature-type both --cfo-feature-norm train_zscore )

DIRECTIONS=(
  "rx1_to_rx2:data/manifest_rx1_to_rx2.csv"
  "rx2_to_rx1:data/manifest_rx2_to_rx1.csv"
)

run_step() {
  local name="$1"; shift
  echo; echo "==> $(date --iso-8601=seconds) ${name}"
  "$@"
}

eval_one() {
  local manifest="$1" ckpt="$2" out_base="$3" mode="$4" vote="$5"
  run_step "eval ${out_base##*/} ${mode} ${vote}" \
    "${PY}" scripts/evaluate.py --manifest "${manifest}" \
      --checkpoint "${ckpt}" "${EVAL_COMMON[@]}" \
      --mode "${mode}" --file-vote-mode "${vote}" \
      --out-dir "${out_base}/${mode}_${vote}"
}

train_cnn() {
  local direction="$1" manifest="$2"
  local ckpt_dir="${RUN_ROOT}/${direction}/B_cnn_iq_rms"
  local out_base="${OUT_ROOT}/${direction}/B_cnn_iq_rms"
  run_step "train ${direction}/B_cnn_iq_rms" \
    "${PY}" scripts/finetune.py --manifest "${manifest}" \
      --model-type osu_cnn --cnn-input-type iq \
      "${COMMON_TRAIN[@]}" \
      --out-dir "${ckpt_dir}"
  eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" classifier mean_logits
  eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" prototype mean_prob
}

train_hybrid() {
  local direction="$1" manifest="$2" cfg_name="$3"; shift 3
  local extra=("$@")
  local ckpt_dir="${RUN_ROOT}/${direction}/${cfg_name}"
  local out_base="${OUT_ROOT}/${direction}/${cfg_name}"
  run_step "train ${direction}/${cfg_name}" \
    "${PY}" scripts/finetune.py --manifest "${manifest}" \
      "${COMMON_TRAIN[@]}" "${HYBRID_ARCH[@]}" ${extra[@]+"${extra[@]}"} \
      --out-dir "${ckpt_dir}"
  eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" classifier mean_logits
  eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" classifier confidence_weighted
  eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" prototype mean_prob
}

{
  echo "cross_receiver_aug_cfo_sweep started at $(date --iso-8601=seconds)"
  echo "gpu_id=${GPU_ID} epochs=${EPOCHS} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} pin_memory=${PIN_STATUS} center_loss=none"
  for dir_entry in "${DIRECTIONS[@]}"; do
    direction="${dir_entry%%:*}"; manifest="${dir_entry#*:}"
    train_cnn "${direction}" "${manifest}"
    train_hybrid "${direction}" "${manifest}" D0_oob_ratio_only
    train_hybrid "${direction}" "${manifest}" D1_oob_ratio_rxaug "${RX_AUG[@]}"
    train_hybrid "${direction}" "${manifest}" D2_oob_ratio_cfo "${CFO_FEAT[@]}"
    train_hybrid "${direction}" "${manifest}" D3_oob_ratio_rxaug_cfo "${RX_AUG[@]}" "${CFO_FEAT[@]}"
  done

  run_step "summarize aug_cfo sweep" "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"
  echo
  echo "cross_receiver_aug_cfo_sweep finished at $(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"
