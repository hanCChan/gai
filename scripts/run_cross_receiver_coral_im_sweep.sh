#!/usr/bin/env bash
# CORAL + IM target-unlabeled alignment sweep (QUICK, 30 epochs).
# Base model: Hybrid oob_ratio + CFO (D2). No pseudo-label / center loss / SupCon.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
RUN_ROOT=${RUN_ROOT:-runs/cross_receiver_coral_im_sweep}
OUT_ROOT=${OUT_ROOT:-outputs/cross_receiver_coral_im_sweep}

EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
PIN_MEMORY=${PIN_MEMORY:-1}
CORAL_W=${CORAL_W:-1.0}
IM_W=${IM_W:-0.1}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs "${OUT_ROOT}"
LOG="logs/cross_receiver_coral_im_sweep_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

LOADER_ARGS=( --num-workers "${NUM_WORKERS}" )
if [[ "${PIN_MEMORY}" == "1" ]] && "${PY}" scripts/finetune.py --help 2>/dev/null | grep -q -- '--pin-memory'; then
  LOADER_ARGS+=( --pin-memory )
  PIN_STATUS="enabled"
elif [[ "${PIN_MEMORY}" == "1" ]]; then
  PIN_STATUS="requested-but-unsupported(skipped)"
else
  PIN_STATUS="disabled"
fi

HYBRID_BASE=(
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
  --patch-embed-type cnn_stem
  --cnn-stem-dim 32
  --oob-fusion-type cross_attn_oob
  --use-oob-cross-attention
  --use-chirp-embedding
  --use-cfo-feature
  --cfo-feature-type both
  --cfo-feature-norm train_zscore
  "${LOADER_ARGS[@]}"
)
EVAL_COMMON=(
  --batch-size "${BATCH_SIZE}"
  --samples-per-file 256
  --eval-samples-per-file 256
  --device cuda
  "${LOADER_ARGS[@]}"
)
CNN_BASE=(
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
  --model-type osu_cnn
  --cnn-input-type iq
  "${LOADER_ARGS[@]}"
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

eval_one() {
  local manifest="$1" ckpt="$2" out_base="$3" mode="$4" vote="$5"
  run_step "eval $(basename "${out_base}") ${mode} ${vote}" \
    "${PY}" scripts/evaluate.py --manifest "${manifest}" \
      --checkpoint "${ckpt}" "${EVAL_COMMON[@]}" \
      --mode "${mode}" --file-vote-mode "${vote}" \
      --out-dir "${out_base}/${mode}_${vote}"
}

train_hybrid() {
  local direction="$1" manifest="$2" cfg="$3"; shift 3
  local extra=("$@")
  local ckpt_dir="${RUN_ROOT}/${direction}/${cfg}"
  local out_base="${OUT_ROOT}/${direction}/${cfg}"
  run_step "train ${direction}/${cfg}" \
    "${PY}" scripts/finetune.py --manifest "${manifest}" \
      "${HYBRID_BASE[@]}" ${extra[@]+"${extra[@]}"} \
      --out-dir "${ckpt_dir}"
  eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" classifier mean_logits
  eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" classifier confidence_weighted
  eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" prototype mean_prob
}

train_cnn_ref() {
  local direction="$1" manifest="$2"
  local ckpt_dir="${RUN_ROOT}/${direction}/B_cnn_iq_rms"
  local out_base="${OUT_ROOT}/${direction}/B_cnn_iq_rms"
  run_step "train ${direction}/B_cnn_iq_rms" \
    "${PY}" scripts/finetune.py --manifest "${manifest}" \
      "${CNN_BASE[@]}" --out-dir "${ckpt_dir}"
  eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" classifier mean_logits
  eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" prototype mean_prob
}

{
  echo "cross_receiver_coral_im_sweep started at $(date --iso-8601=seconds)"
  echo "gpu_id=${GPU_ID} epochs=${EPOCHS} batch=${BATCH_SIZE} coral_w=${CORAL_W} im_w=${IM_W} pin=${PIN_STATUS}"
  for dir_entry in "${DIRECTIONS[@]}"; do
    direction="${dir_entry%%:*}"; manifest="${dir_entry#*:}"
    train_cnn_ref "${direction}" "${manifest}"
    # D2 source-only baseline (no target alignment)
    train_hybrid "${direction}" "${manifest}" D2_source_only \
      --domain-align-loss none
    # CORAL only: target unlabeled = val split of same manifest (other receiver)
    train_hybrid "${direction}" "${manifest}" D2_coral \
      --use-target-unlabeled \
      --target-manifest "${manifest}" \
      --domain-align-loss coral \
      --domain-align-weight "${CORAL_W}" \
      --target-loader-ratio 1
    # CORAL + IM
    train_hybrid "${direction}" "${manifest}" D2_coral_im \
      --use-target-unlabeled \
      --target-manifest "${manifest}" \
      --domain-align-loss coral_im \
      --domain-align-weight "${CORAL_W}" \
      --im-weight "${IM_W}" \
      --target-loader-ratio 1
  done
  run_step "summarize coral_im sweep" "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"
  echo
  echo "cross_receiver_coral_im_sweep finished at $(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"
