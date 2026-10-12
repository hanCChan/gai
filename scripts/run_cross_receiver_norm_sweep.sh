#!/usr/bin/env bash
# Cross-receiver input/spectrum normalization sweep (QUICK, 30 epochs).
# Tests whether receiver/spectrum normalization closes the cross-receiver gap.
# No backbone change, NO center loss / SupCon / multi-scale / hard margin / TTA.
# Hybrid = center_none main line. CNN-IQ baseline for reference.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
RUN_ROOT=${RUN_ROOT:-runs/cross_receiver_norm_sweep}
OUT_ROOT=${OUT_ROOT:-outputs/cross_receiver_norm_sweep}

# Throughput knobs (override via env). Model/normalization code is NOT touched.
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-8}
PIN_MEMORY=${PIN_MEMORY:-1}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs "${OUT_ROOT}"
LOG="logs/cross_receiver_norm_sweep_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

# DataLoader args: always pass --num-workers; pass --pin-memory only if the CLI
# actually supports it (avoids touching python code when it does not).
LOADER_ARGS=( --num-workers "${NUM_WORKERS}" )
if [[ "${PIN_MEMORY}" == "1" ]] && "${PY}" scripts/finetune.py --help 2>/dev/null | grep -q -- '--pin-memory'; then
  LOADER_ARGS+=( --pin-memory )
  PIN_STATUS="enabled"
elif [[ "${PIN_MEMORY}" == "1" ]]; then
  PIN_STATUS="requested-but-unsupported(skipped)"
else
  PIN_STATUS="disabled"
fi

# Fixed training hyper-parameters (quick sweep).
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
  "${LOADER_ARGS[@]}"
)
EVAL_COMMON=(
  --batch-size "${BATCH_SIZE}"
  --samples-per-file 256
  --eval-samples-per-file 256
  --device cuda
  "${LOADER_ARGS[@]}"
)

# Normalization settings: name | input_norm | fft_norm | oob_norm
SETTINGS=(
  "raw_none:none:none:none"
  "iq_rms:iq_rms:none:none"
  "iq_rms_fft_log:iq_rms:log_zscore:none"
  "iq_rms_fft_log_oob_ratio:iq_rms:log_zscore:ratio"
  "iq_rms_fft_log_oob_log_ratio:iq_rms:log_zscore:log_ratio"
)

# direction | manifest
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
  local direction="$1" manifest="$2" setting_name="$3" inn="$4" fftn="$5" oobn="$6"
  local norm_args=(--input-norm "${inn}" --fft-norm "${fftn}" --oob-norm "${oobn}")
  local base="${RUN_ROOT}/${direction}/${setting_name}"
  local out_base="${OUT_ROOT}/${direction}/${setting_name}"

  # ---- CNN baseline (only input_norm affects it) ----
  run_step "train CNN ${direction}/${setting_name}" \
    "${PY}" scripts/finetune.py --manifest "${manifest}" \
      --model-type osu_cnn --cnn-input-type iq \
      "${COMMON_TRAIN[@]}" "${norm_args[@]}" \
      --out-dir "${base}/cnn"
  run_step "eval CNN ${direction}/${setting_name} classifier mean_logits" \
    "${PY}" scripts/evaluate.py --manifest "${manifest}" \
      --checkpoint "${base}/cnn/best.pt" "${EVAL_COMMON[@]}" \
      --mode classifier --file-vote-mode mean_logits \
      --out-dir "${out_base}/cnn/classifier_mean_logits"

  # ---- Hybrid (center_none) ----
  run_step "train Hybrid ${direction}/${setting_name}" \
    "${PY}" scripts/finetune.py --manifest "${manifest}" \
      "${COMMON_TRAIN[@]}" "${norm_args[@]}" \
      --patch-embed-type cnn_stem --cnn-stem-dim 32 \
      --oob-fusion-type cross_attn_oob --use-oob-cross-attention --use-chirp-embedding \
      --out-dir "${base}/hybrid"
  for vote in mean_logits confidence_weighted; do
    run_step "eval Hybrid ${direction}/${setting_name} classifier ${vote}" \
      "${PY}" scripts/evaluate.py --manifest "${manifest}" \
        --checkpoint "${base}/hybrid/best.pt" "${EVAL_COMMON[@]}" \
        --mode classifier --file-vote-mode "${vote}" \
        --out-dir "${out_base}/hybrid/classifier_${vote}"
  done
  run_step "eval Hybrid ${direction}/${setting_name} prototype mean_prob" \
    "${PY}" scripts/evaluate.py --manifest "${manifest}" \
      --checkpoint "${base}/hybrid/best.pt" "${EVAL_COMMON[@]}" \
      --mode prototype --file-vote-mode mean_prob \
      --out-dir "${out_base}/hybrid/prototype_mean_prob"
}

{
  echo "cross_receiver_norm_sweep started at $(date --iso-8601=seconds)"
  echo "gpu_id=${GPU_ID} epochs=${EPOCHS} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} pin_memory=${PIN_STATUS} center_loss=none"
  for dir_entry in "${DIRECTIONS[@]}"; do
    direction="${dir_entry%%:*}"; manifest="${dir_entry#*:}"
    for setting in "${SETTINGS[@]}"; do
      IFS=':' read -r s_name s_in s_fft s_oob <<< "${setting}"
      train_and_eval "${direction}" "${manifest}" "${s_name}" "${s_in}" "${s_fft}" "${s_oob}"
    done
  done

  run_step "summarize norm sweep" "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"
  echo
  echo "cross_receiver_norm_sweep finished at $(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"
