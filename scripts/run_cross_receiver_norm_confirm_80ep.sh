#!/usr/bin/env bash
# Cross-receiver normalization CONFIRMATION run (80 epochs).
# Confirms the 30ep finding that oob_ratio is the winning normalization.
# Includes legacy_zscore (== old implementation: iq_rms+log_zscore+oob zscore) as control.
# No backbone change, NO center loss / SupCon / multi-scale / hard margin / TTA / adversarial.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
RUN_ROOT=${RUN_ROOT:-runs/cross_receiver_norm_confirm_80ep}
OUT_ROOT=${OUT_ROOT:-outputs/cross_receiver_norm_confirm_80ep}

EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
PIN_MEMORY=${PIN_MEMORY:-1}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs "${OUT_ROOT}"
LOG="logs/cross_receiver_norm_confirm_80ep_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

# DataLoader args: always pass --num-workers; pass --pin-memory only if the CLI supports it.
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

# config | model_type | input_norm | fft_norm | oob_norm
CONFIGS=(
  "A_cnn_raw_none:osu_cnn:none:none:none"
  "B_cnn_iq_rms:osu_cnn:iq_rms:none:none"
  "C_hybrid_legacy_zscore:hybrid:iq_rms:log_zscore:zscore"
  "D_hybrid_oob_ratio:hybrid:iq_rms:log_zscore:ratio"
  "E_hybrid_raw_none:hybrid:none:none:none"
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
  run_step "eval $(basename "$(dirname "${ckpt}")") ${mode} ${vote}" \
    "${PY}" scripts/evaluate.py --manifest "${manifest}" \
      --checkpoint "${ckpt}" "${EVAL_COMMON[@]}" \
      --mode "${mode}" --file-vote-mode "${vote}" \
      --out-dir "${out_base}/${mode}_${vote}"
}

train_and_eval() {
  local direction="$1" manifest="$2" cfg_name="$3" mtype="$4" inn="$5" fftn="$6" oobn="$7"
  local norm_args=(--input-norm "${inn}" --fft-norm "${fftn}" --oob-norm "${oobn}")
  local ckpt_dir="${RUN_ROOT}/${direction}/${cfg_name}"
  local out_base="${OUT_ROOT}/${direction}/${cfg_name}"

  if [[ "${mtype}" == "osu_cnn" ]]; then
    run_step "train ${direction}/${cfg_name}" \
      "${PY}" scripts/finetune.py --manifest "${manifest}" \
        --model-type osu_cnn --cnn-input-type iq \
        "${COMMON_TRAIN[@]}" "${norm_args[@]}" \
        --out-dir "${ckpt_dir}"
    eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" classifier mean_logits
    eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" prototype mean_prob
  else
    run_step "train ${direction}/${cfg_name}" \
      "${PY}" scripts/finetune.py --manifest "${manifest}" \
        "${COMMON_TRAIN[@]}" "${HYBRID_ARCH[@]}" "${norm_args[@]}" \
        --out-dir "${ckpt_dir}"
    eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" classifier mean_logits
    eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" classifier confidence_weighted
    eval_one "${manifest}" "${ckpt_dir}/best.pt" "${out_base}" prototype mean_prob
  fi
}

paired_compare() {
  # Headline test: Hybrid oob_ratio (D) vs CNN iq_rms (B), file-level McNemar.
  local direction="$1"
  run_step "paired ${direction} (CNN iq_rms vs Hybrid oob_ratio)" \
    "${PY}" scripts/paired_compare_models.py \
      --a-pred "${OUT_ROOT}/${direction}/B_cnn_iq_rms/classifier_mean_logits/predictions.csv" \
      --b-pred "${OUT_ROOT}/${direction}/D_hybrid_oob_ratio/classifier_mean_logits/predictions.csv" \
      --a-name CNN-iq_rms \
      --b-name Hybrid-oob_ratio \
      --out "${OUT_ROOT}/paired_${direction}.csv" \
      --diff-out "${OUT_ROOT}/paired_${direction}_diff_files.csv"
}

{
  echo "cross_receiver_norm_confirm_80ep started at $(date --iso-8601=seconds)"
  echo "gpu_id=${GPU_ID} epochs=${EPOCHS} batch_size=${BATCH_SIZE} num_workers=${NUM_WORKERS} pin_memory=${PIN_STATUS} center_loss=none"
  for dir_entry in "${DIRECTIONS[@]}"; do
    direction="${dir_entry%%:*}"; manifest="${dir_entry#*:}"
    for cfg in "${CONFIGS[@]}"; do
      IFS=':' read -r c_name c_model c_in c_fft c_oob <<< "${cfg}"
      train_and_eval "${direction}" "${manifest}" "${c_name}" "${c_model}" "${c_in}" "${c_fft}" "${c_oob}"
    done
  done

  run_step "summarize confirm" "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"
  paired_compare rx1_to_rx2
  paired_compare rx2_to_rx1
  echo
  echo "cross_receiver_norm_confirm_80ep finished at $(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"
