#!/usr/bin/env bash
# Phase 5: Cross-receiver stress test (source-only, upper bound, CORAL+IM)
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
EPOCHS=${EPOCHS:-80}
QUICK_EPOCHS=${QUICK_EPOCHS:-30}
SEED=${SEED:-0}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
source scripts/paper/lib/job_helpers.sh
source scripts/paper/lib/paper_env.sh
BATCH="${BATCH_RX}"

TAG=${TAG:-$(date +%Y%m%d_%H%M%S)}
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
BASE=${BASE:-outputs/paper_runs/phase5_cross_receiver_${TAG}_${COMMIT}}
RUNS="${BASE}/runs"
OUTS="${BASE}/outputs"
LOGS="${BASE}/logs"
JOBS_FILE="${BASE}/jobs.tsv"
mkdir -p "${RUNS}" "${OUTS}" "${LOGS}"
: > "${JOBS_FILE}"

COMMON=(
  --epochs "${EPOCHS}" --batch-size "${BATCH}" --lr "${LR}" --seed "${SEED}" --device cuda
  --samples-per-file 256 --eval-samples-per-file 256
  --dim 64 --depth 2
  --train-split train --val-split val --eval-split test
)
HYBRID=(
  --model-type rf_hstu --patch-embed-type cnn_stem --cnn-stem-dim 32
  --oob-fusion-type cross_attn_oob --use-oob-cross-attention --use-chirp-embedding
  --input-norm iq_rms --fft-norm log_zscore --oob-norm ratio
  --label-smoothing 0.05 --weight-decay 5e-4
)

add_rx_job() {
  local tag=$1 manifest=$2 model=$3 epochs=$4
  shift 4
  local id="${tag}_${model}"
  local marker="${OUTS}/${id}/file_predictions.csv"
  local extra
  if [[ "${model}" == "cnn" ]]; then
    extra=(--model-type osu_cnn --cnn-input-type iq --input-norm iq_rms)
  else
    extra=("${HYBRID[@]}")
  fi
  local cmd="$(mgpu_cmd_env) && \
'${PY}' scripts/finetune.py --manifest '${manifest}' --out-dir '${RUNS}/${id}' \
  ${COMMON[*]} --epochs '${epochs}' ${extra[*]} $* && \
'${PY}' scripts/evaluate.py --manifest '${manifest}' --checkpoint '${RUNS}/${id}/best.pt' \
  --mode classifier --file-vote-mode mean_logits --out-dir '${OUTS}/${id}' ${COMMON[*]}"
  mgpu_write_job "${JOBS_FILE}" "${id}" "${cmd}" "${marker}"
}

add_coral_job() {
  local direction=$1 lc=$2 lim=$3
  local id="coral_${direction}_lc${lc}_lim${lim}"
  local manifest="data/paper/${direction}_source_only.csv"
  local marker="${OUTS}/${id}/file_predictions.csv"
  local cmd="$(mgpu_cmd_env) && \
'${PY}' scripts/finetune.py --manifest '${manifest}' --out-dir '${RUNS}/${id}' \
  ${COMMON[*]} --epochs '${QUICK_EPOCHS}' ${HYBRID[*]} \
  --use-target-unlabeled --target-manifest '${manifest}' \
  --domain-align-loss coral_im --domain-align-weight '${lc}' --im-weight '${lim}' && \
'${PY}' scripts/evaluate.py --manifest '${manifest}' --checkpoint '${RUNS}/${id}/best.pt' \
  --mode classifier --file-vote-mode mean_logits --out-dir '${OUTS}/${id}' ${COMMON[*]}"
  mgpu_write_job "${JOBS_FILE}" "${id}" "${cmd}" "${marker}"
}

# Upper bound: same receiver
add_rx_job "rx1_ub" "data/paper/rx1_to_rx1_upper_bound.csv" "cnn" "${EPOCHS}"
add_rx_job "rx1_ub" "data/paper/rx1_to_rx1_upper_bound.csv" "hybrid" "${EPOCHS}"
add_rx_job "rx2_ub" "data/paper/rx2_to_rx2_upper_bound.csv" "cnn" "${EPOCHS}"
add_rx_job "rx2_ub" "data/paper/rx2_to_rx2_upper_bound.csv" "hybrid" "${EPOCHS}"

# Strict source-only
for direction in rx1_to_rx2 rx2_to_rx1; do
  add_rx_job "${direction}_so" "data/paper/${direction}_source_only.csv" "cnn" "${EPOCHS}"
  add_rx_job "${direction}_so" "data/paper/${direction}_source_only.csv" "hybrid" "${EPOCHS}"
done

# CORAL+IM sweep (30ep quick)
for lc in 0.001 0.01 0.05 0.1; do
  for lim in 0 0.001 0.01; do
    for direction in rx1_to_rx2 rx2_to_rx1; do
      add_coral_job "${direction}" "${lc}" "${lim}"
    done
  done
done

echo "==> Launching $(grep -c . "${JOBS_FILE}" || echo 0) cross-receiver jobs on GPUs ${GPUS}"
mgpu_run_jobs "${JOBS_FILE}" "${LOGS}/train_jobs"

# Error analysis (CPU, after all jobs)
for od in "${OUTS}"/*/; do
  if [[ -f "${od}predictions.csv" ]]; then
    "${PY}" scripts/analyze_cross_receiver_errors.py \
      --predictions "${od}predictions.csv" \
      --out-dir "${od}analysis" 2>/dev/null || true
  fi
done

"${PY}" scripts/summarize_results.py --input-dir "${OUTS}" --out "${BASE}/cross_receiver_summary.csv" 2>/dev/null || true
echo "Phase 5 complete: ${BASE}"
