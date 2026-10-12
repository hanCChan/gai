#!/usr/bin/env bash
# Phase 7: Domain-Robust OOB-Gated Hybrid validation matrix (Day1-4->Day5 source-only)
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
EPOCHS=${EPOCHS:-80}
SEED=${SEED:-0}
SEEDS=${SEEDS:-}  # optional comma list overrides SEED for multiseed runs
MULTISEED_ONLY=${MULTISEED_ONLY:-0}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
source scripts/paper/lib/job_helpers.sh
source scripts/paper/lib/paper_env.sh

TAG=${TAG:-$(date +%Y%m%d_%H%M%S)}
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
BASE=${BASE:-outputs/paper_runs/phase7_domain_robust_${TAG}_${COMMIT}}
RUNS="${BASE}/runs"
OUTS="${BASE}/outputs"
LOGS="${BASE}/logs"
JOBS_FILE="${BASE}/jobs.tsv"
mkdir -p "${RUNS}" "${OUTS}" "${LOGS}"
: > "${JOBS_FILE}"

MANIFEST="data/paper/cross_day_day1to5_source_only.csv"
COMMON=(
  --manifest "${MANIFEST}" --epochs "${EPOCHS}" --batch-size "${BATCH}" --lr "${LR}"
  --samples-per-file 256 --eval-samples-per-file 256
  --dim 64 --depth 2 --device cuda
  --train-split train --val-split val --eval-split test
  --input-norm iq_rms --fft-norm log_zscore
)
STEM=(--model-type rf_hstu --patch-embed-type cnn_stem --cnn-stem-dim 32)
PLAIN=(--label-smoothing 0.05 --weight-decay 5e-4)
ROBUST=(
  --checkpoint-metric macro_f1 --class-balanced-ce --loss-type focal
  "${PLAIN[@]}"
)

add_job() {
  local id=$1
  local ckpt_name=${2:-best.pt}
  shift 2
  local extra=("$@")
  local marker="${OUTS}/${id}/file_predictions.csv"
  local cmd="$(mgpu_cmd_env) && \
'${PY}' scripts/finetune.py --out-dir '${RUNS}/${id}' --seed '${SEED}' ${COMMON[*]} ${extra[*]} && \
'${PY}' scripts/evaluate.py --checkpoint '${RUNS}/${id}/${ckpt_name}' \
  --mode classifier --file-vote-mode mean_logits --out-dir '${OUTS}/${id}' --seed '${SEED}' ${COMMON[*]}"
  mgpu_write_job "${JOBS_FILE}" "${id}" "${cmd}" "${marker}"
}

add_multiseed_job() {
  local base_id=$1
  local ckpt_name=$2
  shift 2
  local extra=("$@")
  IFS=',' read -ra SEED_ARR <<< "${SEEDS}"
  for s in "${SEED_ARR[@]}"; do
    local id="${base_id}_seed_${s}"
    local marker="${OUTS}/${id}/file_predictions.csv"
    local cmd="$(mgpu_cmd_env) && \
'${PY}' scripts/finetune.py --out-dir '${RUNS}/${id}' --seed '${s}' ${COMMON[*]} ${extra[*]} && \
'${PY}' scripts/evaluate.py --checkpoint '${RUNS}/${id}/${ckpt_name}' \
  --mode classifier --file-vote-mode mean_logits --out-dir '${OUTS}/${id}' --seed '${s}' ${COMMON[*]}"
    mgpu_write_job "${JOBS_FILE}" "${id}" "${cmd}" "${marker}"
  done
}

# --- Single-seed matrix (M1-M7 + controls) ---
if [[ "${MULTISEED_ONLY}" != "1" ]]; then
add_job M1_cnn_iq --model-type osu_cnn --cnn-input-type iq --oob-norm none
add_job M2_linear_no_oob "${STEM[@]}" --patch-embed-type linear --no-oob --oob-fusion-type no_oob --oob-norm none "${PLAIN[@]}"
add_job M3_cnn_stem_no_oob "${STEM[@]}" --no-oob --oob-fusion-type no_oob --oob-norm none "${PLAIN[@]}"
add_job M4_concat_oob "${STEM[@]}" --oob-fusion-type concat_oob --oob-norm ratio "${PLAIN[@]}"
add_job M4b_concat_robust "${STEM[@]}" --oob-fusion-type concat_oob --oob-norm ratio "${ROBUST[@]}"
add_job M5a_gated_plain "${STEM[@]}" --oob-fusion-type gated_oob --oob-norm ratio --use-chirp-embedding "${PLAIN[@]}"
add_job M5_gated_robust "${STEM[@]}" --oob-fusion-type gated_oob --oob-norm ratio --use-chirp-embedding "${ROBUST[@]}"
add_job M6_gated_oob_dropout "${STEM[@]}" --oob-fusion-type gated_oob --oob-norm ratio --use-chirp-embedding --oob-dropout 0.3 "${ROBUST[@]}"

id=M7_gated_full_robust
marker="${OUTS}/${id}/file_predictions.csv"
cmd="$(mgpu_cmd_env) && \
'${PY}' scripts/finetune.py --out-dir '${RUNS}/${id}' --seed '${SEED}' ${COMMON[*]} ${STEM[*]} --oob-fusion-type gated_oob --oob-norm ratio --use-chirp-embedding --oob-dropout 0.3 --mixstyle --use-swa ${ROBUST[*]} && \
'${PY}' scripts/evaluate.py --checkpoint '${RUNS}/${id}/best.pt' --mode classifier --file-vote-mode mean_logits --out-dir '${OUTS}/${id}' --seed '${SEED}' ${COMMON[*]} && \
'${PY}' scripts/evaluate.py --checkpoint '${RUNS}/${id}/swa.pt' --mode classifier --file-vote-mode mean_logits --out-dir '${OUTS}/${id}_swa' --seed '${SEED}' ${COMMON[*]}"
mgpu_write_job "${JOBS_FILE}" "${id}" "${cmd}" "${marker}"
fi

# --- Optional 3-seed key models ---
if [[ -n "${SEEDS}" ]]; then
  add_multiseed_job M1_cnn_iq best.pt --model-type osu_cnn --cnn-input-type iq --oob-norm none
  add_multiseed_job M4_concat_oob best.pt "${STEM[@]}" --oob-fusion-type concat_oob --oob-norm ratio "${PLAIN[@]}"
  add_multiseed_job M5a_gated_plain best.pt "${STEM[@]}" --oob-fusion-type gated_oob --oob-norm ratio --use-chirp-embedding "${PLAIN[@]}"
  add_multiseed_job M6_gated_oob_dropout best.pt "${STEM[@]}" --oob-fusion-type gated_oob --oob-norm ratio --use-chirp-embedding --oob-dropout 0.3 "${ROBUST[@]}"
  add_multiseed_job M7_gated_full_robust best.pt "${STEM[@]}" --oob-fusion-type gated_oob --oob-norm ratio --use-chirp-embedding --oob-dropout 0.3 --mixstyle --use-swa "${ROBUST[@]}"
  # extra swa eval per seed
  IFS=',' read -ra SEED_ARR <<< "${SEEDS}"
  for s in "${SEED_ARR[@]}"; do
    id="M7_gated_full_robust_swa_seed_${s}"
    marker="${OUTS}/${id}/file_predictions.csv"
    cmd="$(mgpu_cmd_env) && \
'${PY}' scripts/evaluate.py --checkpoint '${RUNS}/M7_gated_full_robust_seed_${s}/swa.pt' \
  --mode classifier --file-vote-mode mean_logits --out-dir '${OUTS}/${id}' --seed '${s}' ${COMMON[*]}"
    mgpu_write_job "${JOBS_FILE}" "${id}" "${cmd}" "${marker}"
  done
fi

echo "==> Launching $(grep -c . "${JOBS_FILE}" || echo 0) domain-robust jobs on GPUs ${GPUS}"
mgpu_run_jobs "${JOBS_FILE}" "${LOGS}/train_jobs"

echo "Phase 7 complete: ${BASE}"
