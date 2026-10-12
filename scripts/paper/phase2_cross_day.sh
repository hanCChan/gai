#!/usr/bin/env bash
# Phase 2: Cross-day main results (source-only protocol)
# Day1-4->Day5 + LODO, CNN-IQ vs Hybrid, multi-seed, stats
# Default: 7-GPU parallel (GPU0 reserved), batch=64 on A100-80G
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
EPOCHS=${EPOCHS:-80}
SEEDS=${SEEDS:-0,1,2,3,4}
EVAL_SPF=${EVAL_SPF:-256}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
source scripts/paper/lib/job_helpers.sh
source scripts/paper/lib/paper_env.sh

TAG=${TAG:-$(date +%Y%m%d_%H%M%S)}
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
BASE=${BASE:-outputs/paper_runs/phase2_cross_day_${TAG}_${COMMIT}}
RUNS="${BASE}/runs"
OUTS="${BASE}/outputs"
LOGS="${BASE}/logs"
JOBS_FILE="${BASE}/jobs.tsv"
mkdir -p "${RUNS}" "${OUTS}" "${LOGS}"
: > "${JOBS_FILE}"
if [[ "${TAG}" != "resume" ]]; then
  echo "{\"phase\":2,\"tag\":\"${TAG}\",\"commit\":\"${COMMIT}\",\"gpus\":\"${GPUS}\",\"batch\":${BATCH},\"lr\":${LR}}" > "${BASE}/run_meta.json"
fi

COMMON=(
  --epochs "${EPOCHS}" --batch-size "${BATCH}" --lr "${LR}"
  --samples-per-file "${EVAL_SPF}" --eval-samples-per-file "${EVAL_SPF}"
  --dim 64 --depth 2 --device cuda
  --train-split train --val-split val --eval-split test
)
HYBRID=(
  --patch-embed-type cnn_stem --cnn-stem-dim 32
  --oob-fusion-type cross_attn_oob --use-oob-cross-attention --use-chirp-embedding
  --input-norm iq_rms --fft-norm log_zscore --oob-norm zscore
  --label-smoothing 0.05 --weight-decay 5e-4
)

add_train_eval_job() {
  local name=$1 manifest=$2 seed=$3
  shift 3
  local rd="${RUNS}/${name}/seed_${seed}"
  local od="${OUTS}/${name}/seed_${seed}/classifier_mean_logits"
  local marker="${od}/file_predictions.csv"
  local extra=("$@")
  local cmd="$(mgpu_cmd_env) && \
'${PY}' scripts/finetune.py --manifest '${manifest}' --seed '${seed}' --out-dir '${rd}' \
  ${COMMON[*]} ${extra[*]} && \
'${PY}' scripts/evaluate.py --manifest '${manifest}' --seed '${seed}' \
  --checkpoint '${rd}/best.pt' --mode classifier --file-vote-mode mean_logits \
  --out-dir '${od}' ${COMMON[*]}"
  mgpu_write_job "${JOBS_FILE}" "${name}_seed${seed}" "${cmd}" "${marker}"
}

IFS=',' read -ra SEED_ARR <<< "${SEEDS}"

# --- Day1-4 -> Day5 source-only ---
MANIFEST="data/paper/cross_day_day1to5_source_only.csv"
for seed in "${SEED_ARR[@]}"; do
  add_train_eval_job "day1to5_cnn" "${MANIFEST}" "${seed}" \
    --model-type osu_cnn --cnn-input-type iq --input-norm iq_rms
  add_train_eval_job "day1to5_hybrid" "${MANIFEST}" "${seed}" \
    --model-type rf_hstu "${HYBRID[@]}"
done

# --- LODO source-only ---
for day in 1 2 3 4 5; do
  MANIFEST="data/paper/lodo_source_only/test_day_${day}.csv"
  for seed in "${SEED_ARR[@]}"; do
    add_train_eval_job "lodo_day${day}_cnn" "${MANIFEST}" "${seed}" \
      --model-type osu_cnn --cnn-input-type iq --input-norm iq_rms
    add_train_eval_job "lodo_day${day}_hybrid" "${MANIFEST}" "${seed}" \
      --model-type rf_hstu "${HYBRID[@]}"
  done
done

echo "==> Launching $(grep -c . "${JOBS_FILE}" || echo 0) training jobs on GPUs ${GPUS}"
mgpu_run_jobs "${JOBS_FILE}" "${LOGS}/train_jobs"

# --- eval_samples_per_file sensitivity (Hybrid, seed 0, Day1-4->Day5) ---
for spf in 128 256 512; do
  MANIFEST="data/paper/cross_day_day1to5_source_only.csv"
  rd="${RUNS}/day1to5_hybrid/seed_0"
  od="${OUTS}/spf_sensitivity/spf_${spf}/hybrid"
  [[ -f "${rd}/best.pt" ]] || continue
  "${PY}" scripts/evaluate.py --manifest "${MANIFEST}" --seed 0 \
    --checkpoint "${rd}/best.pt" \
    --eval-samples-per-file "${spf}" --samples-per-file "${spf}" \
    --eval-split test --train-split train --val-split val \
    --mode classifier --file-vote-mode mean_logits \
    --out-dir "${od}" --device cuda
done

# --- Bootstrap CI + paired tests (CPU) ---
STAT_OUT="${BASE}/stats"
mkdir -p "${STAT_OUT}"
for pair in day1to5; do
  for seed in "${SEED_ARR[@]}"; do
    cnn_pred="${OUTS}/${pair}_cnn/seed_${seed}/classifier_mean_logits/file_predictions.csv"
    hyb_pred="${OUTS}/${pair}_hybrid/seed_${seed}/classifier_mean_logits/file_predictions.csv"
    [[ -f "${cnn_pred}" ]] || continue
    [[ -f "${hyb_pred}" ]] || continue
    "${PY}" scripts/bootstrap_eval_ci.py --predictions "${hyb_pred}" \
      --out "${STAT_OUT}/hybrid_seed${seed}_bootstrap.csv" --n-bootstrap 1000 --format wide
    "${PY}" scripts/paired_compare_models.py \
      --a-pred "${cnn_pred}" --b-pred "${hyb_pred}" \
      --a-name cnn --b-name hybrid \
      --out "${STAT_OUT}/paired_seed${seed}.csv"
  done
done

"${PY}" scripts/summarize_results.py --input-dir "${OUTS}" --out "${BASE}/summary.csv" 2>/dev/null || true
echo "Phase 2 complete: ${BASE}"
