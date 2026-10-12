#!/usr/bin/env bash
# Resume Phase 4 location + distance jobs only (requires non-empty manifests).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
PHASE4_BASE=${PHASE4_BASE:-outputs/paper_runs/phase4_deployment_20260624_124617_b030d34}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
source scripts/paper/lib/job_helpers.sh
source scripts/paper/lib/paper_env.sh

BASE="${PHASE4_BASE}"
RUNS="${BASE}/runs"
OUTS="${BASE}/outputs"
LOGS="${BASE}/logs/train_jobs_locdist"
JOBS_FILE="${BASE}/jobs_locdist.tsv"
mkdir -p "${RUNS}" "${OUTS}" "${LOGS}"
: > "${JOBS_FILE}"

EPOCHS=${EPOCHS:-80}
SEED=${SEED:-0}
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

add_fold_job() {
  local task=$1 manifest=$2 fold=$3 model=$4
  local id="${task}_fold_${fold}_${model}"
  local marker="${OUTS}/${id}/file_predictions.csv"
  local extra
  if [[ "${model}" == "cnn" ]]; then
    extra=(--model-type osu_cnn --cnn-input-type iq --input-norm iq_rms)
  else
    extra=("${HYBRID[@]}")
  fi
  local cmd="$(mgpu_cmd_env) && \
'${PY}' scripts/finetune.py --manifest '${manifest}' --fold '${fold}' --out-dir '${RUNS}/${id}' ${COMMON[*]} ${extra[*]} && \
'${PY}' scripts/evaluate.py --manifest '${manifest}' --fold '${fold}' --checkpoint '${RUNS}/${id}/best.pt' \
  --mode classifier --file-vote-mode mean_logits --out-dir '${OUTS}/${id}' ${COMMON[*]}"
  mgpu_write_job "${JOBS_FILE}" "${id}" "${cmd}" "${marker}"
}

LOC_MANIFEST="data/manifest_locations_leave_one_location.csv"
for loc in 1 2 3; do
  add_fold_job "location_loco" "${LOC_MANIFEST}" "${loc}" "cnn"
  add_fold_job "location_loco" "${LOC_MANIFEST}" "${loc}" "hybrid"
done

DIST_MANIFEST="data/manifest_distances_leave_one_distance.csv"
for d in 5m 10m 15m 20m; do
  add_fold_job "distance_loco" "${DIST_MANIFEST}" "${d}" "cnn"
  add_fold_job "distance_loco" "${DIST_MANIFEST}" "${d}" "hybrid"
done

echo "==> Launching $(grep -c . "${JOBS_FILE}") loc/dist jobs on GPUs ${GPUS}"
mgpu_run_jobs "${JOBS_FILE}" "${LOGS}"

"${PY}" scripts/summarize_results.py --input-dir "${OUTS}" --out "${BASE}/deployment_shift_summary.csv" 2>/dev/null || true
echo "Phase 4 loc/dist resume complete: ${BASE}"
