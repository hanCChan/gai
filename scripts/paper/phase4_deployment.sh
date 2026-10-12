#!/usr/bin/env bash
# Phase 4: P1 deployment-shift (Config / Location / Distance)
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
EPOCHS=${EPOCHS:-80}
SEED=${SEED:-0}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
source scripts/paper/lib/job_helpers.sh
source scripts/paper/lib/paper_env.sh

TAG=${TAG:-$(date +%Y%m%d_%H%M%S)}
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
BASE=${BASE:-outputs/paper_runs/phase4_deployment_${TAG}_${COMMIT}}
RUNS="${BASE}/runs"
OUTS="${BASE}/outputs"
LOGS="${BASE}/logs"
JOBS_FILE="${BASE}/jobs.tsv"
mkdir -p "${RUNS}" "${OUTS}" "${LOGS}"
: > "${JOBS_FILE}"

manifest_data_rows() {
  local f=$1
  [[ -f "${ROOT}/${f}" ]] || { echo 0; return; }
  local n
  n=$(wc -l < "${ROOT}/${f}")
  echo $((n - 1))
}

echo "==> Generate P1 manifests"
"${PY}" scripts/generate_manifest_configs.py --root "${ROOT}"
if ! "${PY}" scripts/generate_manifest_locations_distances.py --root "${ROOT}" --skip-check; then
  echo "WARN: location/distance manifest generation failed (data may be missing)."
  echo "      Run: DRY_RUN=0 bash scripts/download_osu_lora_locations_distances.sh"
fi

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

CFG_MANIFEST="data/manifest_configs_leave_one_config.csv"
if [[ -f "${ROOT}/${CFG_MANIFEST}" ]]; then
  for cfg in 1 2 3 4; do
    add_fold_job "config_loco" "${CFG_MANIFEST}" "${cfg}" "cnn"
    add_fold_job "config_loco" "${CFG_MANIFEST}" "${cfg}" "hybrid"
  done
fi

LOC_MANIFEST="data/manifest_locations_leave_one_location.csv"
LOC_ROWS=$(manifest_data_rows "${LOC_MANIFEST}")
if [[ "${LOC_ROWS}" -gt 0 ]]; then
  echo "==> Location manifest: ${LOC_ROWS} rows"
  for loc in 1 2 3; do
    add_fold_job "location_loco" "${LOC_MANIFEST}" "${loc}" "cnn"
    add_fold_job "location_loco" "${LOC_MANIFEST}" "${loc}" "hybrid"
  done
else
  echo "SKIP location jobs: ${LOC_MANIFEST} empty (download Diff_Locations_Setup first)"
fi

DIST_MANIFEST="data/manifest_distances_leave_one_distance.csv"
DIST_ROWS=$(manifest_data_rows "${DIST_MANIFEST}")
if [[ "${DIST_ROWS}" -gt 0 ]]; then
  echo "==> Distance manifest: ${DIST_ROWS} rows"
  for d in 5m 10m 15m 20m; do
    add_fold_job "distance_loco" "${DIST_MANIFEST}" "${d}" "cnn"
    add_fold_job "distance_loco" "${DIST_MANIFEST}" "${d}" "hybrid"
  done
else
  echo "SKIP distance jobs: ${DIST_MANIFEST} empty (download Diff_Distances_Setup first)"
fi

if [[ -s "${JOBS_FILE}" ]]; then
  echo "==> Launching $(grep -c . "${JOBS_FILE}") deployment jobs on GPUs ${GPUS}"
  mgpu_run_jobs "${JOBS_FILE}" "${LOGS}/train_jobs"
else
  echo "WARN: no deployment manifests found; skipping training jobs"
fi

"${PY}" scripts/summarize_results.py --input-dir "${OUTS}" --out "${BASE}/deployment_shift_summary.csv" 2>/dev/null || true
echo "Phase 4 complete: ${BASE}"
