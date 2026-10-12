#!/usr/bin/env bash
# Ensure full paper pipeline completes: download loc/dist data, Phase4 loc/dist, Phase5, aggregate.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
PHASE2_BASE=${PHASE2_BASE:-outputs/paper_runs/phase2_cross_day_20260624_101936_b030d34}
PHASE3_BASE=${PHASE3_BASE:-outputs/paper_runs/phase3_ablation_20260624_105433_b030d34}
PHASE4_BASE=${PHASE4_BASE:-outputs/paper_runs/phase4_deployment_20260624_124617_b030d34}
PROXY_URL=${PROXY_URL:-http://127.0.0.1:7899}
POLL_SEC=${POLL_SEC:-60}
LOG_DIR="${ROOT}/logs"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/finish_full_pipeline_${TS}.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
source scripts/paper/lib/paper_env.sh
export PHASE2_BASE PHASE3_BASE PHASE4_BASE PROXY_URL

log "=== Finish full pipeline ==="
log "GPUS=${GPUS} BATCH=${BATCH} BATCH_RX=${BATCH_RX}"

count_iq() {
  local loc n dist nd
  loc=$(find data/raw/osu_lora/Diff_Locations_Setup -name 'IQ_*.dat' -size +1M 2>/dev/null | wc -l)
  dist=$(find data/raw/osu_lora/Diff_Distances_Setup -name 'IQ_*.dat' -size +1M 2>/dev/null | wc -l)
  echo "${loc} ${dist}"
}

wait_download() {
  log "Waiting for loc/dist IQ download (target 75 + 100 = 175 files)..."
  while true; do
    read -r loc dist <<< "$(count_iq)"
    log "Downloaded: Location=${loc}/75 Distance=${dist}/100"
    if [[ "${loc}" -ge 72 && "${dist}" -ge 96 ]]; then
      log "Download complete (allowing a few missing devices)."
      return 0
    fi
    if ! pgrep -f "run_aria2_loc_dist_proxy|aria2c.*download_locations_distances" >/dev/null; then
      if [[ "${loc}" -lt 72 || "${dist}" -lt 96 ]]; then
        log "Download process gone but incomplete. Retrying aria2..."
        USE_PROXY=1 PROXY_URL="${PROXY_URL}" MAX_CONCURRENT=16 \
          bash scripts/run_aria2_loc_dist_proxy.sh >> "${LOG}" 2>&1 || \
        USE_PROXY=0 MAX_CONCURRENT=16 \
          bash scripts/run_aria2_loc_dist_proxy.sh >> "${LOG}" 2>&1 || true
      fi
    fi
    sleep "${POLL_SEC}"
  done
}

wait_phase4_config() {
  log "Waiting for Phase4 config jobs (current run)..."
  while pgrep -f "multigpu_launcher.py.*phase4_deployment_20260624" >/dev/null; do
    local n
    n=$(find "${PHASE4_BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
    log "Phase4 progress: ${n}/22 (config+locdist in old jobs.tsv)"
    sleep "${POLL_SEC}"
  done
  log "Phase4 launcher exited."
}

regen_manifests() {
  log "Regenerating location/distance manifests..."
  "${PY}" scripts/generate_manifest_locations_distances.py --root "${ROOT}" --skip-check
  local loc dist
  loc=$(($(wc -l < data/manifest_locations_leave_one_location.csv) - 1))
  dist=$(($(wc -l < data/manifest_distances_leave_one_distance.csv) - 1))
  log "Manifest rows: location=${loc} distance=${dist}"
  if [[ "${loc}" -le 0 || "${dist}" -le 0 ]]; then
    log "ERROR: manifests still empty"
    return 1
  fi
}

run_phase4_locdist() {
  local n target=14
  n=$(find "${PHASE4_BASE}/outputs" \( -path '*location_loco*' -o -path '*distance_loco*' \) -name file_predictions.csv 2>/dev/null | wc -l)
  if [[ "${n}" -ge "${target}" ]]; then
    log "Phase4 loc/dist already ${n}/${target}, skip."
    return 0
  fi
  log "Running Phase4 loc/dist resume (${n}/${target} done)..."
  PHASE4_BASE="${PHASE4_BASE}" GPUS="${GPUS}" BATCH="${BATCH}" LR="${LR}" \
    bash scripts/paper/phase4_resume_loc_dist.sh >> "${LOG}" 2>&1
}

run_phase5() {
  if ls outputs/paper_runs/phase5_cross_receiver_* >/dev/null 2>&1; then
    local base done total
    base=$(ls -td outputs/paper_runs/phase5_cross_receiver_* 2>/dev/null | head -1)
    done=$(find "${base}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
    total=$(($(wc -l < "${base}/jobs.tsv" 2>/dev/null || echo 1) - 0))
    if [[ "${done}" -ge 28 ]]; then
      log "Phase5 already ${done} done, skip."
      return 0
    fi
  fi
  log "Running Phase5..."
  GPUS="${GPUS}" BATCH_RX="${BATCH_RX}" LR="${LR}" \
    bash scripts/paper/phase5_cross_receiver.sh >> "${LOG}" 2>&1
}

# Start aria2 download if not complete
read -r loc dist <<< "$(count_iq)"
if [[ "${loc}" -lt 72 || "${dist}" -lt 96 ]]; then
  if ! pgrep -f "run_aria2_loc_dist_proxy|aria2c.*download_locations_distances" >/dev/null; then
    log "Starting aria2 download (proxy=${PROXY_URL} if reachable)..."
    nohup env PROXY_URL="${PROXY_URL}" USE_PROXY=1 MAX_CONCURRENT=16 SPLIT=8 \
      bash scripts/run_aria2_loc_dist_proxy.sh >> "${LOG}" 2>&1 &
    sleep 3
  fi
fi

# Wait for config jobs in parallel with download
wait_phase4_config &
WP4=$!
wait_download
wait "${WP4}" || true

regen_manifests
run_phase4_locdist
run_phase5

log "Aggregating paper_ready..."
"${PY}" scripts/paper/aggregate_paper_ready.py --root "${ROOT}" >> "${LOG}" 2>&1

p2=$(find "${PHASE2_BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
p3=$(find "${PHASE3_BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
p4=$(find "${PHASE4_BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
p5=$(find outputs/paper_runs/phase5_cross_receiver_*/outputs -name file_predictions.csv 2>/dev/null | wc -l)
log "FINAL: Phase2=${p2}/60 Phase3=${p3}/7 Phase4=${p4}/22 Phase5=${p5}/32"
log "All done. Log: ${LOG}"
