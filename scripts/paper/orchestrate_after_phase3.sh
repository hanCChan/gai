#!/usr/bin/env bash
# Wait for Phase 3, resume Phase 2 (7 GPUs), then Phase 4-5 + aggregation.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
PHASE2_BASE=${PHASE2_BASE:-outputs/paper_runs/phase2_cross_day_20260624_101936_b030d34}
PHASE3_BASE=${PHASE3_BASE:-outputs/paper_runs/phase3_ablation_20260624_105433_b030d34}
PHASE2_GPUS=${PHASE2_GPUS:-1,2,3,4,5,6,7}
PHASE345_GPUS=${PHASE345_GPUS:-1,2,3,4,5,6,7}
POLL_SEC=${POLL_SEC:-120}
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/orchestrate_p3_p2resume_p45_${TS}.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
source scripts/paper/lib/paper_env.sh
export PHASE2_BASE PHASE3_BASE

log "Config: GPUS=${GPUS} BATCH=${BATCH} LR=${LR} BATCH_RX=${BATCH_RX}"

wait_phase3() {
  log "Waiting for Phase 3 to complete (${PHASE3_BASE})..."
  while true; do
    local launcher
    launcher=$(pgrep -f "multigpu_launcher.py.*phase3_ablation" | head -1 || true)
    local done_n
    done_n=$(find "${PHASE3_BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
    log "Phase 3: ${done_n}/7 done, launcher=${launcher:-none}"
    if [[ -z "${launcher}" && "${done_n}" -ge 7 ]]; then
      log "Phase 3 complete."
      return 0
    fi
    if [[ -z "${launcher}" && "${done_n}" -lt 7 ]]; then
      local fail_n
      fail_n=$(grep -l "FAIL(" "${PHASE3_BASE}/logs/train_jobs/"*.log 2>/dev/null | wc -l || true)
      log "ERROR: Phase 3 launcher gone but only ${done_n}/7 done (fail logs: ${fail_n})"
      return 1
    fi
    sleep "${POLL_SEC}"
  done
}

run_phase() {
  local n=$1 script=$2 gpus=$3
  shift 3
  log "========== Starting Phase ${n} (GPUS=${gpus} $*) =========="
  env GPUS="${gpus}" "$@" bash "${script}" >> "${LOG}" 2>&1
  log "========== Phase ${n} complete =========="
}

# Stop old watcher so it does not start Phase 4 before Phase 2 resume.
old_watch=$(pgrep -f "watch_phase2_then_3to5.sh" | head -1 || true)
if [[ -n "${old_watch}" ]]; then
  log "Stopping old watcher PID=${old_watch} (Phase 3 keeps running)"
  kill "${old_watch}" 2>/dev/null || true
  sleep 2
fi

wait_phase3

run_phase 2-full scripts/paper/phase2_rerun.sh "${PHASE2_GPUS}" PHASE2_BASE="${PHASE2_BASE}" BATCH="${BATCH}" LR="${LR}"

run_phase 4 scripts/paper/phase4_deployment.sh "${PHASE345_GPUS}" BATCH="${BATCH}" LR="${LR}"
run_phase 5 scripts/paper/phase5_cross_receiver.sh "${PHASE345_GPUS}" BATCH_RX="${BATCH_RX}" LR="${LR}"

log "========== Aggregating paper_ready =========="
"${PY}" scripts/paper/aggregate_paper_ready.py --root "${ROOT}" >> "${LOG}" 2>&1

p2_n=$(find "${PHASE2_BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
p3_n=$(find "${PHASE3_BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
log "Final counts: Phase2=${p2_n}/60 Phase3=${p3_n}/7"
log "All done. Log: ${LOG}"
