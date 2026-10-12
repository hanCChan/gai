#!/usr/bin/env bash
# Wait for Phase 2 to finish, then run Phase 3-5 + aggregation.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
PHASE2_PID=${PHASE2_PID:-}
POLL_SEC=${POLL_SEC:-300}
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="${LOG_DIR}/watch_phase2_to_5_${TS}.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

cd "${ROOT}"
export GPUS PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

if [[ -z "${PHASE2_PID}" ]]; then
  PHASE2_PID=$(pgrep -f "multigpu_launcher.py.*phase2_cross_day" | head -1 || true)
fi
if [[ -z "${PHASE2_PID}" ]]; then
  PHASE2_PID=$(pgrep -f "bash scripts/paper/phase2_cross_day.sh" | head -1 || true)
fi

if [[ -z "${PHASE2_PID}" ]]; then
  log "WARN: no running phase2_cross_day.sh found; will proceed if outputs look complete."
else
  log "Watching Phase 2 PID=${PHASE2_PID} (poll every ${POLL_SEC}s)"
  while kill -0 "${PHASE2_PID}" 2>/dev/null; do
    BASE=$(ls -td outputs/paper_runs/phase2_cross_day_* 2>/dev/null | head -1 || true)
    done_n=0
    if [[ -n "${BASE}" ]]; then
      done_n=$(find "${BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
    fi
    log "Phase 2 running... completed jobs: ${done_n}/60"
    sleep "${POLL_SEC}"
  done
  log "Phase 2 process exited."
fi

# Verify Phase 2 outputs
BASE=$(ls -td outputs/paper_runs/phase2_cross_day_* 2>/dev/null | head -1 || true)
if [[ -z "${BASE}" ]]; then
  log "ERROR: no phase2 output directory found."
  exit 1
fi
done_n=$(find "${BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
  log "Phase 2 outputs: ${done_n}/60 file_predictions.csv in ${BASE}"
if [[ "${done_n}" -lt 60 ]]; then
  log "WARN: Phase 2 incomplete (${done_n}/60). Will resume after Phase 3 if orchestrator is used."
  fail_n=$(grep -l "FAIL(" "${BASE}/logs/train_jobs/"*.log 2>/dev/null | wc -l || true)
  log "Failed job logs: ${fail_n}"
  if [[ "${fail_n}" -gt 0 ]]; then
    log "Aborting Phase 3-5 until Phase 2 failures are fixed."
    exit 1
  fi
  if [[ "${ALLOW_INCOMPLETE_PHASE2:-0}" != "1" ]]; then
    log "Aborting: Phase 2 incomplete. Set ALLOW_INCOMPLETE_PHASE2=1 to override."
    exit 1
  fi
fi

run_phase() {
  local n=$1 script=$2
  log "========== Starting Phase ${n} =========="
  bash "${script}" >> "${LOG}" 2>&1
  log "========== Phase ${n} complete =========="
}

run_phase 3 scripts/paper/phase3_ablation.sh
run_phase 4 scripts/paper/phase4_deployment.sh
run_phase 5 scripts/paper/phase5_cross_receiver.sh

log "========== Aggregating paper_ready =========="
"${PY}" scripts/paper/aggregate_paper_ready.py --root "${ROOT}" >> "${LOG}" 2>&1
log "All done. Log: ${LOG}"
