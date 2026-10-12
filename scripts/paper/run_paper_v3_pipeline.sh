#!/usr/bin/env bash
# paper_ready_v3 clean pipeline: Step0 audit -> Step1 cross-day ablation -> aggregate
# Usage:
#   bash scripts/paper/run_paper_v3_pipeline.sh              # Step1 only (default)
#   STEPS=step1,step2 bash scripts/paper/run_paper_v3_pipeline.sh
#   WINNER=F_cross_attn_chirp_plain bash scripts/paper/run_paper_v3_pipeline.sh
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
STEPS=${STEPS:-step1}
WINNER=${WINNER:-}
TMUX=${TMUX:-0}
SESSION=${SESSION:-llm4rf_v3}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
source scripts/paper/lib/paper_env.sh
source scripts/paper/lib/job_helpers.sh

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${ROOT}/logs/paper_v3_${TS}"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/pipeline.log"; }

run_step() {
  local step=$1
  log "========== Generate jobs: ${step} =========="
  local gen_args=(--step "${step}")
  if [[ -n "${WINNER}" && "${step}" != "step1" ]]; then
    gen_args+=(--winner "${WINNER}")
  fi
  "${PY}" scripts/paper/generate_paper_v3_jobs.py --step step1 --dry-run --out-dir outputs/paper_ready_v3/step0_audit 2>&1 | tee -a "${LOG_DIR}/gen_${step}.log"
  local base="${ROOT}/outputs/paper_runs/paper_v3/run_${COMMIT}"
  case "${step}" in
    step1) local jobs="${base}/step1/jobs.tsv"; local logs="${base}/step1/logs/train_jobs" ;;
    step2) local jobs="${base}/step2/jobs.tsv"; local logs="${base}/step2/logs/train_jobs" ;;
    step3) local jobs="${base}/step3/jobs.tsv"; local logs="${base}/step3/logs/train_jobs" ;;
    step4) local jobs="${base}/step4/jobs.tsv"; local logs="${base}/step4/logs/eval_jobs" ;;
    *) log "Unknown step ${step}"; return 1 ;;
  esac
  mkdir -p "$(dirname "${logs}")"
  local n
  n=$(grep -c . "${jobs}" || echo 0)
  log "Launch ${n} jobs from ${jobs} on GPUs ${GPUS}"
  if ! mgpu_run_jobs "${jobs}" "${logs}" 2>&1 | tee -a "${LOG_DIR}/run_${step}.log"; then
    log "FAIL: ${step} had failed jobs — see ${logs}/"
    return 1
  fi
  log "${step} complete"
  echo "${base}" > "${LOG_DIR}/last_${step}_base.txt"
}

pick_winner() {
  log "========== Pick Step1 winner (File-Macro-F1 mean) =========="
  local base=$1
  WINNER=$("${PY}" - <<PY
import csv, json
from collections import defaultdict
from pathlib import Path
base = Path("${base}") / "step1" / "outputs"
scores = defaultdict(list)
for mp in sorted(base.rglob("metrics.json")):
    m = json.loads(mp.read_text())
    model = mp.parent.parent.name
    if model.startswith(("A_", "D_", "F_", "H_")):
        scores[model].append(float(m.get("file_macro_f1", 0)))
if not scores:
    print("F_cross_attn_chirp_plain")
else:
    best = max(scores, key=lambda k: sum(scores[k])/len(scores[k]))
    print(best)
PY
)
  log "Winner: ${WINNER}"
  echo "${WINNER}" > "${LOG_DIR}/winner.txt"
}

aggregate() {
  local base=$1
  log "========== Aggregate paper_ready_v3 =========="
  "${PY}" scripts/paper/aggregate_paper_ready_v3.py \
    --root "${ROOT}" \
    --run-base "${base}" \
    --step "${2:-step1}" \
    --out-dir outputs/paper_ready_v3 2>&1 | tee -a "${LOG_DIR}/aggregate.log"
}

main() {
  log "paper_v3 pipeline start commit=${COMMIT} STEPS=${STEPS} GPUS=${GPUS}"

  log "========== STEP 0: Preflight manifest audit =========="
  if ! "${PY}" scripts/paper/preflight_manifest_audit.py --root "${ROOT}" 2>&1 | tee -a "${LOG_DIR}/preflight.log"; then
    log "ABORT: preflight failed"
    exit 1
  fi
  log "Preflight PASS"

  IFS=',' read -ra STEP_ARR <<< "${STEPS}"
  local run_base=""
  for step in "${STEP_ARR[@]}"; do
    step=$(echo "${step}" | tr -d ' ')
    if [[ "${step}" == "step2" || "${step}" == "step3" || "${step}" == "step4" ]]; then
      if [[ -z "${WINNER}" ]]; then
        run_base=$(cat "${LOG_DIR}/last_step1_base.txt" 2>/dev/null || echo "")
        if [[ -z "${run_base}" ]]; then
          run_base=$(ls -td "${ROOT}"/outputs/paper_runs/paper_v3/run_* 2>/dev/null | head -1 || echo "")
        fi
        pick_winner "${run_base}"
      fi
    fi
    run_step "${step}" || exit 1
    run_base="${ROOT}/outputs/paper_runs/paper_v3/run_${COMMIT}"
    echo "${run_base}" > "${LOG_DIR}/last_${step}_base.txt"
  done

  local last_step="${STEP_ARR[${#STEP_ARR[@]}-1]}"
  aggregate "${run_base}" "${last_step}"
  log "DONE — results: ${ROOT}/outputs/paper_ready_v3/ logs: ${LOG_DIR}/"
}

if [[ "${TMUX}" == "1" ]]; then
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session ${SESSION} already exists — attach with: tmux attach -t ${SESSION}"
    exit 1
  fi
  tmux new-session -d -s "${SESSION}" "bash $(realpath "$0") 2>&1 | tee ${LOG_DIR}/tmux_main.log"
  echo "Started tmux session: ${SESSION}"
  echo "  attach: tmux attach -t ${SESSION}"
  echo "  logs:   ${LOG_DIR}/"
else
  main "$@"
fi
