#!/usr/bin/env bash
# Tonight v2 pipeline: preflight -> smoke -> Phase4 -> Phase7 -> aggregate
# Usage: bash scripts/paper/run_tonight_v2_pipeline.sh
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
export GPUS=${GPUS:-1,2,3,4,5,6}
export BATCH=${BATCH:-256}
export LR=${LR:-3e-3}
export EPOCHS=${EPOCHS:-80}
LOG_DIR="${ROOT}/logs/tonight_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/pipeline.log"; }

log "========== STEP 0: Preflight manifest audit =========="
if ! "${PY}" scripts/paper/preflight_manifest_audit.py --root "${ROOT}" 2>&1 | tee -a "${LOG_DIR}/preflight.log"; then
  log "ABORT: preflight failed — fix manifests before training"
  exit 1
fi
log "Preflight PASS"

log "========== STEP 1: Gated smoke test (3 ep, lr=1e-3) =========="
bash scripts/paper/gated_smoke_test.sh 2>&1 | tee -a "${LOG_DIR}/smoke.log"
log "Smoke PASS"

log "========== STEP 2: Phase4 deployment (fixed fold+val) =========="
PHASE4_BASE=""
PHASE4_BASE=$(bash scripts/paper/phase4_deployment.sh 2>&1 | tee -a "${LOG_DIR}/phase4.log" | tail -1 | awk '{print $NF}')
log "Phase4 base: ${PHASE4_BASE}"

log "========== STEP 3: Phase4 aggregate + fold check =========="
"${PY}" scripts/paper/aggregate_paper_ready_v2.py \
  --root "${ROOT}" \
  --phase4-base "${PHASE4_BASE}" \
  --out-dir outputs/paper_ready_v2 2>&1 | tee -a "${LOG_DIR}/aggregate_p4.log"

log "========== STEP 4: Phase7 single-seed ablation =========="
PHASE7_BASE=""
PHASE7_BASE=$(SEED=0 bash scripts/paper/phase7_domain_robust_ablation.sh 2>&1 | tee -a "${LOG_DIR}/phase7.log" | tail -1 | awk '{print $NF}')
log "Phase7 base: ${PHASE7_BASE}"

log "========== STEP 5: Phase7 key models 3-seed =========="
SEEDS=0,1,2 MULTISEED_ONLY=1 BASE="${PHASE7_BASE}" \
  bash scripts/paper/phase7_domain_robust_ablation.sh 2>&1 | tee -a "${LOG_DIR}/phase7_multiseed.log"

log "========== STEP 6: Final aggregate =========="
"${PY}" scripts/paper/aggregate_paper_ready_v2.py \
  --root "${ROOT}" \
  --phase4-base "${PHASE4_BASE}" \
  --phase7-base "${PHASE7_BASE}" \
  --out-dir outputs/paper_ready_v2 2>&1 | tee -a "${LOG_DIR}/aggregate_final.log"

log "========== DONE =========="
log "Results: ${ROOT}/outputs/paper_ready_v2/"
log "Logs: ${LOG_DIR}/"
