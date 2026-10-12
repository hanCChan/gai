#!/usr/bin/env bash
# Resume Phase 2 in an existing run directory (skip jobs with done markers).
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPUS=${GPUS:-1,2,3,4,5,6,7}
PHASE2_BASE=${PHASE2_BASE:-outputs/paper_runs/phase2_cross_day_20260624_101936_b030d34}

if [[ -z "${PHASE2_BASE}" ]]; then
  echo "ERROR: set PHASE2_BASE=outputs/paper_runs/phase2_cross_day_..." >&2
  exit 1
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export BASE="${PHASE2_BASE}"
export GPUS
export TAG="resume"

done_n=$(find "${BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
echo "==> Resume Phase 2: ${BASE} (${done_n}/60 done), GPUs=${GPUS}, batch=${BATCH}, lr=${LR}"

# Re-generate jobs.tsv and launch; launcher skips completed markers.
bash scripts/paper/phase2_cross_day.sh

final_n=$(find "${BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
echo "==> Phase 2 resume finished: ${final_n}/60"
if [[ "${final_n}" -lt 60 ]]; then
  fail_n=$(grep -l "FAIL(" "${BASE}/logs/train_jobs/"*.log 2>/dev/null | wc -l || true)
  echo "ERROR: Phase 2 still incomplete. Failed logs: ${fail_n}" >&2
  exit 1
fi
