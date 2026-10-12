#!/usr/bin/env bash
# Wipe Phase 2 run dir and retrain all 60 jobs from scratch.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
PHASE2_BASE=${PHASE2_BASE:-outputs/paper_runs/phase2_cross_day_20260624_101936_b030d34}

cd "${ROOT}"
source scripts/paper/lib/paper_env.sh

export BASE="${PHASE2_BASE}"
export TAG="rerun"

echo "==> Phase 2 FULL RERUN: ${BASE}"
echo "    GPUs=${GPUS} batch=${BATCH} lr=${LR}"

rm -rf "${BASE}/runs" "${BASE}/outputs" "${BASE}/stats"
rm -rf "${BASE}/logs/train_jobs"
mkdir -p "${BASE}/logs/train_jobs"
rm -f "${BASE}/jobs.tsv" "${BASE}/summary.csv"

echo "{\"phase\":2,\"mode\":\"rerun\",\"batch\":${BATCH},\"lr\":${LR},\"gpus\":\"${GPUS}\"}" > "${BASE}/run_meta.json"

bash scripts/paper/phase2_cross_day.sh

final_n=$(find "${BASE}/outputs" -name file_predictions.csv 2>/dev/null | wc -l)
echo "==> Phase 2 rerun finished: ${final_n}/60"
if [[ "${final_n}" -lt 60 ]]; then
  fail_n=$(grep -l "FAIL(" "${BASE}/logs/train_jobs/"*.log 2>/dev/null | wc -l || true)
  echo "ERROR: incomplete (${final_n}/60), failed logs=${fail_n}" >&2
  exit 1
fi
