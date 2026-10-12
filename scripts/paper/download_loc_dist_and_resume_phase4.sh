#!/usr/bin/env bash
# Download OSU LoRa location/distance IQ data, regenerate manifests, resume Phase 4 loc/dist jobs.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
PHASE4_BASE=${PHASE4_BASE:-outputs/paper_runs/phase4_deployment_20260624_124617_b030d34}
MAX_PARALLEL=${MAX_PARALLEL:-12}

cd "${ROOT}"
source scripts/paper/lib/paper_env.sh

echo "==> [1/3] Download Diff_Locations_Setup + Diff_Distances_Setup (~175 IQ files, ~28GB)"
DRY_RUN=0 MAX_PARALLEL="${MAX_PARALLEL}" bash scripts/download_osu_lora_locations_distances.sh

echo "==> [2/3] Regenerate manifests"
"${PY}" scripts/generate_manifest_locations_distances.py --root "${ROOT}"

LOC_ROWS=$(($(wc -l < data/manifest_locations_leave_one_location.csv) - 1))
DIST_ROWS=$(($(wc -l < data/manifest_distances_leave_one_distance.csv) - 1))
echo "Location rows: ${LOC_ROWS}, Distance rows: ${DIST_ROWS}"
if [[ "${LOC_ROWS}" -le 0 || "${DIST_ROWS}" -le 0 ]]; then
  echo "ERROR: manifests still empty after download" >&2
  exit 1
fi

echo "==> [3/3] Resume Phase 4 location + distance jobs in ${PHASE4_BASE}"
export BASE="${PHASE4_BASE}"
export TAG="resume_locdist"
export GPUS BATCH LR

# Append only loc/dist jobs to existing run dir
bash scripts/paper/phase4_resume_loc_dist.sh

echo "Done. Check: find ${PHASE4_BASE}/outputs -name file_predictions.csv | wc -l"
