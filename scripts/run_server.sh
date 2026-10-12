#!/usr/bin/env bash
# Generate fresh jobs; historical result files and commands are never executed.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
GPUS=${GPUS:-0}
STEP=${STEP:-step1}
TAG=${TAG:-$(date +%Y%m%d_%H%M%S)}
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
"${PY}" scripts/verify_bundle.py --data-only
BASE="outputs/server_${STEP}_${TAG}"
"${PY}" scripts/paper/generate_paper_v3_jobs.py \
  --root "${ROOT}" --py "${PY}" --step "${STEP}" --dry-run \
  --out-dir "${BASE}/jobs" --runs-dir "${ROOT}/${BASE}/runs" \
  --outputs-dir "${ROOT}/${BASE}/eval"
if [[ ${DRY_RUN:-1} == 1 ]]; then
  echo "Jobs prepared in ${BASE}/jobs. Set DRY_RUN=0 to train."
  exit 0
fi
"${PY}" scripts/paper/multigpu_launcher.py \
  --jobs-file "${BASE}/jobs/jobs.tsv" --gpus "${GPUS}" --log-dir "${BASE}/logs"
