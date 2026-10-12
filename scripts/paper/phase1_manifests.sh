#!/usr/bin/env bash
# Phase 1: Generate paper manifests + audit table
set -euo pipefail
ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

echo "==> Phase 1: generate paper manifests"
"${PY}" scripts/paper/generate_paper_manifests.py --root "${ROOT}"

echo "==> Phase 1: audit manifests"
"${PY}" scripts/paper/audit_manifests.py --root "${ROOT}"

echo "Phase 1 complete. See outputs/paper_ready/manifest_audit.csv"
