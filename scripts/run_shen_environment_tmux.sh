#!/usr/bin/env bash
# Run inside tmux; records both setup and verification status.
set -euo pipefail
SHEN_TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SHEN_TASK_ROOT"
mkdir -p logs
SHEN_TASK_LOG="logs/shen_env_setup_20261012.log"
SHEN_TASK_STATUS="logs/shen_env_setup_20261012.status"
printf 'RUNNING\n' > "$SHEN_TASK_STATUS"
trap 'SHEN_EXIT_CODE=$?; if [[ "$SHEN_EXIT_CODE" -ne 0 ]]; then printf "FAILED exit=%s\n" "$SHEN_EXIT_CODE" > "$SHEN_TASK_STATUS"; fi' EXIT
bash scripts/setup_shen_stage1.sh 2>&1 | tee -a "$SHEN_TASK_LOG"
SHEN_CHECK_DIR="inventory/shen_stage1_environment_20261012"
if [[ -d "$SHEN_CHECK_DIR" ]]; then
    SHEN_CHECK_DIR="${SHEN_CHECK_DIR}_$(date -u +%H%M%S)"
fi
/home/keyan/venvs/shen-stage1/bin/python scripts/check_shen_stage1_environment.py \
    --out-dir "$SHEN_CHECK_DIR" 2>&1 | tee -a "$SHEN_TASK_LOG"
printf 'COMPLETE check_dir=%s\n' "$SHEN_CHECK_DIR" > "$SHEN_TASK_STATUS"
