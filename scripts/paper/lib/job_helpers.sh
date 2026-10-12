#!/usr/bin/env bash
# Shared helpers for multi-GPU paper experiment phases.
set -euo pipefail

mgpu_cmd_env() {
  printf "cd '%s' && PYTHONPATH='%s/src:%s'" "${ROOT}" "${ROOT}" "${PYTHONPATH:-}"
}

mgpu_write_job() {
  # mgpu_write_job <jobs_file> <name> <cmd> [done_marker]
  local jf=$1 name=$2 cmd=$3 marker=${4:-}
  printf '%s\t%s' "${name}" "${cmd}" >> "${jf}"
  if [[ -n "${marker}" ]]; then
    printf '\t%s' "${marker}" >> "${jf}"
  fi
  printf '\n' >> "${jf}"
}

mgpu_run_jobs() {
  # mgpu_run_jobs <jobs_file> <log_dir>
  local jf=$1 log_dir=$2
  local gpus=${GPUS:-0}
  export ROOT="${ROOT:-/data1/hcc/llm4RF}"
  "${PY}" scripts/paper/multigpu_launcher.py \
    --jobs-file "${jf}" \
    --gpus "${gpus}" \
    --log-dir "${log_dir}"
}
