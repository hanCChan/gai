#!/usr/bin/env bash
# Creates an isolated environment; does not start an audit or training job.
set -euo pipefail
SHEN_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHEN_ENV_ROOT="${SHEN_ENV_ROOT:-/home/keyan/venvs}"
SHEN_BOOTSTRAP="$SHEN_ENV_ROOT/shen-bootstrap"
SHEN_ENV="$SHEN_ENV_ROOT/shen-stage1"
SHEN_PYTHON_INSTALL="$SHEN_ENV_ROOT/python"
SHEN_PYTHON="$SHEN_PYTHON_INSTALL/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11"
if [[ ! -x "$SHEN_BOOTSTRAP/bin/uv" ]]; then
    python3 -m venv "$SHEN_BOOTSTRAP"
    "$SHEN_BOOTSTRAP/bin/python" -m pip install uv==0.13.0
fi
"$SHEN_BOOTSTRAP/bin/uv" python install --install-dir "$SHEN_PYTHON_INSTALL" --no-bin 3.11.17
# Preserve installed packages while making the interpreter accessible outside
# root's private home directory. Only this task's dedicated venv is affected.
"$SHEN_BOOTSTRAP/bin/uv" venv --allow-existing --python "$SHEN_PYTHON" "$SHEN_ENV"
"$SHEN_BOOTSTRAP/bin/uv" pip install --python "$SHEN_ENV/bin/python" \
    -r "$SHEN_PROJECT_ROOT/requirements-shen-stage1.txt"
"$SHEN_BOOTSTRAP/bin/uv" pip install --python "$SHEN_ENV/bin/python" \
    'torch @ https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-linux_x86_64.whl' \
    --index-url https://pypi.org/simple
printf 'Environment ready: %s\n' "$SHEN_ENV/bin/python"
