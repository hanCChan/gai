#!/usr/bin/env bash
# Push llm4RF to https://github.com/hanCChan/lunwen
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GH="${GH:-$HOME/.local/bin/gh}"
REMOTE_NAME="${REMOTE_NAME:-lunwen}"
REPO="${REPO:-hanCChan/lunwen}"
BRANCH="${BRANCH:-main}"

cd "${ROOT}"

if ! command -v "${GH}" >/dev/null 2>&1; then
  echo "ERROR: gh not found. Install: curl + extract to ~/.local/bin/gh" >&2
  exit 127
fi

if ! "${GH}" auth status >/dev/null 2>&1; then
  echo "ERROR: GitHub 未登录。请在新终端运行: ${GH} auth login" >&2
  exit 1
fi

if ! git diff --cached --quiet || ! git diff --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
  echo "WARN: working tree not clean; commit first before pushing." >&2
  git status --short
  exit 2
fi

if ! git remote get-url "${REMOTE_NAME}" >/dev/null 2>&1; then
  git remote add "${REMOTE_NAME}" "https://github.com/${REPO}.git"
fi

if ! "${GH}" repo view "${REPO}" >/dev/null 2>&1; then
  echo "Creating GitHub repo ${REPO} ..."
  "${GH}" repo create "${REPO}" --public --description "LoRa RF fingerprinting: DI-RF-HSTU / llm4RF thesis code" --source=. --remote="${REMOTE_NAME}" --push
else
  echo "Repo ${REPO} exists; pushing ${BRANCH} ..."
  git push -u "${REMOTE_NAME}" "${BRANCH}"
fi

echo "Done: https://github.com/${REPO}"
