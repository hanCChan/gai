#!/usr/bin/env bash
# Resume OSU LoRa Diff_Configurations downloads with aria2c through a proxy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-data/raw/osu_lora}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:7899}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
SPLIT="${SPLIT:-8}"
LIST_TSV="${DATA_ROOT}/download_configs.tsv"
ARIA2_LIST="${DATA_ROOT}/download_configs.aria2.txt"
LOG_DIR="${ROOT}/logs"

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "ERROR: aria2c not found. Install it first, e.g.: sudo apt update && sudo apt install -y aria2" >&2
  exit 127
fi

proxy_host_port="${PROXY_URL#*://}"
proxy_host="${proxy_host_port%%:*}"
proxy_port="${proxy_host_port##*:}"
if ! timeout 2 bash -c ":</dev/tcp/${proxy_host}/${proxy_port}" 2>/dev/null; then
  echo "ERROR: proxy is not reachable from this server: ${PROXY_URL}" >&2
  echo "If SakuraCat runs on your local laptop, open a local terminal and run:" >&2
  echo "  ssh -N -R 7899:127.0.0.1:7899 hanchengcheng@10.10.41.10" >&2
  exit 2
fi

if [[ ! -f "${LIST_TSV}" ]]; then
  echo "ERROR: missing ${LIST_TSV}. Run scripts/download_osu_lora_configs.sh once with DRY_RUN=1 to generate it." >&2
  exit 3
fi

awk -F '\t' '
{
  target=$2
  dir=target
  sub("/[^/]*$", "", dir)
  out=target
  sub("^.*/", "", out)
  print $1
  print "  dir=" dir
  print "  out=" out
}
' "${LIST_TSV}" > "${ARIA2_LIST}"

log="${LOG_DIR}/download_osu_lora_configs_aria2_proxy_$(date +%Y%m%d_%H%M%S).out"
echo "Using proxy: ${PROXY_URL}"
echo "Using input: ${ARIA2_LIST}"
echo "Writing log: ${log}"

aria2c \
  --input-file="${ARIA2_LIST}" \
  --continue=true \
  --max-concurrent-downloads="${MAX_CONCURRENT}" \
  --split="${SPLIT}" \
  --min-split-size=8M \
  --max-connection-per-server="${SPLIT}" \
  --retry-wait=5 \
  --max-tries=0 \
  --timeout=60 \
  --connect-timeout=30 \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --all-proxy="${PROXY_URL}" \
  --summary-interval=30 \
  --console-log-level=notice \
  2>&1 | tee "${log}"
