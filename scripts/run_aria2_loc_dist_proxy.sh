#!/usr/bin/env bash
# Download OSU LoRa Diff_Locations + Diff_Distances with aria2c through proxy.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-data/raw/osu_lora}"
# Only aria2c receives --all-proxy; never export http_proxy/https_proxy (safe for other traffic).
PROXY_URL="${PROXY_URL:-http://127.0.0.1:7899}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
SPLIT="${SPLIT:-8}"
LIST_TSV="${DATA_ROOT}/download_locations_distances.tsv"
ARIA2_LIST="${DATA_ROOT}/download_locations_distances.aria2.txt"
LOG_DIR="${ROOT}/logs"

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${DATA_ROOT}"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "ERROR: aria2c not found" >&2
  exit 127
fi

# Generate TSV if missing
if [[ ! -f "${LIST_TSV}" ]]; then
  DRY_RUN=1 bash scripts/download_osu_lora_locations_distances.sh >/dev/null
fi

proxy_host_port="${PROXY_URL#*://}"
proxy_host="${proxy_host_port%%:*}"
proxy_port="${proxy_host_port##*:}"
PROXY_ARGS=()
if [[ -n "${PROXY_URL}" && "${USE_PROXY:-1}" == "1" ]]; then
  if timeout 2 bash -c ":</dev/tcp/${proxy_host}/${proxy_port}" 2>/dev/null; then
    PROXY_ARGS=(--all-proxy="${PROXY_URL}")
    echo "Using proxy: ${PROXY_URL}"
  else
    echo "WARN: proxy ${PROXY_URL} unreachable, downloading direct" >&2
    echo "  On local PC (SakuraCat port 7899, TUN off):" >&2
    echo "  ssh -N -R 7899:127.0.0.1:7899 hanchengcheng@<server>" >&2
  fi
else
  echo "Downloading direct (no proxy)"
fi

# aria2 input: only .dat IQ files (skip sigmf-meta)
awk -F '\t' '
$1 ~ /IQ_[0-9]+\.dat$/ {
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

dat_count=$(grep -c '\.dat$' "${ARIA2_LIST}" || echo 0)
log="${LOG_DIR}/download_loc_dist_aria2_proxy_$(date +%Y%m%d_%H%M%S).out"
echo "Proxy: ${PROXY_URL}"
echo "IQ files to fetch: ${dat_count}"
echo "Log: ${log}"

aria2c \
  --input-file="${ARIA2_LIST}" \
  --continue=true \
  --max-concurrent-downloads="${MAX_CONCURRENT}" \
  --split="${SPLIT}" \
  --min-split-size=8M \
  --max-connection-per-server="${SPLIT}" \
  --retry-wait=3 \
  --max-tries=0 \
  --timeout=60 \
  --connect-timeout=30 \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  "${PROXY_ARGS[@]}" \
  --summary-interval=30 \
  --console-log-level=notice \
  2>&1 | tee "${log}"

echo "=== Download summary ==="
for location in 1 2 3; do
  n=$(find "${DATA_ROOT}/Diff_Locations_Setup/Location${location}" -name 'IQ_*.dat' -size +1M 2>/dev/null | wc -l)
  echo "Location${location}: ${n}/24"
done
for distance in 5m 10m 15m 20m; do
  n=$(find "${DATA_ROOT}/Diff_Distances_Setup/${distance}" -name 'IQ_*.dat' -size +1M 2>/dev/null | wc -l)
  echo "${distance}: ${n}/24"
done
