#!/usr/bin/env bash
# Download OSU LoRa Diff_Receivers_Setup_Indoor_SameTx IQ files (RX1 + RX2, Device1-25).
# Only IQ .dat + .sigmf-meta are fetched; FFT .dat is derived online by the project.
# Official dataset: https://research.engr.oregonstate.edu/hamdaoui/datasets
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${ROOT}/data/raw/osu_lora"
SUBSET="Diff_Receivers_Setup/Diff_Receivers_Setup_Indoor_SameTx"
BASE_URL="https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/${SUBSET}"
LOG="${DATA_ROOT}/download_diff_receivers.log"
JOBS="${JOBS:-4}"
MIN_DAT_BYTES="${MIN_DAT_BYTES:-159000000}"

download_one() {
  local url="$1"
  local target="$2"
  local name
  name="$(basename "${target}")"
  mkdir -p "$(dirname "${target}")"
  if [[ "${name}" == *.dat ]]; then
    if [[ -f "${target}" ]] && [[ "$(stat -c%s "${target}")" -ge "${MIN_DAT_BYTES}" ]]; then
      echo "skip ${target}"
      return 0
    fi
  elif [[ -f "${target}" ]] && [[ "$(stat -c%s "${target}")" -gt 0 ]]; then
    echo "skip ${target}"
    return 0
  fi
  echo "download ${url} -> ${target}"
  curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 -C - -o "${target}" "${url}"
}
export -f download_one
export MIN_DAT_BYTES

echo "Starting Diff_Receivers download at $(date -Is)" | tee "${LOG}"
echo "Jobs: ${JOBS}" | tee -a "${LOG}"

for rx in RX1 RX2; do
  for device in $(seq 1 25); do
    for name in "Device${device}_IQ.dat" "Device${device}_IQ.sigmf-meta"; do
      url="${BASE_URL}/${rx}/${name}"
      target="${DATA_ROOT}/${SUBSET}/${rx}/${name}"
      while [[ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]]; do
        wait -n || true
      done
      download_one "${url}" "${target}" >>"${LOG}" 2>&1 &
    done
  done
done

wait
echo "Download finished at $(date -Is)" | tee -a "${LOG}"

for rx in RX1 RX2; do
  count=$(find "${DATA_ROOT}/${SUBSET}/${rx}" -name "Device*_IQ.dat" 2>/dev/null | wc -l)
  echo "${rx}: ${count}/25 IQ.dat files" | tee -a "${LOG}"
done
