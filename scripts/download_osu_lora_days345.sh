#!/usr/bin/env bash
# Download OSU LoRa Diff_Days_Indoor_Setup Day3-Day5 IQ_1 files.
# Official dataset: https://research.engr.oregonstate.edu/hamdaoui/datasets
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${ROOT}/data/raw/osu_lora"
BASE_URL="https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/Diff_Days_Indoor_Setup"
LIST="${DATA_ROOT}/download_days345.tsv"
LOG="${DATA_ROOT}/download_days345.log"
JOBS="${JOBS:-4}"
MIN_DAT_BYTES="${MIN_DAT_BYTES:-160000000}"

mkdir -p "${DATA_ROOT}"

python3 - "${DATA_ROOT}" "${BASE_URL}" "${LIST}" <<'PY'
import sys
from pathlib import Path

data_root = Path(sys.argv[1])
base_url = sys.argv[2]
list_path = Path(sys.argv[3])
rows = []

def add(day: int, device: int) -> None:
    rel = f"Diff_Days_Indoor_Setup/Day{day}/Device{device}"
    for name in ("IQ_1.dat", "IQ_1.sigmf-meta"):
        url = f"{base_url}/Day{day}/Device{device}/{name}"
        out = data_root / rel / name
        rows.append((url, out))

for day in (3, 4, 5):
    for device in range(1, 26):
        add(day, device)

with list_path.open("w", encoding="utf-8") as f:
    for url, out in rows:
        f.write(f"{url}\t{out}\n")

print(f"wrote {list_path} entries={len(rows)}")
PY

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

echo "Starting download at $(date -Is)" | tee "${LOG}"
echo "List: ${LIST}" | tee -a "${LOG}"
echo "Jobs: ${JOBS}" | tee -a "${LOG}"

while IFS=$'\t' read -r url target; do
  while [[ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]]; do
    wait -n || true
  done
  download_one "${url}" "${target}" >>"${LOG}" 2>&1 &
done < "${LIST}"

wait
echo "Download finished at $(date -Is)" | tee -a "${LOG}"

for day in 2 3 4 5; do
  count=$(find "${DATA_ROOT}/Diff_Days_Indoor_Setup/Day${day}" -name "IQ_1.dat" 2>/dev/null | wc -l)
  echo "Day${day}: ${count}/25 IQ_1.dat files" | tee -a "${LOG}"
done
