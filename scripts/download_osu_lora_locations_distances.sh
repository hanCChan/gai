#!/usr/bin/env bash
# Generate download list / fetch OSU LoRa Diff_Locations_Setup and Diff_Distances_Setup IQ files.
# Path templates (verified against official index):
#   Diff_Locations_Setup/Location{l}/IQ_{raw_device}.dat
#   Diff_Distances_Setup/{5m,10m,15m,20m}/IQ_{raw_device}.dat
# flat layout: IQ_n corresponds to raw Device n (1..25).
#
# Does NOT run downloads unless DRY_RUN=0 is set explicitly.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-data/raw/osu_lora}"
DRY_RUN="${DRY_RUN:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
LIST="${DATA_ROOT}/download_locations_distances.tsv"
LOG="${ROOT}/logs/download_osu_lora_locations_distances.out"

mkdir -p "${DATA_ROOT}" "${ROOT}/logs"

python3 - "${DATA_ROOT}" "${LIST}" <<'PY'
import sys
from pathlib import Path

data_root = Path(sys.argv[1])
list_path = Path(sys.argv[2])
base = "https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset"
rows: list[tuple[str, Path]] = []

location_ids = (1, 2, 3)
distance_ids = ("5m", "10m", "15m", "20m")

for location in location_ids:
    rel = f"Diff_Locations_Setup/Location{location}"
    for raw_device in range(1, 26):
        for name in (f"IQ_{raw_device}.dat", f"IQ_{raw_device}.sigmf-meta"):
            url = f"{base}/{rel}/{name}"
            out = data_root / rel / name
            rows.append((url, out))

for distance in distance_ids:
    rel = f"Diff_Distances_Setup/{distance}"
    for raw_device in range(1, 26):
        for name in (f"IQ_{raw_device}.dat", f"IQ_{raw_device}.sigmf-meta"):
            url = f"{base}/{rel}/{name}"
            out = data_root / rel / name
            rows.append((url, out))

list_path.parent.mkdir(parents=True, exist_ok=True)
with list_path.open("w", encoding="utf-8") as f:
    for url, out in rows:
        f.write(f"{url}\t{out}\n")

print(f"wrote {list_path} entries={len(rows)}")
PY

should_skip() {
  local target="$1"
  if [[ -f "${target}" ]] && [[ "$(stat -c%s "${target}" 2>/dev/null || echo 0)" -gt 0 ]]; then
    return 0
  fi
  return 1
}

download_one() {
  local url="$1"
  local target="$2"
  if should_skip "${target}"; then
    echo "skip ${target}"
    return 0
  fi
  mkdir -p "$(dirname "${target}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 -C - -o ${target} ${url}"
    return 0
  fi
  echo "download ${url} -> ${target}"
  curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 -C - -o "${target}" "${url}"
}
export -f download_one should_skip
export DRY_RUN

{
  echo "=== download_osu_lora_locations_distances.sh ==="
  echo "started_at=$(date -Is)"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "MAX_PARALLEL=${MAX_PARALLEL}"
  echo "LIST=${LIST}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "mode=dry_run (set DRY_RUN=0 to download)"
  else
    echo "mode=download"
  fi
} | tee "${LOG}"

while IFS=$'\t' read -r url target; do
  [[ -z "${url}" ]] && continue
  while [[ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]]; do
    wait -n || true
  done
  download_one "${url}" "${target}" >>"${LOG}" 2>&1 &
done < "${LIST}"

wait

{
  echo "finished_at=$(date -Is)"
  for location in 1 2 3; do
    count=$(find "${DATA_ROOT}/Diff_Locations_Setup/Location${location}" -name 'IQ_*.dat' 2>/dev/null | wc -l)
    echo "Location${location}: ${count}/25 IQ_*.dat files on disk"
  done
  for distance in 5m 10m 15m 20m; do
    count=$(find "${DATA_ROOT}/Diff_Distances_Setup/${distance}" -name 'IQ_*.dat' 2>/dev/null | wc -l)
    echo "${distance}: ${count}/25 IQ_*.dat files on disk"
  done
} | tee -a "${LOG}"
