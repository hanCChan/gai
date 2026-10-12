#!/usr/bin/env bash
# Persist authorized G1 work independently of the chat, with retries and failure records.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON=/home/keyan/venvs/shen-stage1/bin/python
STATE="$ROOT/outputs/shen_p2a8/osu_download"
exec 9>"$STATE/background.lock"
flock -n 9 || exit 0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
completed=false
for attempt in 1 2 3 4 5; do
  if "$PYTHON" scripts/verify_shen_p2a8_osu_download.py; then
    completed=true
    break
  fi
  aria2c --input-file="$STATE/targets.aria2" --continue=true --check-integrity=true \
    --max-concurrent-downloads=32 --split=8 --max-connection-per-server=8 --min-split-size=1M \
    --file-allocation=none --disk-cache=32M --auto-file-renaming=false --allow-overwrite=false \
    --max-tries=5 --retry-wait=3 --connect-timeout=20 --timeout=120 \
    --save-session="$STATE/resume.aria2" --save-session-interval=10 --summary-interval=30 \
    --log="$STATE/aria2.log" --log-level=notice >> "$STATE/console.log" 2>&1 || true
done
if "$PYTHON" scripts/verify_shen_p2a8_osu_download.py; then
  completed=true
fi
if [[ "$completed" != true ]]; then
  "$PYTHON" - <<'PY'
from pathlib import Path
import json,datetime
p=Path('outputs/shen_p2a8/g1_osu_orchestration.json')
p.write_text(json.dumps({'status':'failed','reason':'Official OSU download incomplete after five resumable attempts; no partial capture used for training','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
  exit 1
fi
"$PYTHON" scripts/finish_shen_p2a8_g1.py --dataset osu >> outputs/shen_p2a8/g1_osu_orchestration.log 2>&1
