#!/usr/bin/env bash
# Re-run this launcher to resume. aria2 control files and sessions are saved to disk.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATE="$ROOT/logs/osu_download"
SESSION="osu_download"
mkdir -p "$STATE"
if [[ "${1:-}" != "--worker" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    if flock -n "$STATE/worker.lock" true; then
      tmux new-window -t "$SESSION" -n resume -c "$ROOT" "bash scripts/run_osu_download_tmux.sh --worker; result=\$?; echo Worker_exit=\$result; exec bash"
      echo "Resumed in existing tmux session: $SESSION"
    else
      echo "Download worker already running in tmux session: $SESSION"
    fi
    exit 0
  fi
  tmux new-session -d -s "$SESSION" -c "$ROOT" "bash scripts/run_osu_download_tmux.sh --worker; result=\$?; echo Worker_exit=\$result; exec bash"
  echo "Started: tmux attach -t $SESSION"
  exit 0
fi
exec 9>"$STATE/worker.lock"
flock -n 9 || { echo "Another OSU download worker is running"; exit 1; }
cd "$ROOT"
exec > >(tee -a "$STATE/console.log") 2>&1
if [[ ! -f "$STATE/targets.aria2" ]]; then
  python3 scripts/prepare_osu_download.py --scope all
fi
echo "Starting/resuming OSU download $(date -Is)"
if [[ -f "$STATE/max_concurrent" && -z "${MAX_CONCURRENT:-}" ]]; then
  MAX_CONCURRENT=$(cat "$STATE/max_concurrent")
  [[ "$MAX_CONCURRENT" =~ ^[0-9]+$ ]] || { echo "Invalid saved concurrency"; exit 1; }
fi
echo "Files concurrent=${MAX_CONCURRENT:-32}, splits=${SPLIT:-16}, no speed limit"
while true; do
  if aria2c --input-file="$STATE/targets.aria2" \
    --continue=true --check-integrity=true \
    --max-concurrent-downloads="${MAX_CONCURRENT:-32}" \
    --split="${SPLIT:-16}" --max-connection-per-server="${SPLIT:-16}" \
    --min-split-size=4M --max-overall-download-limit=0 --max-download-limit=0 \
    --file-allocation=none --disk-cache=64M \
    --auto-file-renaming=false --allow-overwrite=false \
    --always-resume=true --max-tries=20 --retry-wait=5 \
    --connect-timeout=20 --timeout=60 --lowest-speed-limit=0 \
    --save-session="$STATE/resume.aria2" --save-session-interval=5 \
    --auto-save-interval=1 --force-save=true \
    --enable-rpc=true --rpc-listen-all=false --rpc-listen-port=16801 \
    --max-download-result=2000 --summary-interval=15 \
    --console-log-level=notice --log="$STATE/aria2.log" --log-level=notice; then
    if find "$ROOT/data/raw/osu_lora" -name '*.aria2' -print -quit | read -r _; then
      echo "aria2 stopped with unfinished block files; retaining progress and resuming in 20 seconds."
      sleep 20
    else
      break
    fi
  else
    code=$?
    echo "aria2 exit=$code; resume state retained. Retrying in 20 seconds."
    sleep 20
  fi
done
echo "Downloads finished. Independently checking original SHA256 and manifest paths."
python3 scripts/verify_osu_download.py
echo "OSU verification finished $(date -Is)"
