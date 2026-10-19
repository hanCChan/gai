#!/usr/bin/env bash
# L0 phenomenon analysis on the server. Inference only; no network fitting.
#   bash phase_l0.sh /home/keyan/code-y-r0r1 /home/keyan/l0 [/home/keyan/code-y-r2a]
# Step 1-2 need only numpy/pandas (CPU). Step 3 needs torch + one GPU. Stop at the first failure
# and send the log back; do not edit thresholds after seeing outputs.
set -euo pipefail
ROOT=${1:?R0/R1 worktree}; OUT=${2:?output dir}; R2=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cd "$HERE"
echo "== git state of ROOT"; git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --short | head
echo "== 0. local tests";      for t in test_link_core.py test_physics.py smoke_physics_run.py smoke_summarize.py; do python3 $t | tail -1; done
echo "== 1. model-free atlas"; python3 atlas_l0.py --root "$ROOT" --out "$OUT/atlas" | tail -40
R2ARG=(); [ -n "$R2" ] && R2ARG=(--r2-root "$R2")
echo "== 2. smoke (2 jobs)";   python3 run_l0.py --root "$ROOT" --out "$OUT/smoke" --limit 2 "${R2ARG[@]}"
echo "== 3. full run";         python3 run_l0.py --root "$ROOT" --out "$OUT/full" "${R2ARG[@]}"
echo "== 4. summary";          python3 summarize_l0.py --l0 "$OUT/full" --atlas "$OUT/atlas"
cat "$OUT/full/timing.json"
