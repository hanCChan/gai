#!/usr/bin/env bash
# L0 step 1 (server): CPU tests, model-free atlas, real-model preflight on ONE GPU (one job per
# architecture). Stops after writing preflight.json; it never starts the full run.
#   bash phase_l0_preflight.sh /home/keyan/code-y-r0r1 /home/keyan/l0 [/home/keyan/code-y-r2a]
set -euo pipefail
ROOT=${1:?R0/R1 worktree}; OUT=${2:?output dir}; R2=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; mkdir -p "$OUT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "== code";  git -C "$HERE" rev-parse HEAD; git -C "$HERE" status --short | head
echo "== R0/R1"; git -C "$ROOT" rev-parse HEAD || true
test -f "$ROOT/configs/receiver_r01/registration.json" && test -d "$ROOT/outputs/receiver_r01/formal"
echo "== 0. CPU tests"
for t in test_link_core.py test_physics.py smoke_physics_run.py smoke_summarize.py; do python3 $t | tail -1; done
echo "== 1. model-free atlas (CPU)"; python3 atlas_l0.py --root "$ROOT" --out "$OUT/atlas" | tail -60
R2ARG=(); [ -n "$R2" ] && R2ARG=(--r2-root "$R2")
echo "== 2. preflight on GPU 0"
CUDA_VISIBLE_DEVICES=0 python3 run_l0.py --root "$ROOT" --out "$OUT/preflight" --preflight "${R2ARG[@]}"
echo "== review $OUT/preflight/preflight.json, checks.csv, intervention_validation.csv before the full run"
