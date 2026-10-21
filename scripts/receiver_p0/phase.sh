#!/usr/bin/env bash
# P0 + PoC driver: run everything in order, stop on first failure.
# Usage: bash phase.sh [stage]   stage in {p0, poc, all}; default all.
# GPU: uses CUDA 0-3. Total ~1.5 GPUh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/data/miniforge/envs/env-3.12/bin/python3
OUT=/workspace/rl1/gai/outputs/receiver_p0
PAIRS=$OUT/pairs/pairs.json
STAGE="${1:-all}"

mkdir -p "$OUT"/{p0a,p0c,p0_wisig,pairs,poc,poc_eval}

run_p0 () {
  # P0-A: 30 frozen models, 3 shards (~3 min total)
  for d in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$d "$PY" "$HERE/run_p0a.py" --out "$OUT/p0a" --shard $d/3 \
      > "$OUT/p0a/log_shard$d.log" 2>&1 &
  done
  wait
  # WiSig mirror on GPU 3
  CUDA_VISIBLE_DEVICES=3 "$PY" "$HERE/run_p0_wisig.py" --out "$OUT/p0_wisig" \
    > "$OUT/wisig.log" 2>&1
  # P0-C post-processing
  "$PY" "$HERE/run_p0c.py" --p0a "$OUT/p0a" --out "$OUT/p0c"
  # Freeze hard-pair sets
  "$PY" "$HERE/select_pairs.py" --p0a "$OUT/p0a" --out "$OUT/pairs"
}

run_poc () {
  # CE anchor first: lambda=0 must reproduce the frozen recipe
  for f in 0 1; do for s in 20261013 20261014 20261015; do
    CUDA_VISIBLE_DEVICES=$(( (f*3+s) % 4 )) "$PY" "$HERE/train_pair_poc.py" \
      --fold $f --branch ce --lambda 0 --seed $s --pairs "$PAIRS" --out "$OUT/poc" \
      > "$OUT/poc/anchor_f${f}_s${s}.log" 2>&1
  done; done
  for f in 0 1; do for s in 20261013 20261014 20261015; do
    "$PY" "$HERE/check_ce_anchor.py" --fold $f --seed $s --pairs "$PAIRS" --out "$OUT/poc"
  done; done
  echo 'anchor checks written; inspect outputs/receiver_p0/poc/anchor_check_*.json'
  echo 'if max_abs_param_dev > 0 or epochs differ -> STOP and investigate before arms'
  # 36 arm fits: branch x lambda x fold x seed, round-robin on 4 GPUs
  i=0
  for f in 0 1; do for br in sub drift pair; do for lam in 0.1 1.0; do for s in 20261013 20261014 20261015; do
    g=$(( i % 4 )); i=$((i+1))
    echo "f$f $br lam$lam s$s -> gpu$g"
    CUDA_VISIBLE_DEVICES=$g "$PY" "$HERE/train_pair_poc.py" \
      --fold $f --branch $br --lambda $lam --seed $s --pairs "$PAIRS" --out "$OUT/poc" \
      > "$OUT/poc/f${f}_${br}_${lam}_${s}.log" 2>&1 &
    if (( i % 4 == 0 )); then wait; fi
  done; done; done; done
  wait
  # Evaluate all PoC checkpoints on outer RX
  CUDA_VISIBLE_DEVICES=0 "$PY" "$HERE/eval_pair_poc.py" --poc-out "$OUT/poc" \
    --pairs "$PAIRS" --out "$OUT/poc_eval" > "$OUT/poc_eval/log.txt" 2>&1
  # Final report skeleton
  "$PY" "$HERE/summarize_p0.py" --p0a "$OUT/p0a" --p0c "$OUT/p0c" \
    --wisig "$OUT/p0_wisig" --poc-eval "$OUT/poc_eval" --out /workspace/rl1/gai/reports/receiver_p0
}

case "$STAGE" in
  p0)  run_p0 ;;
  poc) run_poc ;;
  all) run_p0; run_poc ;;
  *)   echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "P0 driver done: $STAGE"
