#!/usr/bin/env bash
# L0 step 2 (server): full run sharded over N GPUs (default 4); refuses to start unless the
# preflight decided GO_FULL_L0 with identical code. Then merge and apply the frozen rules.
#   bash phase_l0_full.sh /home/keyan/code-y-r0r1 /home/keyan/l0 [/home/keyan/code-y-r2a] [N_GPUS]
set -euo pipefail
ROOT=${1:?R0/R1 worktree}; OUT=${2:?output dir}; R2=${3:-}; N=${4:-4}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 -c "import json,sys;d=json.load(open('$OUT/preflight/preflight.json'));print('preflight:',d['decision'],d['reasons'],'estimate GPUh',round(d['full_estimate_reserved_gpuh_with_margin'],3));sys.exit(d['decision']!='GO_FULL_L0')"
R2ARG=(); [ -n "$R2" ] && R2ARG=(--r2-root "$R2")
mkdir -p "$OUT/full"; pids=(); shards=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i python3 run_l0.py --root "$ROOT" --out "$OUT/full/shard$i" --full --preflight-dir "$OUT/preflight" \
      --shard "$i/$N" "${R2ARG[@]}" > "$OUT/full/shard$i.log" 2>&1 &
  pids+=($!); shards+=("$OUT/full/shard$i")
done
fail=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "shard $i FAILED, see $OUT/full/shard$i.log"; fail=1; }; done
[ $fail -eq 0 ] || exit 1
python3 merge_l0.py --shards "${shards[@]}" --out "$OUT/full/merged"
python3 summarize_l0.py --l0 "$OUT/full/merged" --atlas "$OUT/atlas"
echo "== done: $OUT/full/merged/summary.json"
