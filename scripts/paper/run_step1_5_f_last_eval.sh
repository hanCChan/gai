#!/usr/bin/env bash
# Step1.5 eval-only: F_cross_attn_chirp_plain last.pt vs existing best.pt results.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU=${GPU:-0}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

STEP1="${ROOT}/outputs/paper_ready_v3/step1_phase7_clean"
RUNS="${STEP1}/runs/F_cross_attn_chirp_plain"
OUT="${STEP1}/statistics/F_last_eval"
MANIFEST="data/paper/cross_day_day1to5_source_only.csv"

COMMON=(
  --manifest "${MANIFEST}"
  --batch-size 128
  --samples-per-file 256
  --eval-samples-per-file 256
  --dim 64
  --depth 2
  --device cuda
  --train-split train
  --val-split val
  --eval-split test
  --input-norm iq_rms
  --fft-norm log_zscore
)
MODEL=(
  --model-type rf_hstu
  --patch-embed-type cnn_stem
  --cnn-stem-dim 32
  --oob-fusion-type cross_attn_oob
  --use-oob-cross-attention
  --use-chirp-embedding
  --oob-norm zscore
)
EVAL_ONLY=(
  --mode classifier
  --file-vote-mode mean_logits
)

mkdir -p "${OUT}"

for seed in 0 1 2 3 4; do
  ckpt="${RUNS}/seed_${seed}/last.pt"
  out_dir="${OUT}/seed_${seed}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "SKIP seed ${seed}: missing ${ckpt}"
    continue
  fi
  echo "==> eval last.pt seed ${seed}"
  "${PY}" scripts/evaluate.py \
    "${COMMON[@]}" "${MODEL[@]}" "${EVAL_ONLY[@]}" \
    --seed "${seed}" \
    --checkpoint "${ckpt}" \
    --out-dir "${out_dir}"
done

echo "Done. Re-run: ${PY} scripts/paper/step1_5_statistics.py"
