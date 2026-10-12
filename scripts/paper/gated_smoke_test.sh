#!/usr/bin/env bash
# 3-epoch smoke test: gated OOB + oob dropout + MixStyle + macro_f1 + SWA code path
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
OUT=${OUT:-outputs/paper_ready_v2/smoke_gated_oob}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

MANIFEST="data/paper/cross_day_day1to5_source_only.csv"
EVAL_COMMON=(
  --manifest "${MANIFEST}"
  --batch-size 64
  --seed 0
  --device cuda
  --samples-per-file 32
  --eval-samples-per-file 32
  --dim 64
  --depth 2
  --train-split train
  --val-split val
  --eval-split test
  --input-norm iq_rms
  --fft-norm log_zscore
  --oob-norm ratio
  --model-type rf_hstu
  --patch-embed-type cnn_stem
  --cnn-stem-dim 32
  --oob-fusion-type gated_oob
  --use-chirp-embedding
)
TRAIN_EXTRA=(
  --epochs 3
  --lr 1e-3
  --oob-dropout 0.3
  --mixstyle
  --use-swa
  --checkpoint-metric macro_f1
  --class-balanced-ce
  --loss-type focal
  --label-smoothing 0.05
  --weight-decay 5e-4
)

echo "==> Smoke train (3 ep)"
"${PY}" scripts/finetune.py --out-dir "${OUT}" "${EVAL_COMMON[@]}" "${TRAIN_EXTRA[@]}"

echo "==> Smoke eval best.pt"
"${PY}" scripts/evaluate.py \
  --checkpoint "${OUT}/best.pt" \
  --mode classifier \
  --file-vote-mode mean_logits \
  --out-dir "${OUT}/eval_best" \
  "${EVAL_COMMON[@]}"

if [[ -f "${OUT}/swa.pt" ]]; then
  echo "==> Smoke eval swa.pt"
  "${PY}" scripts/evaluate.py \
    --checkpoint "${OUT}/swa.pt" \
    --mode classifier \
    --file-vote-mode mean_logits \
    --out-dir "${OUT}/eval_swa" \
    "${EVAL_COMMON[@]}"
fi

echo "SMOKE PASS: ${OUT}"
