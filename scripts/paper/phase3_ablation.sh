#!/usr/bin/env bash
# Phase 3: Component ablation (Day1-4->Day5 source-only, 80ep)
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
EPOCHS=${EPOCHS:-80}
SEED=${SEED:-0}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
source scripts/paper/lib/job_helpers.sh
source scripts/paper/lib/paper_env.sh

TAG=${TAG:-$(date +%Y%m%d_%H%M%S)}
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
BASE=${BASE:-outputs/paper_runs/phase3_ablation_${TAG}_${COMMIT}}
RUNS="${BASE}/runs"
OUTS="${BASE}/outputs"
LOGS="${BASE}/logs"
JOBS_FILE="${BASE}/jobs.tsv"
mkdir -p "${RUNS}" "${OUTS}" "${LOGS}"
: > "${JOBS_FILE}"

MANIFEST="data/paper/cross_day_day1to5_source_only.csv"
COMMON=(
  --manifest "${MANIFEST}" --epochs "${EPOCHS}" --batch-size "${BATCH}" --lr "${LR}"
  --samples-per-file 256 --eval-samples-per-file 256
  --dim 64 --depth 2 --seed "${SEED}" --device cuda
  --train-split train --val-split val --eval-split test
  --input-norm iq_rms --fft-norm log_zscore
)

add_ablation_job() {
  local id=$1
  shift
  local extra=("$@")
  local marker="${OUTS}/${id}/file_predictions.csv"
  local cmd="$(mgpu_cmd_env) && \
'${PY}' scripts/finetune.py --out-dir '${RUNS}/${id}' ${COMMON[*]} ${extra[*]} && \
'${PY}' scripts/evaluate.py --checkpoint '${RUNS}/${id}/best.pt' \
  --mode classifier --file-vote-mode mean_logits --out-dir '${OUTS}/${id}' ${COMMON[*]}"
  mgpu_write_job "${JOBS_FILE}" "${id}" "${cmd}" "${marker}"
}

add_ablation_job A_cnn_iq --model-type osu_cnn --cnn-input-type iq --oob-norm none
add_ablation_job B_linear_no_oob --model-type rf_hstu --patch-embed-type linear --no-oob --oob-fusion-type no_oob
add_ablation_job C_cnn_no_oob --model-type rf_hstu --patch-embed-type cnn_stem --cnn-stem-dim 32 --no-oob --oob-fusion-type no_oob
add_ablation_job D_concat_oob --model-type rf_hstu --patch-embed-type cnn_stem --cnn-stem-dim 32 --oob-fusion-type concat_oob --oob-norm zscore
add_ablation_job E_cross_attn --model-type rf_hstu --patch-embed-type cnn_stem --cnn-stem-dim 32 --oob-fusion-type cross_attn_oob --use-oob-cross-attention --oob-norm zscore
add_ablation_job F_full_hybrid --model-type rf_hstu --patch-embed-type cnn_stem --cnn-stem-dim 32 --oob-fusion-type cross_attn_oob --use-oob-cross-attention --use-chirp-embedding --oob-norm zscore --label-smoothing 0.05 --weight-decay 5e-4
add_ablation_job G_center_loss --model-type rf_hstu --patch-embed-type cnn_stem --cnn-stem-dim 32 --oob-fusion-type cross_attn_oob --use-oob-cross-attention --use-chirp-embedding --oob-norm zscore --label-smoothing 0.05 --weight-decay 5e-4 --use-center-loss --center-loss-weight 0.01

echo "==> Launching $(grep -c . "${JOBS_FILE}" || echo 0) ablation jobs on GPUs ${GPUS}"
mgpu_run_jobs "${JOBS_FILE}" "${LOGS}/train_jobs"

"${PY}" scripts/summarize_results.py --input-dir "${OUTS}" --out "${BASE}/ablation_summary.csv" 2>/dev/null || true
echo "Phase 3 complete: ${BASE}"
