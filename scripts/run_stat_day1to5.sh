#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
OUT_ROOT=${OUT_ROOT:-}
STAT_ROOT=${STAT_ROOT:-outputs/stat_day1to5}
N_BOOTSTRAP=${N_BOOTSTRAP:-1000}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT=$(find outputs -maxdepth 1 -type d -name 'single_gpu_compare_*' | sort | tail -1)
fi

mkdir -p "${STAT_ROOT}/bootstrap"

file_level_csv() {
  local dir="$1"
  if [[ -f "${dir}/file_predictions.csv" ]]; then
    echo "${dir}/file_predictions.csv"
  else
    echo "${dir}/predictions.csv"
  fi
}

declare -A PREDS
PREDS[cnn_classifier]="$(file_level_csv "${OUT_ROOT}/osu_cnn_iq/classifier")"
PREDS[cnn_prototype_mean_prob]="$(file_level_csv "${OUT_ROOT}/osu_cnn_iq/prototype_mean_prob")"
PREDS[hybrid_classifier]="$(file_level_csv "${OUT_ROOT}/hybrid_cnnstem_cross_attn_chirp/classifier")"
PREDS[hybrid_prototype_mean_prob]="$(file_level_csv "${OUT_ROOT}/hybrid_cnnstem_cross_attn_chirp/prototype_mean_prob")"
if [[ -n "${HYB_PROTO_CW:-}" ]]; then
  PREDS[hybrid_prototype_confidence_weighted]="${HYB_PROTO_CW}"
fi

for method in "${!PREDS[@]}"; do
  pred="${PREDS[$method]}"
  if [[ ! -f "${pred}" ]]; then
    echo "Missing predictions for ${method}: ${pred}" >&2
    exit 1
  fi
  "${PY}" scripts/bootstrap_eval_ci.py \
    --predictions "${pred}" \
    --out "${STAT_ROOT}/bootstrap/${method}.csv" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --format wide
done

"${PY}" scripts/paired_compare_models.py \
  --a-pred "${PREDS[cnn_classifier]}" \
  --b-pred "${PREDS[hybrid_classifier]}" \
  --a-name cnn_classifier \
  --b-name hybrid_classifier \
  --out "${STAT_ROOT}/paired_cnn_classifier_vs_hybrid_classifier.csv" \
  --diff-out "${STAT_ROOT}/paired_cnn_classifier_vs_hybrid_classifier_diff.csv"

"${PY}" scripts/paired_compare_models.py \
  --a-pred "${PREDS[cnn_prototype_mean_prob]}" \
  --b-pred "${PREDS[hybrid_prototype_mean_prob]}" \
  --a-name cnn_prototype_mean_prob \
  --b-name hybrid_prototype_mean_prob \
  --out "${STAT_ROOT}/paired_cnn_proto_vs_hybrid_proto.csv" \
  --diff-out "${STAT_ROOT}/paired_cnn_proto_vs_hybrid_proto_diff.csv"

"${PY}" scripts/paired_compare_models.py \
  --a-pred "${PREDS[cnn_prototype_mean_prob]}" \
  --b-pred "${PREDS[hybrid_classifier]}" \
  --a-name cnn_prototype_mean_prob \
  --b-name hybrid_classifier \
  --out "${STAT_ROOT}/paired_cnn_proto_vs_hybrid_classifier.csv" \
  --diff-out "${STAT_ROOT}/paired_cnn_proto_vs_hybrid_classifier_diff.csv"

echo "stat_day1to5=${STAT_ROOT}"
