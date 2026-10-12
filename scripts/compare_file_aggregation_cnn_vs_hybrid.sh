#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
OUT_ROOT=${OUT_ROOT:-}
ANALYSIS_ROOT=${ANALYSIS_ROOT:-outputs/file_aggregation_analysis}

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT=$(find outputs -maxdepth 1 -type d -name 'single_gpu_compare_*' | sort | tail -1)
fi

if [[ -z "${OUT_ROOT}" || ! -d "${OUT_ROOT}" ]]; then
  echo "Could not infer OUT_ROOT. Set OUT_ROOT=outputs/<run_dir>." >&2
  exit 1
fi

mkdir -p "${ANALYSIS_ROOT}"

file_level_csv() {
  local dir="$1"
  if [[ -f "${dir}/file_predictions.csv" ]]; then
    echo "${dir}/file_predictions.csv"
  else
    echo "${dir}/predictions.csv"
  fi
}

analyze() {
  local name="$1"
  local pred="$2"
  local out_dir="${ANALYSIS_ROOT}/${name}"
  if [[ ! -f "${pred}" ]]; then
    echo "Missing predictions for ${name}: ${pred}" >&2
    exit 1
  fi
  "${PY}" scripts/analyze_file_aggregation.py --predictions "${pred}" --out-dir "${out_dir}"
  cp "${out_dir}/file_analysis.csv" "${ANALYSIS_ROOT}/${name}_file_analysis.csv"
}

CNN_CLS="${OUT_ROOT}/osu_cnn_iq/classifier/predictions.csv"
CNN_PROTO="${OUT_ROOT}/osu_cnn_iq/prototype_mean_prob/predictions.csv"
HYB_CLS="${OUT_ROOT}/hybrid_cnnstem_cross_attn_chirp/classifier/predictions.csv"
HYB_PROTO_MP="${OUT_ROOT}/hybrid_cnnstem_cross_attn_chirp/prototype_mean_prob/predictions.csv"
HYB_PROTO_CW=${HYB_PROTO_CW:-}

analyze cnn_classifier "${CNN_CLS}"
analyze cnn_prototype "${CNN_PROTO}"
analyze hybrid_classifier "${HYB_CLS}"
analyze hybrid_prototype "${HYB_PROTO_MP}"

if [[ -n "${HYB_PROTO_CW}" && -f "${HYB_PROTO_CW}" ]]; then
  analyze hybrid_prototype_confidence_weighted "${HYB_PROTO_CW}"
fi

"${PY}" scripts/paired_compare_models.py \
  --a-pred "$(file_level_csv "${OUT_ROOT}/osu_cnn_iq/classifier")" \
  --b-pred "$(file_level_csv "${OUT_ROOT}/hybrid_cnnstem_cross_attn_chirp/classifier")" \
  --a-name cnn_classifier \
  --b-name hybrid_classifier \
  --out "${ANALYSIS_ROOT}/paired_cnn_classifier_vs_hybrid_classifier.csv" \
  --diff-out "${ANALYSIS_ROOT}/cnn_classifier_vs_hybrid_classifier_diff.csv"

"${PY}" scripts/paired_compare_models.py \
  --a-pred "$(file_level_csv "${OUT_ROOT}/osu_cnn_iq/prototype_mean_prob")" \
  --b-pred "$(file_level_csv "${OUT_ROOT}/hybrid_cnnstem_cross_attn_chirp/prototype_mean_prob")" \
  --a-name cnn_prototype_mean_prob \
  --b-name hybrid_prototype_mean_prob \
  --out "${ANALYSIS_ROOT}/paired_cnn_proto_vs_hybrid_proto.csv" \
  --diff-out "${ANALYSIS_ROOT}/cnn_proto_vs_hybrid_proto_diff.csv"

"${PY}" - <<'PY' "${ANALYSIS_ROOT}" "${OUT_ROOT}"
import csv
import json
import sys
from pathlib import Path

analysis_root = Path(sys.argv[1])
out_root = sys.argv[2]
files = [
    ("cnn_classifier", analysis_root / "cnn_classifier_file_analysis.csv", Path(out_root) / "osu_cnn_iq/classifier/metrics.json"),
    ("cnn_prototype_mean_prob", analysis_root / "cnn_prototype_file_analysis.csv", Path(out_root) / "osu_cnn_iq/prototype_mean_prob/metrics.json"),
    ("hybrid_classifier", analysis_root / "hybrid_classifier_file_analysis.csv", Path(out_root) / "hybrid_cnnstem_cross_attn_chirp/classifier/metrics.json"),
    ("hybrid_prototype_mean_prob", analysis_root / "hybrid_prototype_file_analysis.csv", Path(out_root) / "hybrid_cnnstem_cross_attn_chirp/prototype_mean_prob/metrics.json"),
]
if (analysis_root / "hybrid_prototype_confidence_weighted_file_analysis.csv").exists():
    files.append((
        "hybrid_prototype_confidence_weighted",
        analysis_root / "hybrid_prototype_confidence_weighted_file_analysis.csv",
        Path(out_root) / "hybrid_cnnstem_cross_attn_chirp/prototype_confidence_weighted/metrics.json",
    ))
rows = []
for name, path, metrics_path in files:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        data = list(csv.DictReader(f))
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    n = len(data)
    vote_file_acc = sum(int(r["file_correct"]) for r in data) / max(1, n)
    mean_window = sum(float(r["window_acc_inside_file"]) for r in data) / max(1, n)
    mean_dom = sum(float(r["dominant_pred_ratio"]) for r in data) / max(1, n)
    rows.append({
        "method": name,
        "out_root": out_root,
        "num_files": n,
        "eval_window_acc": metrics.get("window_acc", ""),
        "eval_file_acc": metrics.get("file_acc", ""),
        "eval_macro_f1": metrics.get("macro_f1", ""),
        "vote_majority_file_acc": vote_file_acc,
        "mean_window_acc_inside_file": mean_window,
        "mean_dominant_pred_ratio": mean_dom,
    })
out = analysis_root / "compare_summary.csv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"compare_summary={out}")
PY

echo "file_aggregation_analysis=${ANALYSIS_ROOT}"
