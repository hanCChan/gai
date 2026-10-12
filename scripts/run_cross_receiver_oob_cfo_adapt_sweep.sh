#!/usr/bin/env bash
# Cross-receiver RF-specific target-unlabeled adaptation sweep.
# Main method under test: OOB/CFO-aware pseudo-prototype calibration.
# No center loss / SupCon / hard margin / multi-scale / deeper backbone / receiver-style augmentation.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
RUN_ROOT=${RUN_ROOT:-runs/cross_receiver_oob_cfo_adapt_sweep}
OUT_ROOT=${OUT_ROOT:-outputs/cross_receiver_oob_cfo_adapt_sweep}

EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}
ADAPT_BATCH_SIZE=${ADAPT_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
PIN_MEMORY=${PIN_MEMORY:-1}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs "${OUT_ROOT}"
LOG="logs/cross_receiver_oob_cfo_adapt_sweep_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

LOADER_ARGS=( --num-workers "${NUM_WORKERS}" )
if [[ "${PIN_MEMORY}" == "1" ]] && "${PY}" scripts/finetune.py --help 2>/dev/null | grep -q -- '--pin-memory'; then
  LOADER_ARGS+=( --pin-memory )
  PIN_STATUS="enabled"
elif [[ "${PIN_MEMORY}" == "1" ]]; then
  PIN_STATUS="requested-but-unsupported(skipped)"
else
  PIN_STATUS="disabled"
fi

COMMON_TRAIN=(
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --samples-per-file 256
  --eval-samples-per-file 256
  --dim 64
  --depth 2
  --device cuda
  --label-smoothing 0.05
  --weight-decay 5e-4
  --input-norm iq_rms
  --fft-norm log_zscore
  --oob-norm ratio
  "${LOADER_ARGS[@]}"
)

EVAL_COMMON=(
  --batch-size "${BATCH_SIZE}"
  --adapt-batch-size "${ADAPT_BATCH_SIZE}"
  --samples-per-file 256
  --eval-samples-per-file 256
  --device cuda
  "${LOADER_ARGS[@]}"
)

HYBRID_D2=(
  --patch-embed-type cnn_stem
  --cnn-stem-dim 32
  --oob-fusion-type cross_attn_oob
  --use-oob-cross-attention
  --use-chirp-embedding
  --use-cfo-feature
  --cfo-feature-type both
  --cfo-feature-norm train_zscore
)

DIRECTIONS=(
  "rx1_to_rx2:data/manifest_cross_receiver_rx1_to_rx2.csv"
  "rx2_to_rx1:data/manifest_cross_receiver_rx2_to_rx1.csv"
)

run_step() {
  local name="$1"; shift
  echo
  echo "==> $(date --iso-8601=seconds) ${name}"
  "$@"
}

eval_one() {
  local manifest="$1" ckpt="$2" out_base="$3" mode="$4" vote="$5"; shift 5
  local extra=("$@")
  run_step "eval ${out_base##*/} ${mode} ${vote} ${extra[*]:-}" \
    "${PY}" scripts/evaluate.py --manifest "${manifest}" \
      --checkpoint "${ckpt}" "${EVAL_COMMON[@]}" \
      --mode "${mode}" --file-vote-mode "${vote}" \
      "${extra[@]}" \
      --out-dir "${out_base}/${mode}_${vote}"
}

eval_group() {
  local direction="$1" manifest="$2" ckpt="$3" tag="$4"; shift 4
  local adapt_args=("$@")
  local out_base="${OUT_ROOT}/${direction}/D2_oob_ratio_cfo/${tag}"
  eval_one "${manifest}" "${ckpt}" "${out_base}" classifier mean_logits "${adapt_args[@]}"
  eval_one "${manifest}" "${ckpt}" "${out_base}" classifier confidence_weighted "${adapt_args[@]}"
  eval_one "${manifest}" "${ckpt}" "${out_base}" prototype mean_prob "${adapt_args[@]}"
}

train_source() {
  local direction="$1" manifest="$2"
  local ckpt_dir="${RUN_ROOT}/${direction}/D2_oob_ratio_cfo"
  if [[ -f "${ckpt_dir}/best.pt" ]]; then
    echo "reuse_checkpoint=${ckpt_dir}/best.pt"
  else
    run_step "train ${direction}/D2_oob_ratio_cfo" \
      "${PY}" scripts/finetune.py --manifest "${manifest}" \
        "${COMMON_TRAIN[@]}" "${HYBRID_D2[@]}" \
        --out-dir "${ckpt_dir}"
  fi
  CKPT_PATH="${ckpt_dir}/best.pt"
}

write_pseudo_stats() {
  "${PY}" - "${OUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for metrics_path in sorted(root.rglob("metrics.json")):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("adapt_mode") not in {"pseudo_proto", "oob_cfo_pseudo_proto"}:
        continue
    experiment = str(metrics_path.parent.relative_to(root)).replace("\\", "/")
    direction = experiment.split("/", 1)[0]
    rows.append({
        "direction": direction,
        "experiment": experiment,
        "adapt_mode": metrics.get("adapt_mode", ""),
        "pseudo_threshold": metrics.get("pseudo_threshold", ""),
        "prototype_momentum": metrics.get("prototype_momentum", ""),
        "num_pseudo_selected": metrics.get("num_pseudo_selected", ""),
        "num_classes_updated": metrics.get("num_classes_updated", ""),
        "pseudo_class_distribution": metrics.get("pseudo_class_distribution", ""),
        "num_rejected_by_confidence": metrics.get("num_rejected_by_confidence", ""),
        "num_rejected_by_cls_proto_disagree": metrics.get("num_rejected_by_cls_proto_disagree", ""),
        "num_rejected_by_cfo": metrics.get("num_rejected_by_cfo", ""),
        "num_rejected_by_oob": metrics.get("num_rejected_by_oob", ""),
        "mean_cfo_z_selected": metrics.get("mean_cfo_z_selected", ""),
        "mean_oob_sim_selected": metrics.get("mean_oob_sim_selected", ""),
        "window_acc": metrics.get("window_acc", ""),
        "file_acc": metrics.get("file_acc", ""),
        "macro_f1": metrics.get("macro_f1", ""),
    })
out = root / "pseudo_stats.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "direction",
    "experiment",
    "adapt_mode",
    "pseudo_threshold",
    "prototype_momentum",
    "num_pseudo_selected",
    "num_classes_updated",
    "pseudo_class_distribution",
    "num_rejected_by_confidence",
    "num_rejected_by_cls_proto_disagree",
    "num_rejected_by_cfo",
    "num_rejected_by_oob",
    "mean_cfo_z_selected",
    "mean_oob_sim_selected",
    "window_acc",
    "file_acc",
    "macro_f1",
]
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"pseudo_stats={out} rows={len(rows)}")
PY
}

{
  echo "cross_receiver_oob_cfo_adapt_sweep started at $(date --iso-8601=seconds)"
  echo "gpu_id=${GPU_ID} epochs=${EPOCHS} batch_size=${BATCH_SIZE} adapt_batch_size=${ADAPT_BATCH_SIZE} num_workers=${NUM_WORKERS} pin_memory=${PIN_STATUS}"
  echo "source_model=Hybrid_oob_ratio_CFO"
  for dir_entry in "${DIRECTIONS[@]}"; do
    direction="${dir_entry%%:*}"
    manifest="${dir_entry#*:}"
    train_source "${direction}" "${manifest}"
    ckpt="${CKPT_PATH}"

    eval_group "${direction}" "${manifest}" "${ckpt}" A_none --adapt-mode none
    eval_group "${direction}" "${manifest}" "${ckpt}" B_pseudo_proto_thr08_m05 \
      --adapt-mode pseudo_proto --pseudo-threshold 0.8 --pseudo-topk-per-class 128 --pseudo-min-per-class 16 --prototype-momentum 0.5
    eval_group "${direction}" "${manifest}" "${ckpt}" C_oob_cfo_thr08_m05_z20_oob03 \
      --adapt-mode oob_cfo_pseudo_proto --pseudo-threshold 0.8 --prototype-momentum 0.5 --cfo-max-z 2.0 --oob-sim-threshold 0.3
    eval_group "${direction}" "${manifest}" "${ckpt}" D_oob_cfo_thr09_m05_z20_oob03 \
      --adapt-mode oob_cfo_pseudo_proto --pseudo-threshold 0.9 --prototype-momentum 0.5 --cfo-max-z 2.0 --oob-sim-threshold 0.3
    eval_group "${direction}" "${manifest}" "${ckpt}" E_oob_cfo_thr08_m08_z20_oob03 \
      --adapt-mode oob_cfo_pseudo_proto --pseudo-threshold 0.8 --prototype-momentum 0.8 --cfo-max-z 2.0 --oob-sim-threshold 0.3
    eval_group "${direction}" "${manifest}" "${ckpt}" F_oob_cfo_thr08_m05_z15_oob04 \
      --adapt-mode oob_cfo_pseudo_proto --pseudo-threshold 0.8 --prototype-momentum 0.5 --cfo-max-z 1.5 --oob-sim-threshold 0.4
  done

  run_step "summarize oob/cfo adapt sweep" "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"
  run_step "write pseudo stats" write_pseudo_stats
  echo
  echo "cross_receiver_oob_cfo_adapt_sweep finished at $(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"
