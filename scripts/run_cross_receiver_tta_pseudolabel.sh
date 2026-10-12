#!/usr/bin/env bash
# Cross-receiver target-unlabeled adaptation quick matrix.
# Protocol:
#   RX1 -> RX2: source labels from RX1, target unlabeled windows from RX2, target labels only for final metrics.
#   RX2 -> RX1: source labels from RX2, target unlabeled windows from RX1, target labels only for final metrics.
# Fixed source model: Hybrid oob_ratio + CFO, 30 epochs.
# No center loss / SupCon / hard margin / multi-scale / deeper backbone / receiver-style augmentation.
set -euo pipefail

ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
GPU_ID=${GPU_ID:-1}
RUN_ROOT=${RUN_ROOT:-runs/cross_receiver_tta_pseudolabel}
OUT_ROOT=${OUT_ROOT:-outputs/cross_receiver_tta_pseudolabel}

EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}
ADAPT_BATCH_SIZE=${ADAPT_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
PIN_MEMORY=${PIN_MEMORY:-1}
CFO_FEATURE_TYPE=${CFO_FEATURE_TYPE:-both}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

cd "${ROOT}"
mkdir -p logs "${OUT_ROOT}"
LOG="logs/cross_receiver_tta_pseudolabel_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

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
  --cfo-feature-type "${CFO_FEATURE_TYPE}"
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

eval_adapt_set() {
  local direction="$1" manifest="$2" ckpt="$3" tag="$4"; shift 4
  local adapt_args=("$@")
  local out_base="${OUT_ROOT}/${direction}/D2_oob_ratio_cfo/${tag}"
  eval_one "${manifest}" "${ckpt}" "${out_base}" classifier mean_logits "${adapt_args[@]}"
  eval_one "${manifest}" "${ckpt}" "${out_base}" classifier confidence_weighted "${adapt_args[@]}"
  eval_one "${manifest}" "${ckpt}" "${out_base}" prototype mean_prob "${adapt_args[@]}"
}

eval_pseudo_proto() {
  local direction="$1" manifest="$2" ckpt="$3" tag="$4" threshold="$5" momentum="$6"
  local out_base="${OUT_ROOT}/${direction}/D2_oob_ratio_cfo/${tag}"
  eval_one "${manifest}" "${ckpt}" "${out_base}" prototype mean_prob \
    --adapt-mode pseudo_proto \
    --pseudo-threshold "${threshold}" \
    --pseudo-topk-per-class 128 \
    --pseudo-min-per-class 16 \
    --prototype-momentum "${momentum}"
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
  "${PY}" - <<'PY'
import csv
import json
from pathlib import Path

root = Path("outputs/cross_receiver_tta_pseudolabel")
rows = []
for metrics_path in sorted(root.rglob("metrics.json")):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("adapt_mode") != "pseudo_proto":
        continue
    rows.append({
        "experiment": str(metrics_path.parent.relative_to(root)).replace("\\", "/"),
        "adapt_mode": metrics.get("adapt_mode", ""),
        "pseudo_threshold": metrics.get("pseudo_threshold", ""),
        "pseudo_topk_per_class": metrics.get("pseudo_topk_per_class", ""),
        "pseudo_min_per_class": metrics.get("pseudo_min_per_class", ""),
        "prototype_momentum": metrics.get("prototype_momentum", ""),
        "num_pseudo_selected": metrics.get("num_pseudo_selected", ""),
        "num_classes_updated": metrics.get("num_classes_updated", ""),
        "pseudo_class_distribution": metrics.get("pseudo_class_distribution", ""),
        "window_acc": metrics.get("window_acc", ""),
        "file_acc": metrics.get("file_acc", ""),
        "macro_f1": metrics.get("macro_f1", ""),
    })
out = root / "pseudo_stats.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "experiment",
    "adapt_mode",
    "pseudo_threshold",
    "pseudo_topk_per_class",
    "pseudo_min_per_class",
    "prototype_momentum",
    "num_pseudo_selected",
    "num_classes_updated",
    "pseudo_class_distribution",
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
  echo "cross_receiver_tta_pseudolabel started at $(date --iso-8601=seconds)"
  echo "gpu_id=${GPU_ID} epochs=${EPOCHS} batch_size=${BATCH_SIZE} adapt_batch_size=${ADAPT_BATCH_SIZE} num_workers=${NUM_WORKERS} pin_memory=${PIN_STATUS}"
  echo "source_model=Hybrid_oob_ratio_CFO cfo_feature_type=${CFO_FEATURE_TYPE}"
  for dir_entry in "${DIRECTIONS[@]}"; do
    direction="${dir_entry%%:*}"
    manifest="${dir_entry#*:}"
    train_source "${direction}" "${manifest}"
    ckpt="${CKPT_PATH}"

    eval_adapt_set "${direction}" "${manifest}" "${ckpt}" none --adapt-mode none
    eval_adapt_set "${direction}" "${manifest}" "${ckpt}" bn_adapt --adapt-mode bn_adapt --adapt-steps 1
    eval_adapt_set "${direction}" "${manifest}" "${ckpt}" entropy_min_s1_lr1e4 --adapt-mode entropy_min --adapt-steps 1 --adapt-lr 1e-4
    eval_adapt_set "${direction}" "${manifest}" "${ckpt}" entropy_min_s3_lr1e4 --adapt-mode entropy_min --adapt-steps 3 --adapt-lr 1e-4
    eval_pseudo_proto "${direction}" "${manifest}" "${ckpt}" pseudo_proto_thr08_m05 0.8 0.5
    eval_pseudo_proto "${direction}" "${manifest}" "${ckpt}" pseudo_proto_thr09_m05 0.9 0.5
    eval_pseudo_proto "${direction}" "${manifest}" "${ckpt}" pseudo_proto_thr08_m08 0.8 0.8
  done

  run_step "summarize tta/pseudolabel" "${PY}" scripts/summarize_results.py --root "${OUT_ROOT}"
  run_step "write pseudo stats" write_pseudo_stats
  echo
  echo "cross_receiver_tta_pseudolabel finished at $(date --iso-8601=seconds)"
} 2>&1 | tee "${LOG}"
