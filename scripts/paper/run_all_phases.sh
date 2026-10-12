#!/usr/bin/env bash
# Master runner: 7-GPU parallel (GPU0 reserved), batch=64 on A100-80G
set -euo pipefail
ROOT=${ROOT:-/data1/hcc/llm4RF}
PY=${PY:-/new_nfs/liuyida/anaconda3/envs/qwen3_lf/bin/python}
cd "${ROOT}"
source scripts/paper/lib/paper_env.sh
export GPUS BATCH LR BATCH_RX
chmod +x scripts/paper/phase*.sh scripts/paper/*.py 2>/dev/null || true

PHASES=${PHASES:-1,2,3,4,5,6}
IFS=',' read -ra P <<< "${PHASES}"

for p in "${P[@]}"; do
  case "${p}" in
    1) bash scripts/paper/phase1_manifests.sh ;;
    2) bash scripts/paper/phase2_cross_day.sh ;;
    3) bash scripts/paper/phase3_ablation.sh ;;
    4) bash scripts/paper/phase4_deployment.sh ;;
    5) bash scripts/paper/phase5_cross_receiver.sh ;;
    6) "${PY}" scripts/paper/phase6_edge_benchmark.py ;;
  esac
done

"${PY}" scripts/paper/aggregate_paper_ready.py --root "${ROOT}"
echo "All requested phases complete."
