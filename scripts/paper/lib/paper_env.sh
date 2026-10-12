#!/usr/bin/env bash
# Shared defaults for paper experiment scripts (7x A100-80G, GPU0 reserved).
set -euo pipefail

# GPU 0 is occupied by VLLM on this machine; use 1-7 for training.
export GPUS=${GPUS:-0}

# bs=16 used ~0.7GB on hybrid; bs=128 still << 80GB on A100.
export BATCH=${BATCH:-128}

# sqrt scaling vs bs=16, lr=1e-3: sqrt(128/16)=2.83 -> 3e-3
export LR=${LR:-3e-3}

# Cross-receiver: larger batches, same lr scale headroom.
export BATCH_RX=${BATCH_RX:-256}
