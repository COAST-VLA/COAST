#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
cd /vast/projects/ungar/stellar/miaom/openpi-new
echo "=== pi0_fast_libero train | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES} ==="

# Compute normalization stats (idempotent; skips recomputation if assets present)
uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero \
    --exp-name pi0_fast_libero_b200_bs512 \
    --batch-size 512 \
    --num-train-steps 2001 \
    --overwrite

echo "=== Done: pi0_fast_libero ==="
