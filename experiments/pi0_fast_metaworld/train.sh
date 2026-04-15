#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
cd /vast/projects/ungar/stellar/miaom/openpi-new
echo "=== pi0_fast_metaworld train | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES} ==="

# Compute normalization stats (idempotent; skips recomputation if assets present)
uv run scripts/compute_norm_stats.py --config-name pi0_fast_metaworld

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_metaworld \
    --exp-name pi0_fast_metaworld_b200_bs512 \
    --overwrite

echo "=== Done: pi0_fast_metaworld ==="
