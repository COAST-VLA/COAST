#!/usr/bin/env bash
# Full ML45-train sweep for the filtered-BC baseline on MetaWorld.
#
# Launch with nohup so it survives terminal detach:
#     nohup bash experiments/filtered_bc/run_metaworld.sh \
#         > experiments/filtered_bc/logs/metaworld.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=egl
# JAX must release GPU memory between the train and eval phases of each task;
# without these two, its pool holds ~20 GB and the PyTorch build OOMs.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

BASE_CKPT=${BASE_CKPT:-/home/kim34/projects_brandon/openpi-metaworld/checkpoints/openpi-metaworld-5000}
RESULTS_JSON=${RESULTS_JSON:-experiments/filtered_bc/results_metaworld.json}

mkdir -p experiments/filtered_bc/logs

uv run python -u -m experiments.filtered_bc.run_filtered_bc \
    --args.env metaworld \
    --args.base-ckpt "$BASE_CKPT" \
    --args.split train \
    --args.num-rollouts 15 \
    --args.num-train-steps 500 \
    --args.batch-size 8 \
    --args.eval-num-episodes 15 \
    --args.results-json "$RESULTS_JSON"
