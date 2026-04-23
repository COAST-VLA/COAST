#!/usr/bin/env bash
# Flow-DPO sweep on LIBERO libero_spatial.
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

BASE_CKPT=${BASE_CKPT:-/home/kim34/projects_brandon/openpi-metaworld/checkpoints/openpi-libero-2000}
RESULTS_JSON=${RESULTS_JSON:-experiments/preference_bc/results_libero.json}
LIBERO_SUITE=${LIBERO_SUITE:-libero_spatial}
BETA=${BETA:-2000.0}

mkdir -p experiments/preference_bc/logs

uv run python -u -m experiments.preference_bc.run_preference_bc \
    --args.env libero \
    --args.base-ckpt "$BASE_CKPT" \
    --args.libero-suite "$LIBERO_SUITE" \
    --args.split train \
    --args.num-rollouts 15 \
    --args.num-train-steps 500 \
    --args.batch-size 8 \
    --args.eval-num-episodes 15 \
    --args.max-steps 220 \
    --args.replan-steps 5 \
    --args.beta "$BETA" \
    --args.results-json "$RESULTS_JSON"
