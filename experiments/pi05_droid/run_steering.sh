#!/usr/bin/env bash
# Launch the DROID pi0.5 steered policy server for ONE condition.
#
# DROID evaluation is manual — a human runs the physical robot client. This
# script just starts the WebSocket policy server with a single steering spec
# applied. To sweep conditions, run this script multiple times (stopping the
# server between runs).
#
# Usage:
#   bash experiments/pi05_droid/run_steering.sh                       # baseline
#   STRATEGY=global LAYER=11 ALPHA=1.0 BETA=0.3 \
#       bash experiments/pi05_droid/run_steering.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

: "${TASK:=PickUpPineapple}"
: "${STRATEGY:=baseline}"            # baseline | linear | global | per_step | positive_only | random
: "${LAYER:=11}"
: "${ALPHA:=1.0}"
: "${BETA:=0.3}"
: "${LINEAR_ALPHA:=0.5}"
: "${PORT:=8000}"
: "${CHECKPOINT_DIR:=${HOME}/.cache/openpi/openpi-assets/checkpoints/pi05_droid}"
: "${CONCEPTOR_NPZ:=${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}/droid_conceptors.npz}"

# Pick a free GPU if caller hasn't set one.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES before running (see CLAUDE.md GPU selection)." >&2
  exit 1
fi

echo "=== pi0.5 DROID steering server ==="
echo "  task       = ${TASK}"
echo "  strategy   = ${STRATEGY}"
echo "  layer      = ${LAYER}"
echo "  alpha      = ${ALPHA}      (conceptor aperture)"
echo "  beta       = ${BETA}       (conceptor mix weight)"
echo "  linear_a   = ${LINEAR_ALPHA}"
echo "  ckpt       = ${CHECKPOINT_DIR}"
echo "  conceptors = ${CONCEPTOR_NPZ}"
echo "  port       = ${PORT}"
echo "  GPU        = ${CUDA_VISIBLE_DEVICES}"

uv run experiments/pi05_droid/conceptor_steering.py \
    --task "${TASK}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --strategy "${STRATEGY}" \
    --layer "${LAYER}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    --linear-alpha "${LINEAR_ALPHA}" \
    --conceptor-npz "${CONCEPTOR_NPZ}" \
    --port "${PORT}"
