#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# diffusion_policy / RoboCasa steering pipeline driver
#
# Three stages:
#   1. build_conceptors.py    → ~/.cache/diffusion_policy/diffusion_policy_conceptors.npz
#   2. select_parameters.py   → selected_params.json
#   3. steering.py            → one invocation per task, in-process eval.
#
# Edit CHECKPOINT and ACTIVATIONS_DIR below before the first run. Everything
# runs in the diffusion_policy venv (.venv at the repo root).
#
# Typical invocation (from repo root):
#   bash experiments/robocasa_steering/run_steering.sh
#   bash experiments/robocasa_steering/run_steering.sh --skip-build --skip-select
#   bash experiments/robocasa_steering/run_steering.sh --dry-run
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=false
SKIP_BUILD=false
SKIP_SELECT=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=true ;;
        --skip-build)  SKIP_BUILD=true ;;
        --skip-select) SKIP_SELECT=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── User-editable ──────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-checkpoints/latest.ckpt}"
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-activations/latest}"   # from collect_activations_robocasa.py
SPLIT="${SPLIT:-pretrain}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-15}"
NUM_ENVS="${NUM_ENVS:-5}"
DEVICE="${DEVICE:-cuda:0}"

STRATEGIES="linear global per_step positive_only random"
SWEEP_LAYERS="5 8 11"
SWEEP_ALPHAS="0.1 0.5 1.0 2.0 10.0"
SWEEP_BETAS="0.1 0.3"
LINEAR_ALPHAS="0.1 0.5 1.0"

# Curated 7-task subset, mirrors openpi-metaworld's RoboCasa default set.
TASKS=(
    "CloseFridge"
    "CoffeeSetupMug"
    "OpenDrawer"
    "OpenStandMixerHead"
    "PickPlaceCounterToCabinet"
    "PickPlaceCounterToStove"
    "TurnOnElectricKettle"
)

# ── Auto-derived ──────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${REPO_ROOT}/experiments/robocasa_steering"
PYTHON="${REPO_ROOT}/.venv/bin/python"

DP_DATA_HOME="${DP_DATA_HOME:-${HOME}/.cache/diffusion_policy}"
CONCEPTOR_NPZ="${DP_DATA_HOME}/diffusion_policy_conceptors.npz"
PARAM_JSON="${EXP_DIR}/selected_params.json"

OUTPUT_ROOT="${EXP_DIR}/steering_results"
mkdir -p "$OUTPUT_ROOT" "$DP_DATA_HOME"

# ── Stage 1 ───────────────────────────────────────────────────────────────
if ! $SKIP_BUILD; then
    echo "=== Stage 1: build_conceptors.py ==="
    CMD=(
        "$PYTHON" "${EXP_DIR}/build_conceptors.py"
        --activations-dir "$ACTIVATIONS_DIR"
        --output-npz "$CONCEPTOR_NPZ"
        --layers 0 3 5 8 11
        --alphas 0.1 0.5 1.0 2.0 10.0
    )
    echo "  ${CMD[*]}"
    $DRY_RUN || "${CMD[@]}"
else
    echo "=== Stage 1: skipped ==="
fi

# ── Stage 2 ───────────────────────────────────────────────────────────────
if ! $SKIP_SELECT; then
    echo ""
    echo "=== Stage 2: select_parameters.py ==="
    CMD=(
        "$PYTHON" "${EXP_DIR}/select_parameters.py"
        --conceptor-npz "$CONCEPTOR_NPZ"
        --output-json "$PARAM_JSON"
    )
    echo "  ${CMD[*]}"
    $DRY_RUN || "${CMD[@]}"
else
    echo "=== Stage 2: skipped ==="
fi

# ── Stage 3: one call per task ────────────────────────────────────────────
echo ""
echo "=== Stage 3: steering.py (one call per task) ==="
for TASK in "${TASKS[@]}"; do
    CMD=(
        "$PYTHON" "${EXP_DIR}/steering.py"
        --checkpoint "$CHECKPOINT"
        --conceptor_npz "$CONCEPTOR_NPZ"
        --task "$TASK"
        --split "$SPLIT"
        --device "$DEVICE"
        --num_rollouts "$NUM_ROLLOUTS"
        --num_envs "$NUM_ENVS"
        --layers $SWEEP_LAYERS
        --alphas $SWEEP_ALPHAS
        --betas $SWEEP_BETAS
        --linear_alphas $LINEAR_ALPHAS
        --strategies $STRATEGIES
        --output_dir "$OUTPUT_ROOT"
    )
    echo ""
    echo "  $TASK"
    echo "  ${CMD[*]}"
    $DRY_RUN || "${CMD[@]}"
done

echo ""
echo "=== Done. Results under: $OUTPUT_ROOT ==="
