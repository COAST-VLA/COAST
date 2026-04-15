#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Optimized Conceptor Steering Sweep for pi0.5 RoboCasa
# ──────────────────────────────────────────────────────────────────────────────
# Two-stage pipeline:
#   Stage 1 (CPU-only):  select_parameters.py reads conceptors, picks
#                        best layer + sweet-spot alphas + safe betas.
#   Stage 2 (GPU):       run the EXISTING conceptor_steering.py with
#                        the narrowed parameter list.
#
# Usage:
#   bash experiments/pi05_robocasa/src/run_steering_optimized.sh
#   bash experiments/pi05_robocasa/src/run_steering_optimized.sh --dry-run
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Fixed params ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR="/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_pretrain_human300/multitask_learning/75000"
CONFIG="pi05_robocasa"
NUM_EPISODES=15
OUTPUT_ROOT="experiments/pi05_robocasa/steering_results"
REPO_ROOT="/vast/projects/ungar/stellar/miaom/openpi-new"
PYTHON="${REPO_ROOT}/.venv/bin/python"

CONCEPTOR_NPZ="/vast/projects/ungar/stellar/miaom/.cache/openpi/robocasa_conceptors.npz"
PARAM_JSON="${REPO_ROOT}/experiments/pi05_robocasa/selected_params.json"

# ── All 7 RoboCasa tasks ─────────────────────────────────────────────────────
TASKS=(
    "CloseFridge"
    "CoffeeSetupMug"
    "OpenDrawer"
    "OpenStandMixerHead"
    "PickPlaceCounterToCabinet"
    "PickPlaceCounterToStove"
    "TurnOnElectricKettle"
)

BASE_PORT=8200

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Parameter Selection (CPU-only, ~2 seconds)
# ══════════════════════════════════════════════════════════════════════════════
echo "═══ Stage 1: Parameter Selection ═══"
$PYTHON "${REPO_ROOT}/experiments/shared/select_parameters.py" \
    --conceptor-npz "$CONCEPTOR_NPZ" \
    --output-json "$PARAM_JSON"

# Parse the JSON to get the selected parameters
BEST_LAYER=$($PYTHON -c "import json; d=json.load(open('$PARAM_JSON')); print(d['best_layer'])")
SEL_ALPHAS=$($PYTHON -c "import json; d=json.load(open('$PARAM_JSON')); print(' '.join(str(a) for a in d['selected_alphas']))")
SEL_BETAS=$($PYTHON -c "import json; d=json.load(open('$PARAM_JSON')); print(' '.join(str(b) for b in d['selected_betas']))")

echo ""
echo "Selected: layer=${BEST_LAYER}, alphas=[${SEL_ALPHAS}], betas=[${SEL_BETAS}]"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Steering Sweep with Narrowed Parameters
# ══════════════════════════════════════════════════════════════════════════════
echo "═══ Stage 2: Submitting Steering Jobs ═══"

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))

    SCRIPT="${SCRIPT_DIR}/task_opt_${i}.sh"
    cat > "$SCRIPT" << 'SLURM_HEADER'
#!/bin/bash
set -e
SLURM_HEADER
    cat >> "$SCRIPT" << EOF
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Task ${i}: ${TASK} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
echo "=== Optimized sweep: layer=${BEST_LAYER}, alphas=[${SEL_ALPHAS}], betas=[${SEL_BETAS}] ==="
uv run experiments/pi05_robocasa/src/conceptor_steering.py \\
    --task "${TASK}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --output-dir ${OUTPUT_ROOT} \\
    --layers ${BEST_LAYER} \\
    --alphas ${SEL_ALPHAS} \\
    --betas ${SEL_BETAS}
echo "=== Done: ${TASK} ==="
EOF
    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] task_${i}: ${TASK}  (port=${PORT})"
    else
        echo "Submitting task_${i}: ${TASK}"
        sbatch \
            --job-name="rc-opt-${i}" \
            --partition=dgx-b200 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=1-00:00:00 \
            --output="${LOG_DIR}/task_opt_${i}_%j.out" \
            --error="${LOG_DIR}/task_opt_${i}_%j.err" \
            "$SCRIPT"
    fi
done

echo ""
echo "Total jobs: ${#TASKS[@]}"
echo "Results will appear in: ${OUTPUT_ROOT}/{task_name}/"
