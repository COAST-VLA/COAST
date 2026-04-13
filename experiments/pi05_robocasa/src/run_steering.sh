#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Conceptor Steering Sweep for pi0.5 RoboCasa — grouped by task
# ──────────────────────────────────────────────────────────────────────────────
# One SLURM job per task (7 jobs total).  Each job:
#   1. Loads the model ONCE on 1 GPU
#   2. Sweeps all (layer × alpha × beta × strategy) + baseline + random
#   3. Saves results under  steering_results/{task}/
#
# Per-job conditions:
#   1 baseline + 3 layers × 5 alphas × 3 betas × 3 strategies + 3×3 random = 145
#
# RoboCasa is ~10x slower per step than LIBERO, so jobs need more time.
#
# Usage:
#   bash experiments/pi05_robocasa/src/run_steering.sh          # submit all
#   bash experiments/pi05_robocasa/src/run_steering.sh --dry-run # list without submitting
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

# ── All 7 RoboCasa tasks (matching conceptor npz) ───────────────────────────
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

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))

    SCRIPT="${SCRIPT_DIR}/task_${i}.sh"
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
uv run experiments/pi05_robocasa/src/conceptor_steering.py \\
    --task "${TASK}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done: ${TASK} ==="
EOF
    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] task_${i}: ${TASK}  (port=${PORT})"
    else
        echo "Submitting task_${i}: ${TASK}"
        sbatch \
            --job-name="rc-steer-${i}" \
            --partition=dgx-b200 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=1-00:00:00 \
            --output="${LOG_DIR}/task_${i}_%j.out" \
            --error="${LOG_DIR}/task_${i}_%j.err" \
            "$SCRIPT"
    fi
done

echo ""
echo "Total jobs: ${#TASKS[@]}"
echo "Results will appear in: ${OUTPUT_ROOT}/{task_name}/"
