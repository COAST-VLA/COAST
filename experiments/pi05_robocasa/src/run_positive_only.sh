#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Positive-Only Conceptor Steering — 7 RoboCasa tasks
# ──────────────────────────────────────────────────────────────────────────────
# Minimal sweep: 2 layers × 2 alphas × 2 betas = 8 conditions per task
#
# Usage:
#   bash experiments/pi05_robocasa/src/run_positive_only.sh          # submit all
#   bash experiments/pi05_robocasa/src/run_positive_only.sh --dry-run # list only
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

CHECKPOINT_DIR="/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_pretrain_human300/multitask_learning/75000"
CONFIG="pi05_robocasa"
NUM_EPISODES=15
OUTPUT_ROOT="experiments/pi05_robocasa/steering_results"
REPO_ROOT="/vast/projects/ungar/stellar/miaom/openpi-new"

TASKS=(
    "CloseFridge"
    "CoffeeSetupMug"
    "OpenDrawer"
    "OpenStandMixerHead"
    "PickPlaceCounterToCabinet"
    "PickPlaceCounterToStove"
    "TurnOnElectricKettle"
)

BASE_PORT=8300  # offset from main steering jobs to avoid port conflicts

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))

    SCRIPT="${SCRIPT_DIR}/pos_only_task_${i}.sh"
    cat > "$SCRIPT" << 'SLURM_HEADER'
#!/bin/bash
set -e
SLURM_HEADER
    cat >> "$SCRIPT" << EOF
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Pos-Only Task ${i}: ${TASK} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_robocasa/src/positive_only_steering.py \\
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
        echo "[DRY-RUN] pos_only_task_${i}: ${TASK}  (port=${PORT})"
    else
        echo "Submitting pos_only_task_${i}: ${TASK}"
        sbatch \
            --job-name="rc-pos-${i}" \
            --partition=dgx-b200 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=0-06:00:00 \
            --output="${LOG_DIR}/pos_only_task_${i}_%j.out" \
            --error="${LOG_DIR}/pos_only_task_${i}_%j.err" \
            "$SCRIPT"
    fi
done

echo ""
echo "Total jobs: ${#TASKS[@]}"
echo "Conditions per task: 8 (2 layers × 2 alphas × 2 betas)"
echo "Results will be appended to existing summary.json files"
