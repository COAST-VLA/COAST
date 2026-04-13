#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# TRIMMED Conceptor Steering Sweep — top-parameter-only version
# ──────────────────────────────────────────────────────────────────────────────
# Picks the top-performing hyperparams from the CloseFridge full sweep:
#   layers=[11], alphas=[0.1,0.5,1.0], betas=[0.1,0.3],
#   strategies=[global,per_step_9], n_random_controls=1
# → 12 steered + baseline + 1 random = 14 conditions per task.
#
# Skips task_0 (CloseFridge full sweep already running).
# Short time budget (2.5h) so each job fits in remaining billing quota.
#
# Usage:
#   bash experiments/pi05_robocasa/src/run_steering_trim.sh
#   bash experiments/pi05_robocasa/src/run_steering_trim.sh --dry-run
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

BASE_PORT=8400

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

# Skip index 0 (CloseFridge — full sweep already running)
for i in 1 2 3 4 5 6; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))

    SCRIPT="${SCRIPT_DIR}/trim_task_${i}.sh"
    cat > "$SCRIPT" << 'SLURM_HEADER'
#!/bin/bash
set -e
SLURM_HEADER
    cat >> "$SCRIPT" << EOF
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Trim Task ${i}: ${TASK} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_robocasa/src/conceptor_steering.py \\
    --task "${TASK}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --layers 11 \\
    --alphas 0.1 0.5 1.0 \\
    --betas 0.1 0.3 \\
    --strategies global per_step_9 \\
    --n-random-controls 1 \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done: ${TASK} ==="
EOF
    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] trim_task_${i}: ${TASK}  (port=${PORT})"
    else
        echo "Submitting trim_task_${i}: ${TASK}"
        sbatch \
            --job-name="rc-trim-${i}" \
            --partition=dgx-b200 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=02:30:00 \
            --output="${LOG_DIR}/trim_task_${i}_%j.out" \
            --error="${LOG_DIR}/trim_task_${i}_%j.err" \
            "$SCRIPT"
    fi
done

echo ""
echo "Submitted 6 trimmed jobs (tasks 1-6). Task 0 (CloseFridge) untouched."
