#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive steering sweep — 17 jobs:
#   (1-7)   per_step_0 + comprehensive random, all 7 tasks
#   (8-14)  positive-only comprehensive, all 7 tasks
#   (15)    PPCtC re-confirmation @ 30 episodes (separate output_dir)
#   (16)    TOEK comprehensive per_step_9
#   (17)    PPCtC comprehensive global re-run
# All jobs use skip-existing-conditions logic, so re-runs are safe.
#
# Usage:
#   bash experiments/pi05_robocasa/src/run_comprehensive_sweep.sh
#   bash experiments/pi05_robocasa/src/run_comprehensive_sweep.sh --dry-run
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

CHECKPOINT_DIR="/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_pretrain_human300/multitask_learning/75000"
CONFIG="pi05_robocasa"
NUM_EPISODES=15
REPO_ROOT="/vast/projects/ungar/stellar/miaom/openpi-new"
OUTPUT_ROOT="experiments/pi05_robocasa/steering_results"
OUTPUT_ROOT_30EP="experiments/pi05_robocasa/steering_results_30ep"

TASKS=(
    "CloseFridge"
    "CoffeeSetupMug"
    "OpenDrawer"
    "OpenStandMixerHead"
    "PickPlaceCounterToCabinet"
    "PickPlaceCounterToStove"
    "TurnOnElectricKettle"
)

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

submit() {
    local name=$1 script=$2 timelim=$3
    if $DRY_RUN; then
        echo "[DRY-RUN] $name  (time=$timelim)  $script"
        return
    fi
    echo "Submitting $name (time=$timelim)"
    sbatch \
        --job-name="$name" \
        --partition=dgx-b200 \
        --gres=gpu:1 \
        --cpus-per-task=12 \
        --mem=96G \
        --time="$timelim" \
        --output="${LOG_DIR}/${name}_%j.out" \
        --error="${LOG_DIR}/${name}_%j.err" \
        "$script"
}

# ── Jobs 1-7: per_step_0 + comprehensive random ──
for i in 0 1 2 3 4 5 6; do
    TASK="${TASKS[$i]}"
    PORT=$((8500 + i))
    SCRIPT="${SCRIPT_DIR}/ps0rand_${i}_${TASK}.sh"
    cat > "$SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Job ps0rand ${i}: ${TASK} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_robocasa/src/conceptor_steering.py \\
    --task "${TASK}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --layers 5 11 \\
    --alphas 0.1 0.5 1.0 2.0 10.0 \\
    --betas 0.1 0.3 0.5 \\
    --strategies per_step_0 \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done: ${TASK} ==="
EOF
    chmod +x "$SCRIPT"
    submit "ps0rand-${i}" "$SCRIPT" "05:00:00"
done

# ── Jobs 8-14: positive-only comprehensive ──
for i in 0 1 2 3 4 5 6; do
    TASK="${TASKS[$i]}"
    PORT=$((8510 + i))
    SCRIPT="${SCRIPT_DIR}/posonly_${i}_${TASK}.sh"
    cat > "$SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Job posonly ${i}: ${TASK} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_robocasa/src/positive_only_steering.py \\
    --task "${TASK}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --layers 5 11 \\
    --alphas 0.1 0.5 1.0 2.0 10.0 \\
    --betas 0.1 0.3 0.5 \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done: ${TASK} ==="
EOF
    chmod +x "$SCRIPT"
    submit "posonly-${i}" "$SCRIPT" "05:00:00"
done

# ── Job 15: PPCtC re-confirmation @ 30 episodes (separate output_dir) ──
SCRIPT="${SCRIPT_DIR}/ppctc_reconf_30ep.sh"
cat > "$SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Job PPCtC re-conf 30ep | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_robocasa/src/conceptor_steering.py \\
    --task "PickPlaceCounterToCabinet" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes 30 \\
    --port 8520 \\
    --layers 5 \\
    --alphas 0.1 1.0 \\
    --betas 0.1 \\
    --strategies global \\
    --n-random-controls 0 \\
    --output-dir ${OUTPUT_ROOT_30EP}
echo "=== Done: PPCtC re-conf ==="
EOF
chmod +x "$SCRIPT"
submit "ppctc-reconf" "$SCRIPT" "02:00:00"

# ── Job 16: TOEK comprehensive per_step_9 ──
SCRIPT="${SCRIPT_DIR}/toek_per_step_9_comprehensive.sh"
cat > "$SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Job TOEK per_step_9 comprehensive | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_robocasa/src/conceptor_steering.py \\
    --task "TurnOnElectricKettle" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} \\
    --port 8521 \\
    --layers 5 11 \\
    --alphas 0.1 0.5 1.0 2.0 10.0 \\
    --betas 0.1 0.3 0.5 \\
    --strategies per_step_9 \\
    --n-random-controls 0 \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done: TOEK per_step_9 ==="
EOF
chmod +x "$SCRIPT"
submit "toek-ps9" "$SCRIPT" "04:00:00"

# ── Job 17: PPCtC comprehensive global re-run ──
SCRIPT="${SCRIPT_DIR}/ppctc_global_comprehensive.sh"
cat > "$SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Job PPCtC global comprehensive | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_robocasa/src/conceptor_steering.py \\
    --task "PickPlaceCounterToCabinet" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} \\
    --port 8522 \\
    --layers 5 11 \\
    --alphas 0.1 0.5 1.0 2.0 10.0 \\
    --betas 0.1 0.3 0.5 \\
    --strategies global \\
    --n-random-controls 0 \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done: PPCtC global ==="
EOF
chmod +x "$SCRIPT"
submit "ppctc-global" "$SCRIPT" "03:30:00"

echo ""
echo "All 17 jobs submitted."
