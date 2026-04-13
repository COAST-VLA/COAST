#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Gap-fill sweep — resubmits contrastive + pos-only jobs with:
#   (a) the race-condition fix (merge-on-save) already patched in, and
#   (b) larger time limits sized per-task based on remaining gaps.
# Skip-existing logic means already-completed conditions are skipped instantly.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

CHECKPOINT_DIR="/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_pretrain_human300/multitask_learning/75000"
CONFIG="pi05_robocasa"
NUM_EPISODES=15
REPO_ROOT="/vast/projects/ungar/stellar/miaom/openpi-new"
OUTPUT_ROOT="experiments/pi05_robocasa/steering_results"

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

submit() {
    local name=$1 script=$2 timelim=$3
    if $DRY_RUN; then
        echo "[DRY-RUN] $name  (time=$timelim)"
        return
    fi
    echo "Submitting $name (time=$timelim)"
    sbatch --job-name="$name" --partition=dgx-b200 --gres=gpu:1 \
        --cpus-per-task=12 --mem=96G --time="$timelim" \
        --output="${LOG_DIR}/${name}_%j.out" \
        --error="${LOG_DIR}/${name}_%j.err" "$script"
}

# (task, port-contrastive, port-posonly, time-contrastive, time-posonly)
# time budgets sized to (new_conds × 7min + 15min load), rounded up
declare -a ROWS=(
    "CloseFridge                 8530 8540 02:30:00 02:30:00"
    "CoffeeSetupMug              8531 8541 08:00:00 02:00:00"
    "OpenDrawer                  8532 8542 08:00:00 02:00:00"
    "OpenStandMixerHead          8533 8543 06:00:00 -"
    "PickPlaceCounterToCabinet   8534 8544 06:30:00 02:00:00"
    "PickPlaceCounterToStove     8535 8545 08:00:00 02:00:00"
    "TurnOnElectricKettle        8536 8546 04:00:00 -"
)

for row in "${ROWS[@]}"; do
    read -r TASK PC PP TC TP <<< "$row"

    # ── Contrastive gap-fill (global + per_step_0 + per_step_9 + random) ──
    SCRIPT="${SCRIPT_DIR}/gapfill_contr_${TASK}.sh"
    cat > "$SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== GapFill contrastive ${TASK} | \$(hostname) | GPU=\${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_robocasa/src/conceptor_steering.py \\
    --task "${TASK}" --config ${CONFIG} --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} --port ${PC} \\
    --layers 5 11 --alphas 0.1 0.5 1.0 2.0 10.0 --betas 0.1 0.3 0.5 \\
    --strategies global per_step_0 per_step_9 \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done contrastive ${TASK} ==="
EOF
    chmod +x "$SCRIPT"
    submit "gapC-${TASK:0:8}" "$SCRIPT" "$TC"

    # ── Pos-only gap-fill (skip if already complete) ──
    if [[ "$TP" != "-" ]]; then
        SCRIPT="${SCRIPT_DIR}/gapfill_pos_${TASK}.sh"
        cat > "$SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== GapFill pos-only ${TASK} | \$(hostname) | GPU=\${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_robocasa/src/positive_only_steering.py \\
    --task "${TASK}" --config ${CONFIG} --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} --port ${PP} \\
    --layers 5 11 --alphas 0.1 0.5 1.0 2.0 10.0 --betas 0.1 0.3 0.5 \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done pos-only ${TASK} ==="
EOF
        chmod +x "$SCRIPT"
        submit "gapP-${TASK:0:8}" "$SCRIPT" "$TP"
    fi
done

echo ""
echo "Gap-fill submission complete."
