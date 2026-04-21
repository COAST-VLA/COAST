#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Transfer sweep (Design B) — pi0.5 RoboCasa
# One SLURM job per target task. Each job iterates all 6 sources × (global+2 per-step).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

CHECKPOINT_DIR="/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_pretrain_human300/multitask_learning/75000"
CONFIG="pi05_robocasa"
NUM_EPISODES=15
OUTPUT_ROOT="experiments/pi05_robocasa/transfer_results"
RESULTS_ROOT="experiments/pi05_robocasa/steering_results"
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

BASE_PORT=8300

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

echo "═══ Submitting transfer jobs (${#TASKS[@]} targets) ═══"

SUBMITTED_IDS=()

for i in "${!TASKS[@]}"; do
    TARGET="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))

    SCRIPT="${SCRIPT_DIR}/transfer_${i}.sh"
    cat > "$SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Transfer job ${i}: target=${TARGET} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_robocasa/src/transfer_steering.py \\
    --target-task "${TARGET}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --results-root ${RESULTS_ROOT} \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done: target=${TARGET} ==="
EOF
    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] transfer_${i}: target=${TARGET} (port=${PORT})"
    else
        OUT=$(sbatch \
            --job-name="rc-transfer-${i}" \
            --partition=dgx-b200 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=0-08:00:00 \
            --output="${LOG_DIR}/transfer_${i}_%j.out" \
            --error="${LOG_DIR}/transfer_${i}_%j.err" \
            "$SCRIPT")
        echo "  transfer_${i} (${TARGET}): ${OUT}"
        JID=$(echo "$OUT" | awk '{print $NF}')
        SUBMITTED_IDS+=("$JID")
    fi
done

if ! $DRY_RUN; then
    echo ""
    echo "Submitted job IDs: ${SUBMITTED_IDS[*]}"
    # Emit colon-joined list for the aggregation dependency.
    printf '%s' "${SUBMITTED_IDS[0]}"
    for jid in "${SUBMITTED_IDS[@]:1}"; do printf ':%s' "$jid"; done
    printf '\n' > "${SCRIPT_DIR}/_last_job_ids.txt"
    ( printf '%s' "${SUBMITTED_IDS[0]}"; for jid in "${SUBMITTED_IDS[@]:1}"; do printf ':%s' "$jid"; done; printf '\n' ) > "${SCRIPT_DIR}/_last_job_ids.txt"
    echo "Job ID chain saved to ${SCRIPT_DIR}/_last_job_ids.txt"
fi

echo ""
echo "Total target jobs: ${#TASKS[@]}"
echo "Results will appear in: ${OUTPUT_ROOT}/target_{task_name}/"
