#!/bin/bash
# One spectrum-matched random control per task, at the best-global hyperparams.
set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

REPO_ROOT="/vast/projects/ungar/stellar/miaom/openpi-new"
OUTPUT_ROOT="experiments/pi05_robocasa/steering_results"
SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

TASKS=(
    "CloseFridge"
    "CoffeeSetupMug"
    "OpenDrawer"
    "OpenStandMixerHead"
    "PickPlaceCounterToCabinet"
    "PickPlaceCounterToStove"
    "TurnOnElectricKettle"
)

BASE_PORT=8600
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))
    SCRIPT="${SCRIPT_DIR}/randmatch_${TASK}.sh"
    cat > "$SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== RandMatched ${TASK} | \$(hostname) | GPU=\${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_robocasa/src/random_control_matched.py \\
    --task "${TASK}" \\
    --port ${PORT} \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done: ${TASK} ==="
EOF
    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] randmatch-$i ${TASK}"
    else
        echo "Submitting randmatch-$i ${TASK}"
        sbatch --job-name="rmatch-${i}" --partition=dgx-b200 --gres=gpu:1 \
            --cpus-per-task=12 --mem=96G --time=00:45:00 \
            --output="${LOG_DIR}/randmatch_${i}_%j.out" \
            --error="${LOG_DIR}/randmatch_${i}_%j.err" \
            "$SCRIPT"
    fi
done
