#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Collect steered activations for pi0.5 LIBERO
# ──────────────────────────────────────────────────────────────────────────────
# Single SLURM job that:
#   1. Loads model once on 1 GPU
#   2. Starts collection-mode server with per-task conceptor steering hooks
#   3. Runs 15-episode eval for each of the 10 LIBERO tasks
#   4. Saves post-steering activations to $OPENPI_DATA_HOME/activations/pi05_steered_activations/
#
# Usage:
#   bash experiments/pi05_libero/src/run_collect_steered.sh          # submit
#   bash experiments/pi05_libero/src/run_collect_steered.sh --dry-run # show command
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

REPO_ROOT="/vast/projects/ungar/stellar/miaom/openpi-new"
SCRIPT_DIR="${REPO_ROOT}/experiments/pi05_libero/steering_results/scripts"
LOG_DIR="${REPO_ROOT}/experiments/pi05_libero/steering_results/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

SCRIPT="${SCRIPT_DIR}/collect_steered.sh"
cat > "$SCRIPT" << 'EOF'
#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd /vast/projects/ungar/stellar/miaom/openpi-new
echo "=== Steered activation collection | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES} ==="
echo "Output: ${OPENPI_DATA_HOME}/activations/pi05_steered_activations/"
uv run experiments/pi05_libero/src/collect_steered_activations.py --num_episodes 15
echo "=== Collection complete ==="
EOF
chmod +x "$SCRIPT"

if $DRY_RUN; then
    echo "[DRY-RUN] Would submit: $SCRIPT"
    echo "  Output: \$OPENPI_DATA_HOME/activations/pi05_steered_activations/"
else
    echo "Submitting steered activation collection job..."
    JOB_ID=$(sbatch \
        --job-name="steer-collect" \
        --partition=dgx-b200 \
        --gres=gpu:1 \
        --cpus-per-task=12 \
        --mem=96G \
        --time=0-10:00:00 \
        --output="${LOG_DIR}/collect_steered_%j.out" \
        --error="${LOG_DIR}/collect_steered_%j.err" \
        "$SCRIPT" | awk '{print $4}')
    echo "Submitted job: ${JOB_ID}"
    echo "Log: ${LOG_DIR}/collect_steered_${JOB_ID}.out"
    echo "Monitor: squeue -j ${JOB_ID}"
fi
