#!/bin/bash
# pi0.5 RoboCasa SAE-ActAdd sweep — one slurm job per task.
# Reads v_sae from $OPENPI_DATA_HOME/robocasa_pi05_sae_vectors.npz.
# 7 tasks × 1 layer × 2 αs.
set -euo pipefail
DRY_RUN=${DRY_RUN:-false}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pi05_pretrain_human300/multitask_learning/75000}"
CONFIG="pi05_robocasa"
NUM_EPISODES=15

OUTPUT_ROOT="experiments/pi05_robocasa/sae_steering_results"
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
BASE_PORT=8900
SWEEP_LAYER=11
SAE_ALPHAS="0.5 1.0"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))
    SCRIPT="${SCRIPT_DIR}/sae_task_${i}.sh"

    cat > "$SCRIPT" << HEADER
#!/bin/bash
#SBATCH --job-name=rc-sae-${i}
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/sae_task_${i}_%j.out
#SBATCH --error=${LOG_DIR}/sae_task_${i}_%j.err

set -e
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}

echo "=== pi0.5 RoboCasa SAE task ${i}: ${TASK} ==="
echo "Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES:-auto} | port: ${PORT}"
echo "Start: \$(date)"

uv run experiments/pi05_robocasa/src/sae_steering.py \\
    --task "${TASK}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --output-dir ${REPO_ROOT}/${OUTPUT_ROOT} \\
    --layer ${SWEEP_LAYER} \\
    --alphas ${SAE_ALPHAS}

echo "=== Done: \$(date) ==="
HEADER
    chmod +x "$SCRIPT"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] task_${i}: ${TASK}  port=${PORT}  -> $SCRIPT"
    else
        sbatch "$SCRIPT"
    fi
done
echo ""
echo "${#TASKS[@]} tasks submitted, layer=$SWEEP_LAYER  αs=[$SAE_ALPHAS]"
