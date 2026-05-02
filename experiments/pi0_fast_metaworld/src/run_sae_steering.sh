#!/bin/bash
# Bulk pi0_fast MetaWorld SAE-ActAdd sweep — one slurm job per task.
# Mirrors run_linear_final_sweep.sh task list and resource asks.
set -euo pipefail
DRY_RUN=${DRY_RUN:-false}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

OUTPUT_ROOT="experiments/pi0_fast_metaworld/sae_steering_results"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR"

# Paper-table task subset (10 tasks).
TASKS=(
    "coffee-push-v3" "push-v3" "pick-place-v3" "plate-slide-back-v3"
    "faucet-close-v3" "pick-place-wall-v3" "reach-v3" "coffee-pull-v3"
    "disassemble-v3" "stick-push-v3"
)
BASE_PORT=8800
SAE_ALPHAS="0.25 0.5 1.0 2.0"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))
    SCRIPT="${SCRIPT_DIR}/sae_task_${i}.sh"

    cat > "$SCRIPT" << HEADER
#!/bin/bash
#SBATCH --job-name=fm-sae-${i}
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem=224G
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/sae_task_${i}_%j.out
#SBATCH --error=${LOG_DIR}/sae_task_${i}_%j.err

set -e
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
export MUJOCO_GL=egl
cd ${REPO_ROOT}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TMPDIR="\${OPENPI_DATA_HOME}/tmp_xla"
mkdir -p "\$TMPDIR"

echo "=== pi0-fast MetaWorld SAE task ${i}: ${TASK} ==="
echo "Node: \$(hostname) | port: ${PORT}"
echo "Start: \$(date)"

uv run python experiments/pi0_fast_metaworld/src/sae_steering.py \\
    --task "${TASK}" \\
    --checkpoint-dir checkpoints/pi0_fast_metaworld/pi0_fast_metaworld_b200_bs512/2500/ \\
    --port ${PORT} \\
    --alphas ${SAE_ALPHAS} \\
    --num-episodes 1 \\
    --num-envs 16 \\
    --output-dir ${OUTPUT_ROOT}

echo "=== Done: \$(date) ==="
HEADER
    chmod +x "$SCRIPT"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] task_${i}: ${TASK}  -> $SCRIPT"
    else
        sbatch "$SCRIPT"
    fi
done
echo ""
echo "${#TASKS[@]} tasks submitted, αs=[$SAE_ALPHAS]"
