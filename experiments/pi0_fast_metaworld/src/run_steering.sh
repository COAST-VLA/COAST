#!/bin/bash
#SBATCH --job-name=fast-mw-steer
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem=224G
#SBATCH --time=1-12:00:00
#SBATCH --output=experiments/pi0_fast_metaworld/logs/steer_%A_%a.out
#SBATCH --error=experiments/pi0_fast_metaworld/logs/steer_%A_%a.err
#SBATCH --array=0-44

set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
cd /vast/projects/ungar/stellar/miaom/openpi-new

# --- B200 / JAX tuning (single GPU) -------------------------------------
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TMPDIR="/vast/projects/ungar/stellar/miaom/.cache/openpi/tmp_xla"
mkdir -p "$TMPDIR"

TASKS=(
    assembly-v3
    basketball-v3
    button-press-topdown-v3
    button-press-topdown-wall-v3
    button-press-v3
    button-press-wall-v3
    coffee-button-v3
    coffee-pull-v3
    coffee-push-v3
    dial-turn-v3
    disassemble-v3
    door-close-v3
    door-open-v3
    drawer-close-v3
    drawer-open-v3
    faucet-close-v3
    faucet-open-v3
    hammer-v3
    handle-press-side-v3
    handle-press-v3
    handle-pull-side-v3
    handle-pull-v3
    lever-pull-v3
    peg-insert-side-v3
    peg-unplug-side-v3
    pick-out-of-hole-v3
    pick-place-v3
    pick-place-wall-v3
    plate-slide-back-side-v3
    plate-slide-back-v3
    plate-slide-side-v3
    plate-slide-v3
    push-back-v3
    push-v3
    push-wall-v3
    reach-v3
    reach-wall-v3
    shelf-place-v3
    soccer-v3
    stick-pull-v3
    stick-push-v3
    sweep-into-v3
    sweep-v3
    window-close-v3
    window-open-v3
)

TASK_NAME="${TASKS[${SLURM_ARRAY_TASK_ID}]}"

echo "=== pi0-fast MetaWorld steering sweep ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}  Task: ${TASK_NAME}"
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Start: $(date)"

mkdir -p experiments/pi0_fast_metaworld/logs

uv run python experiments/pi0_fast_metaworld/src/conceptor_steering.py \
    --task "${TASK_NAME}"

echo "=== Done: $(date) ==="
