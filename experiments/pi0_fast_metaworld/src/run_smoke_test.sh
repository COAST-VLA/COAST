#!/bin/bash
#SBATCH --job-name=fast-mw-smoke
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem=224G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/pi0_fast_metaworld/logs/smoke_%j.out
#SBATCH --error=experiments/pi0_fast_metaworld/logs/smoke_%j.err

set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
cd /vast/projects/ungar/stellar/miaom/openpi-new

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TMPDIR="/vast/projects/ungar/stellar/miaom/.cache/openpi/tmp_xla"
mkdir -p "$TMPDIR"

TASK="assembly-v3"

echo "=== pi0-fast MetaWorld smoke test (global_a1.0_b0.1 only) ==="
echo "Task: ${TASK}"
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Start: $(date)"

mkdir -p experiments/pi0_fast_metaworld/logs

uv run python experiments/pi0_fast_metaworld/src/conceptor_steering.py \
    --task "${TASK}" \
    --global-alphas 1.0 \
    --per-step-combined-alphas \
    --positive-only-alphas \
    --betas 0.1 \
    --output-dir experiments/pi0_fast_metaworld/steering_results_smoke

echo "=== Done: $(date) ==="
