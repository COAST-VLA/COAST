#!/bin/bash
#SBATCH --job-name=fast-lib-smoke
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem=224G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/pi0_fast_libero/logs/smoke_%j.out
#SBATCH --error=experiments/pi0_fast_libero/logs/smoke_%j.err

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

TASK="KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"

echo "=== pi0-fast LIBERO smoke test (global_a1.0_b0.1 only) ==="
echo "Task: ${TASK}"
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Start: $(date)"

mkdir -p experiments/pi0_fast_libero/logs

uv run python experiments/pi0_fast_libero/src/conceptor_steering.py \
    --task "${TASK}" \
    --checkpoint-dir checkpoints/pi0_fast_libero/pi0_fast_libero_b200_bs512/2000/ \
    --global-alphas 1.0 \
    --per-step-combined-alphas \
    --positive-only-alphas \
    --betas 0.1 \
    --num-episodes 30 \
    --output-dir experiments/pi0_fast_libero/steering_results_smoke

echo "=== Done: $(date) ==="
