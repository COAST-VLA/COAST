#!/bin/bash
#SBATCH --job-name=fast-mw-collect
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/pi0_fast_metaworld/collect_logs/collect_%j.out
#SBATCH --error=experiments/pi0_fast_metaworld/collect_logs/collect_%j.err

set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
cd /vast/projects/ungar/stellar/miaom/openpi-new

echo "=== pi0-fast MetaWorld activation collection ==="
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Start: $(date)"

# Collect activations for all 45 ML45 train tasks
uv run examples/metaworld/collect_activations_fast.py \
    --policy.config=pi0_fast_metaworld \
    --policy.dir=checkpoints/pi0_fast_metaworld/pi0_fast_metaworld_b200_bs512/2500/ \
    --split train \
    --num_envs 2 \
    --max_steps 300 \
    --replan_steps 10 \
    --output_dir /vast/projects/ungar/stellar/miaom/.cache/openpi/activations_fast

echo "=== Done: $(date) ==="
