#!/bin/bash
#SBATCH --job-name=pi05_libero
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00

set -e

export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_TOKEN=$(cat ~/.hf_token)
export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')
export HF_HOME=/vast/projects/ungar/stellar/miaom/.cache/huggingface
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /vast/projects/ungar/stellar/miaom/openpi-metaworld

echo "=== Running on: $(hostname) ==="
echo "=== GPU info ==="
nvidia-smi

NORM_STATS_PATH="assets/pi05_libero/physical-intelligence/libero/norm_stats.json"
if [ -f "$NORM_STATS_PATH" ]; then
    echo "=== Norm stats already cached, skipping ==="
else
    echo "=== Computing norm stats ==="
    uv run scripts/compute_norm_stats.py --config-name pi05_libero
fi

echo "=== Starting training ==="
uv run scripts/train.py pi05_libero \
    --exp-name=libero_b200_bs512 \
    --overwrite
