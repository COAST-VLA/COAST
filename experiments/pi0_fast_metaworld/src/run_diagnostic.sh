#!/bin/bash
#SBATCH --job-name=fast-mw-diag
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:0
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/pi0_fast_metaworld/logs/diag_%j.out
#SBATCH --error=experiments/pi0_fast_metaworld/logs/diag_%j.err

set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
cd /vast/projects/ungar/stellar/miaom/openpi-new

echo "=== pi0-fast MetaWorld conceptor fitting ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

mkdir -p experiments/pi0_fast_metaworld/logs

uv run python experiments/pi0_fast_metaworld/src/conceptor_diagnostic.py \
    --checkpoint_step 2500

echo "=== Done: $(date) ==="
