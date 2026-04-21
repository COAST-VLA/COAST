#!/bin/bash
#SBATCH --job-name=fast-lib-diag
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:0
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/pi0_fast_libero/logs/diag_%j.out
#SBATCH --error=experiments/pi0_fast_libero/logs/diag_%j.err

set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
cd /vast/projects/ungar/stellar/miaom/openpi-new

echo "=== pi0-fast LIBERO conceptor fitting ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

mkdir -p experiments/pi0_fast_libero/logs

uv run python experiments/pi0_fast_libero/src/conceptor_diagnostic.py \
    --checkpoint_step 2000 \
    --activations_dir "${OPENPI_DATA_HOME}/pi0fast-libero-activations-v1-2000-15env"

echo "=== Done: $(date) ==="
