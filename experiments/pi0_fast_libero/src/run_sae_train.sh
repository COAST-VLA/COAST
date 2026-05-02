#!/bin/bash
# Train per-task TopK SAEs for pi0_fast LIBERO — one SLURM job, sequential over tasks.
# Outputs go to $OPENPI_DATA_HOME/sae_checkpoints/pi0fast_libero/{task}.pt
set -euo pipefail
DRY_RUN=${DRY_RUN:-false}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

LOG_DIR="${REPO_ROOT}/experiments/sae/logs/pi0fast_libero"
SCRIPT_DIR="${REPO_ROOT}/experiments/sae/scripts"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR"

SCRIPT="${SCRIPT_DIR}/train_pi0fast_libero.sh"
cat > "$SCRIPT" << HEADER
#!/bin/bash
#SBATCH --job-name=sae-train-fl
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=${LOG_DIR}/train_%j.out
#SBATCH --error=${LOG_DIR}/train_%j.err

set -e
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
export PYTHONUNBUFFERED=1
cd ${REPO_ROOT}

echo "=== Train SAEs: pi0_fast LIBERO ==="
echo "Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES:-auto}"
echo "Start: \$(date)"

uv run python experiments/sae/src/train_sae.py \\
    --schema pi0fast \\
    --activations-dir \${OPENPI_DATA_HOME}/pi0fast-libero-activations-v1-2000-15env/2000 \\
    --output-dir \${OPENPI_DATA_HOME}/sae_checkpoints/pi0fast_libero \\
    --d-sae-mult 4 --k 64 --n-steps 30000 --batch-size 4096

echo "=== Done: \$(date) ==="
HEADER
chmod +x "$SCRIPT"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] -> $SCRIPT"
else
    sbatch "$SCRIPT"
fi
