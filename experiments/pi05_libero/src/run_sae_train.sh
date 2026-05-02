#!/bin/bash
# Train per-task TopK SAEs for pi0.5 LIBERO — one SLURM job, sequential.
# 10 tasks × 1 layer (L=11, matches SWEEP_LAYER convention) = 10 SAEs.
# Per-task data is moderate (~110k vectors / task) → ~5–10 min/SAE → ~1.5h wall.
set -euo pipefail
DRY_RUN=${DRY_RUN:-false}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

LOG_DIR="${REPO_ROOT}/experiments/sae/logs/pi05_libero"
SCRIPT_DIR="${REPO_ROOT}/experiments/sae/scripts"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR"

SCRIPT="${SCRIPT_DIR}/train_pi05_libero.sh"
cat > "$SCRIPT" << HEADER
#!/bin/bash
#SBATCH --job-name=sae-train-ll
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

echo "=== Train SAEs: pi0.5 LIBERO (10 tasks × 1 layer = 10 SAEs) ==="
echo "Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES:-auto}"
echo "Start: \$(date)"

uv run python experiments/sae/src/train_sae.py \\
    --schema pi05 \\
    --activations-dir \${OPENPI_DATA_HOME}/activations/pi05_libero_2000_15env/openpi-libero-2000 \\
    --output-dir \${OPENPI_DATA_HOME}/sae_checkpoints/pi05_libero \\
    --layers 11 \\
    --d-sae-mult 4 --k 64 --n-steps 30000 --batch-size 4096

echo "=== Done: \$(date) ==="
HEADER
chmod +x "$SCRIPT"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] -> $SCRIPT"
else
    sbatch "$SCRIPT"
fi
