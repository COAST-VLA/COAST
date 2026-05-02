#!/bin/bash
# Run fit_sae_vectors.py on the trained pi0_fast MetaWorld SAEs (paper-table 10 tasks).
set -euo pipefail
DRY_RUN=${DRY_RUN:-false}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

LOG_DIR="${REPO_ROOT}/experiments/sae/logs/pi0fast_metaworld"
SCRIPT_DIR="${REPO_ROOT}/experiments/sae/scripts"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR"

SCRIPT="${SCRIPT_DIR}/fit_pi0fast_metaworld.sh"
cat > "$SCRIPT" << HEADER
#!/bin/bash
#SBATCH --job-name=sae-fit-fm
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/fit_%j.out
#SBATCH --error=${LOG_DIR}/fit_%j.err

set -e
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
export PYTHONUNBUFFERED=1
cd ${REPO_ROOT}

echo "=== Fit SAE vectors: pi0_fast MetaWorld ==="
echo "Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES:-auto}"
echo "Start: \$(date)"

uv run python experiments/sae/src/fit_sae_vectors.py \\
    --schema pi0fast \\
    --activations-dir \${OPENPI_DATA_HOME}/pi0fast-metaworld-activations-v1-ml45train-16env/2500 \\
    --sae-dir \${OPENPI_DATA_HOME}/sae_checkpoints/pi0fast_metaworld \\
    --output-npz \${OPENPI_DATA_HOME}/pi0fast_metaworld_sae_vectors.npz

echo "=== Done: \$(date) ==="
HEADER
chmod +x "$SCRIPT"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] -> $SCRIPT"
else
    sbatch "$SCRIPT"
fi
