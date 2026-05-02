#!/bin/bash
# Train per-task TopK SAEs for pi0_fast MetaWorld — one SLURM job, sequential.
# 10 tasks (subset that matches the paper's MetaWorld table).
# At ~5–8 min/task that's ~1–2 hours wall-clock.
set -euo pipefail
DRY_RUN=${DRY_RUN:-false}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

LOG_DIR="${REPO_ROOT}/experiments/sae/logs/pi0fast_metaworld"
SCRIPT_DIR="${REPO_ROOT}/experiments/sae/scripts"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR"

# Paper-table task subset (10 tasks).
TASKS=(
    "coffee-push-v3" "push-v3" "pick-place-v3" "plate-slide-back-v3"
    "faucet-close-v3" "pick-place-wall-v3" "reach-v3" "coffee-pull-v3"
    "disassemble-v3" "stick-push-v3"
)
TASKS_STR="${TASKS[@]}"

SCRIPT="${SCRIPT_DIR}/train_pi0fast_metaworld.sh"
cat > "$SCRIPT" << HEADER
#!/bin/bash
#SBATCH --job-name=sae-train-fm
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

echo "=== Train SAEs: pi0_fast MetaWorld (10 paper-table tasks) ==="
echo "Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES:-auto}"
echo "Start: \$(date)"

uv run python experiments/sae/src/train_sae.py \\
    --schema pi0fast \\
    --activations-dir \${OPENPI_DATA_HOME}/pi0fast-metaworld-activations-v1-ml45train-16env/2500 \\
    --output-dir \${OPENPI_DATA_HOME}/sae_checkpoints/pi0fast_metaworld \\
    --tasks ${TASKS_STR} \\
    --d-sae-mult 4 --k 64 --n-steps 30000 --batch-size 4096

echo "=== Done: \$(date) ==="
HEADER
chmod +x "$SCRIPT"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] -> $SCRIPT"
else
    sbatch "$SCRIPT"
fi
