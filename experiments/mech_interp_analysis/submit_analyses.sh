#!/bin/bash
# Submit mechanistic interpretability analyses to SLURM
# Runs all 5 analyses + master figure assembly on a single GPU node

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/nlpgpu/data/miaom/openpi-metaworld"
OUT_DIR="${SCRIPT_DIR}/results"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

mkdir -p "${OUT_DIR}"

cat > /tmp/mech_interp_job.sh << 'SLURM_SCRIPT'
#!/bin/bash
#SBATCH --job-name=mech-interp
#SBATCH --partition=p_nlp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
SLURM_SCRIPT

# Append dynamic paths (not inside heredoc to allow variable expansion)
cat >> /tmp/mech_interp_job.sh << EOF
#SBATCH --output=${OUT_DIR}/slurm_%j.out
#SBATCH --error=${OUT_DIR}/slurm_%j.err

set -e

export HF_HOME=/nlp/data/huggingface_cache
export MUJOCO_GL=osmesa
export TORCH_COMPILE_DISABLE=1

cd ${PROJECT_DIR}

echo "=== Starting mechanistic interpretability analyses ==="
echo "Node: \$(hostname)"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: \$(date)"
echo ""

${PYTHON} experiments/mech_interp_analysis/run_all_analyses.py

echo ""
echo "=== All analyses complete ==="
echo "Results in: ${OUT_DIR}"
ls -lh ${OUT_DIR}/*.pdf ${OUT_DIR}/*.png ${OUT_DIR}/*.json 2>/dev/null || true
EOF

echo "Submitting mechanistic interpretability job..."
JOB_ID=$(sbatch /tmp/mech_interp_job.sh | awk '{print $4}')
echo "Submitted job ${JOB_ID}"
echo "  Logs: ${OUT_DIR}/slurm_${JOB_ID}.out"
echo "  Errors: ${OUT_DIR}/slurm_${JOB_ID}.err"
echo ""
echo "Monitor with: tail -f ${OUT_DIR}/slurm_${JOB_ID}.err"
