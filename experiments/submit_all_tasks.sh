#!/bin/bash
# Submit conceptor steering experiments for all mixed-outcome ML45 tasks.
# Submits all jobs at once (no dependency chaining) with reduced wall time
# to fit before the cluster maintenance window.
#
# Usage: bash experiments/submit_all_tasks.sh
#        bash experiments/submit_all_tasks.sh --dry-run   # print without submitting

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Will print commands without submitting"
fi

cd /nlpgpu/data/miaom/openpi-metaworld
mkdir -p experiments/steering_results

# ── 26 mixed-outcome tasks (have both success and failure episodes) ──
TASKS=(
    assembly-v3
    basketball-v3
    coffee-pull-v3
    coffee-push-v3
    disassemble-v3
    door-open-v3
    faucet-close-v3
    hammer-v3
    handle-pull-side-v3
    handle-pull-v3
    lever-pull-v3
    peg-insert-side-v3
    pick-out-of-hole-v3
    pick-place-v3
    pick-place-wall-v3
    plate-slide-back-side-v3
    plate-slide-back-v3
    push-back-v3
    push-v3
    reach-v3
    shelf-place-v3
    soccer-v3
    stick-pull-v3
    stick-push-v3
    sweep-into-v3
    sweep-v3
)

TOTAL=${#TASKS[@]}
echo "Total tasks: $TOTAL"
echo ""

# Common experiment args
COMMON_ARGS="--alphas 0.1 0.5 1.0 --betas 0.1 0.3 0.5 --steering-layer 11 --linear-alphas 0.5 1.0 2.0 5.0 --num-envs 15 --max-steps 300 --replan-steps 10"

submit_job() {
    local TASK=$1

    if $DRY_RUN; then
        echo "[DRY RUN] Would submit: steer-${TASK}"
        return
    fi

    local TMPFILE=$(mktemp /tmp/steer_XXXXXX.sh)
    cat > "$TMPFILE" <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=steer-${TASK}
#SBATCH --partition=p_nlp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --constraint=48GBgpu
#SBATCH --output=experiments/steering_results/slurm_%j_${TASK}.out
#SBATCH --error=experiments/steering_results/slurm_%j_${TASK}.err

set -e

PYTHON=/nlpgpu/data/miaom/openpi-metaworld/.venv/bin/python
export MUJOCO_GL=osmesa
export HF_HOME=/nlp/data/huggingface_cache
export TORCH_COMPILE_DISABLE=1

cd /nlpgpu/data/miaom/openpi-metaworld

\$PYTHON experiments/conceptor_steering.py --policy.config=pi05_metaworld --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ --tasks ${TASK} ${COMMON_ARGS} --output-dir experiments/steering_results
SBATCH_EOF

    local JOB_ID
    JOB_ID=$(sbatch "$TMPFILE" | awk '{print $4}')
    rm -f "$TMPFILE"
    echo "  Submitted $TASK -> Job $JOB_ID"
}

# Submit all jobs at once (no dependency chaining)
echo "=== Submitting all $TOTAL tasks ==="
for TASK in "${TASKS[@]}"; do
    submit_job "$TASK"
done

echo ""
echo "All $TOTAL jobs submitted! Use 'squeue -u \$USER' to monitor."
