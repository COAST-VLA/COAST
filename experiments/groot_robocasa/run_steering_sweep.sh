#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Full Conceptor Steering Sweep for GR00T N1.5 RoboCasa (all 7 tasks)
# ──────────────────────────────────────────────────────────────────────────────
# One SLURM job per task. Each job runs conceptor_steering.py with:
#   Strategies: global, per_step, positive_only, linear, random
#   - global/per_step/positive_only: full sweep (layer × alpha × beta)
#   - linear/random: limited sweep (baselines)
#
# Usage:
#   bash experiments/groot_robocasa/run_steering_sweep.sh
#   bash experiments/groot_robocasa/run_steering_sweep.sh --dry-run
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Repo / paths ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_ENV_DIR="${REPO_ROOT}/groot_env"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
CONCEPTOR_NPZ="${OPENPI_DATA_HOME}/groot_n15_robocasa_conceptors.npz"

OUTPUT_ROOT="${REPO_ROOT}/experiments/groot_robocasa/steering_results"
CHECKPOINT_DIR="../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000"
NUM_EPISODES=15

# ── Sweep parameters ─────────────────────────────────────────────────────────
# Reduced grid: 28 conditions/task (from 138) to conserve GPU hours.
ALPHAS="0.1 0.5 1.0"
BETAS="0.1 0.3"
# Layer 10 selected by select_parameters.py (highest mean quota).
LAYERS="10"
# Linear baselines — limited sweep
LINEAR_ALPHAS="0.5 1.0"
# Random controls — limited
N_RANDOM=2

TASKS=(
    "CloseFridge"
    "CoffeeSetupMug"
    "OpenDrawer"
    "OpenStandMixerHead"
    "PickPlaceCounterToCabinet"
    "PickPlaceCounterToStove"
    "TurnOnElectricKettle"
)

BASE_PORT=8300

# ── Directories ──────────────────────────────────────────────────────────────
SCRIPT_DIR="${OUTPUT_ROOT}/scripts"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

echo "═══ GR00T N1.5 RoboCasa Steering Sweep ═══"
echo "Tasks:   ${#TASKS[@]}"
echo "Layers:  [${LAYERS}]"
echo "Alphas:  [${ALPHAS}]"
echo "Betas:   [${BETAS}]"
echo "Linear:  [${LINEAR_ALPHAS}]"
echo "Random:  ${N_RANDOM} controls/task"
echo "Strategies: global, per_step, positive_only, linear (limited), random (limited)"
echo ""

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))

    SCRIPT="${SCRIPT_DIR}/steer_${TASK}.sh"
    cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
set -e
HEADER
    cat >> "$SCRIPT" << EOF
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export UV_LINK_MODE=copy

# ── Fix torch for B200 (sm_100) in both venvs ──
# gr00t[base] pins torch==2.5.1+cu124 which lacks sm_100 support.
# uv run would re-sync and revert the upgrade, so we use the venv python
# directly after force-installing torch 2.7.0+cu128.
GROOT_PYTHON="${GROOT_ENV_DIR}/.venv/bin/python"
ROBOCASA_PYTHON="${REPO_ROOT}/examples/robocasa_env/.venv/bin/python"

echo "=== Steering ${TASK} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} | Port: ${PORT} ==="

# Verify torch 2.7.0+cu128 (pre-installed from login node) and CUDA work
\${GROOT_PYTHON} -c "import torch; assert 'cu128' in torch.__version__, f'wrong torch: {torch.__version__}'; assert torch.cuda.is_available(), 'no CUDA'; print(f'groot_env torch={torch.__version__} arch={torch.cuda.get_arch_list()}')"
\${ROBOCASA_PYTHON} -c "import torch; assert 'cu128' in torch.__version__, f'wrong torch: {torch.__version__}'; assert torch.cuda.is_available(), 'no CUDA'; print(f'robocasa_env torch={torch.__version__} arch={torch.cuda.get_arch_list()}')"

cd ${GROOT_ENV_DIR}

# ── Pass 1: Main experiment (global) — full alpha/beta sweep ──
echo "=== Pass 1: global (full sweep) ==="
\${GROOT_PYTHON} ${REPO_ROOT}/experiments/groot_robocasa/src/conceptor_steering.py \\
    --task ${TASK} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --conceptor-npz ${CONCEPTOR_NPZ} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --output-dir ${OUTPUT_ROOT} \\
    --layers ${LAYERS} \\
    --alphas ${ALPHAS} \\
    --betas ${BETAS} \\
    --strategies global \\
    --n-random-controls 0

# ── Pass 1b: per_step — sweep betas (alpha baked in from build_conceptors) ──
echo "=== Pass 1b: per_step (beta sweep) ==="
\${GROOT_PYTHON} ${REPO_ROOT}/experiments/groot_robocasa/src/conceptor_steering.py \\
    --task ${TASK} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --conceptor-npz ${CONCEPTOR_NPZ} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --output-dir ${OUTPUT_ROOT} \\
    --layers ${LAYERS} \\
    --betas 0.1 0.2 0.3 0.5 \\
    --strategies per_step \\
    --n-random-controls 0

# ── Pass 2: Baselines (positive_only, linear, random) — minimal sweep ──
echo "=== Pass 2: positive_only + linear + random (limited) ==="
\${GROOT_PYTHON} ${REPO_ROOT}/experiments/groot_robocasa/src/conceptor_steering.py \\
    --task ${TASK} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --conceptor-npz ${CONCEPTOR_NPZ} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --output-dir ${OUTPUT_ROOT} \\
    --layers ${LAYERS} \\
    --alphas 1.0 \\
    --betas 0.1 \\
    --linear-alphas 1.0 \\
    --strategies positive_only linear \\
    --n-random-controls 1
echo "=== Done: ${TASK} ==="
EOF
    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] ${TASK} -> ${SCRIPT} (port=${PORT})"
    else
        JID=$(sbatch \
            --parsable \
            --job-name="grc-steer-${TASK}" \
            --partition=dgx-b200 \
            --exclude=dgx016 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=2-00:00:00 \
            --output="${LOG_DIR}/steer_${TASK}_%j.out" \
            --error="${LOG_DIR}/steer_${TASK}_%j.err" \
            "$SCRIPT")
        echo "Submitted ${TASK} as job ${JID} (port=${PORT})"
    fi
done

echo ""
echo "Total: ${#TASKS[@]} jobs"
echo "Results: ${OUTPUT_ROOT}/{task_name}/summary.json"
