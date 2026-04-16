#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Optimized Conceptor + Linear Steering Sweep for GR00T N1.5 RoboCasa
# ──────────────────────────────────────────────────────────────────────────────
# Two-stage pipeline (mirrors pi05_robocasa/src/run_steering_optimized.sh):
#   Stage 1 (CPU-only):  select_parameters.py reads conceptors, picks
#                        best layer + sweet-spot alphas + safe betas.
#   Stage 2 (GPU):       fan out one SLURM job per mixed-outcome task
#                        running conceptor_steering.py with the narrowed
#                        parameter list. Each job `cd`s into groot_env
#                        (Python 3.10) — GR00T N1.5 server runs in-process.
#
# Usage:
#   bash experiments/groot_robocasa/run_steering_optimized.sh
#   bash experiments/groot_robocasa/run_steering_optimized.sh --dry-run
#   bash experiments/groot_robocasa/run_steering_optimized.sh --skip-select   # reuse existing JSON
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=false
SKIP_SELECT=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --skip-select) SKIP_SELECT=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── Repo / paths ──────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_ENV_DIR="${REPO_ROOT}/groot_env"
GROOT_PYTHON="${GROOT_ENV_DIR}/.venv/bin/python"
SHARED_PYTHON="${GROOT_PYTHON}"   # numpy is available in groot_env

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
CONCEPTOR_NPZ="${OPENPI_DATA_HOME}/groot_n15_robocasa_conceptors.npz"

OUTPUT_ROOT="experiments/groot_robocasa/steering_results"
PARAM_JSON="${REPO_ROOT}/experiments/groot_robocasa/selected_params/selected_params.json"

# ── Fixed run params ──────────────────────────────────────────────────────────
# Path is relative to groot_env/ (where the job will cd into).
CHECKPOINT_DIR="../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000"
NUM_EPISODES=15

# All 7 RoboCasa tasks. Tasks without valid contrastive conceptors in the npz
# (currently PickPlaceCounterToCabinet, PickPlaceCounterToStove,
# TurnOnElectricKettle — need activation re-collection) are auto-skipped below.
TASKS=(
    "PickPlaceCounterToCabinet"
    "OpenDrawer"
    "OpenStandMixerHead"
    "CloseFridge"
    "PickPlaceCounterToStove"
    "TurnOnElectricKettle"
    "CoffeeSetupMug"
)

BASE_PORT=8300   # offset from pi05 (8200) to avoid collisions on shared nodes

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Parameter Selection (CPU-only, ~seconds)
# ══════════════════════════════════════════════════════════════════════════════
if $SKIP_SELECT && [[ -f "$PARAM_JSON" ]]; then
    echo "═══ Stage 1: Reusing existing ${PARAM_JSON} ═══"
else
    echo "═══ Stage 1: Parameter Selection ═══"
    mkdir -p "$(dirname "$PARAM_JSON")"
    "$SHARED_PYTHON" "${REPO_ROOT}/experiments/shared/select_parameters.py" \
        --conceptor-npz "$CONCEPTOR_NPZ" \
        --output-json "$PARAM_JSON" \
        --tasks "${TASKS[@]}"
fi

# Expanded sweep: selector picked L10 / α=0.1 / β=[0.1, 0.3], but GR00T layer
# quotas are almost flat (selector had no strong signal) and α=0.1 was a
# fallback — no α landed in the pi05-calibrated overlap sweet-spot band.
# Widen to L={10, 13} × α={0.1, 0.5} to hedge against the selector missing
# the real optimum. Betas come straight from the JSON.
SEL_LAYERS="10 13"
SEL_ALPHAS_EXPANDED="0.1 0.5"
SEL_BETAS=$("$SHARED_PYTHON" -c "import json; d=json.load(open('$PARAM_JSON')); print(' '.join(str(b) for b in d['selected_betas']))")

echo ""
echo "Sweep (expanded): layers=[${SEL_LAYERS}], alphas=[${SEL_ALPHAS_EXPANDED}], betas=[${SEL_BETAS}]"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Steering Sweep with Narrowed Parameters
# ══════════════════════════════════════════════════════════════════════════════
echo "═══ Stage 2: Submitting Steering Jobs ═══"

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

# Detect which tasks actually have contrastive conceptors in the npz.
HAS_CONCEPTOR_PY='
import json, sys, numpy as np
npz = np.load(sys.argv[1], allow_pickle=False)
keys = set(npz.files)
layer = int(sys.argv[2])
alpha = sys.argv[3]
for t in sys.argv[4:]:
    k = f"{t}__L{layer}__{alpha}__C_contrastive"
    print(f"{t}\t{int(k in keys)}")
'
declare -a AVAILABLE_TASKS=()
declare -a SKIPPED_TASKS=()
SAMPLE_LAYER=$(echo "$SEL_LAYERS" | awk "{print \$1}")
SAMPLE_ALPHA=$(echo "$SEL_ALPHAS_EXPANDED" | awk "{print \$1}")
while IFS=$'\t' read -r t ok; do
    if [[ "$ok" == "1" ]]; then
        AVAILABLE_TASKS+=("$t")
    else
        SKIPPED_TASKS+=("$t")
    fi
done < <("$SHARED_PYTHON" -c "$HAS_CONCEPTOR_PY" "$CONCEPTOR_NPZ" "$SAMPLE_LAYER" "$SAMPLE_ALPHA" "${TASKS[@]}")

if [[ ${#SKIPPED_TASKS[@]} -gt 0 ]]; then
    echo "Skipping (no contrastive conceptor in npz — re-collect activations): ${SKIPPED_TASKS[*]}"
fi
echo "Will submit: ${AVAILABLE_TASKS[*]}"
echo ""

for i in "${!AVAILABLE_TASKS[@]}"; do
    TASK="${AVAILABLE_TASKS[$i]}"
    PORT=$((BASE_PORT + i))

    SCRIPT="${SCRIPT_DIR}/task_opt_${i}.sh"
    cat > "$SCRIPT" << 'SLURM_HEADER'
#!/bin/bash
set -e
SLURM_HEADER
    cat >> "$SCRIPT" << EOF
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${GROOT_ENV_DIR}
echo "=== Task ${i}: ${TASK} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
echo "=== Expanded sweep: layers=[${SEL_LAYERS}], alphas=[${SEL_ALPHAS_EXPANDED}], betas=[${SEL_BETAS}] ==="
uv run python ${REPO_ROOT}/experiments/groot_robocasa/src/conceptor_steering.py \\
    --task "${TASK}" \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --conceptor-npz ${CONCEPTOR_NPZ} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --output-dir ${REPO_ROOT}/${OUTPUT_ROOT} \\
    --layers ${SEL_LAYERS} \\
    --alphas ${SEL_ALPHAS_EXPANDED} \\
    --betas ${SEL_BETAS}
echo "=== Done: ${TASK} ==="
EOF
    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] task_${i}: ${TASK}  (port=${PORT})  -> ${SCRIPT}"
    else
        echo "Submitting task_${i}: ${TASK}"
        sbatch \
            --job-name="grc-opt-${i}" \
            --partition=dgx-b200 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=1-00:00:00 \
            --output="${LOG_DIR}/task_opt_${i}_%j.out" \
            --error="${LOG_DIR}/task_opt_${i}_%j.err" \
            "$SCRIPT"
    fi
done

echo ""
echo "Total jobs submitted: ${#AVAILABLE_TASKS[@]} (of ${#TASKS[@]} tasks)"
echo "Results: ${OUTPUT_ROOT}/{task_name}/summary.json"
