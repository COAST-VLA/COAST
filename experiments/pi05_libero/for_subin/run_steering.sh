#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Full LIBERO steering sweep launcher for pi0.5 (one SLURM job per task).
#
# Three-stage pipeline:
#   Stage 1 (CPU):  build_conceptors.py   → $OPENPI_DATA_HOME/libero_conceptors.npz
#   Stage 2 (CPU):  select_parameters.py  → selected_params.json
#   Stage 3 (GPU):  fan out one sbatch per task running conceptor_steering.py.
#
# Typical run (from repo root):
#   bash experiments/pi05_libero/for_subin/run_steering.sh
#   bash experiments/pi05_libero/for_subin/run_steering.sh --dry-run
#   bash experiments/pi05_libero/for_subin/run_steering.sh --skip-build --skip-select
#
# Before first run, EDIT the constants in the "User-editable" section below —
# at minimum, CHECKPOINT_DIR must point at the trained pi0.5 LIBERO checkpoint.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=false
SKIP_BUILD=false
SKIP_SELECT=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)      DRY_RUN=true ;;
        --skip-build)   SKIP_BUILD=true ;;
        --skip-select)  SKIP_SELECT=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ══════════════════════════════════════════════════════════════════════════════
# User-editable
# ══════════════════════════════════════════════════════════════════════════════

# Path to the trained pi0.5 LIBERO checkpoint directory.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/pi05_libero_checkpoint}"
CONFIG="pi05_libero"
NUM_EPISODES=15

# SLURM — adjust for your cluster.
SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
SLURM_GRES="${SLURM_GRES:-gpu:1}"
SLURM_CPUS="${SLURM_CPUS:-12}"
SLURM_MEM="${SLURM_MEM:-96G}"
SLURM_TIME="${SLURM_TIME:-1-00:00:00}"

# Strategies to sweep. Keep all five unless you want to save GPU hours.
STRATEGIES="linear global per_step positive_only random"

# "Generous" sweep axes (you have plenty of GPU — widen freely).
SWEEP_LAYERS="5 11 17"
SWEEP_ALPHAS="0.1 0.5 1.0 2.0 10.0"
SWEEP_BETAS="0.1 0.3"
LINEAR_ALPHAS="0.1 0.5 1.0"

# ══════════════════════════════════════════════════════════════════════════════
# Paths (auto-derived)
# ══════════════════════════════════════════════════════════════════════════════

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FOR_SUBIN_DIR="${REPO_ROOT}/experiments/pi05_libero/for_subin"
PYTHON="${REPO_ROOT}/.venv/bin/python"

OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
CONCEPTOR_NPZ="${OPENPI_DATA_HOME}/libero_conceptors.npz"
PARAM_JSON="${FOR_SUBIN_DIR}/selected_params.json"

OUTPUT_ROOT="experiments/pi05_libero/steering_results"
SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

TASKS=(
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
)
BASE_PORT=8200

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: build conceptors
# ══════════════════════════════════════════════════════════════════════════════
if $SKIP_BUILD && [[ -f "$CONCEPTOR_NPZ" ]]; then
    echo "═══ Stage 1: reusing $CONCEPTOR_NPZ ═══"
else
    echo "═══ Stage 1: Build conceptors ═══"
    $PYTHON "${FOR_SUBIN_DIR}/build_conceptors.py" --output-npz "$CONCEPTOR_NPZ"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: pick narrowed params (advisory — actual sweep still goes wide)
# ══════════════════════════════════════════════════════════════════════════════
if $SKIP_SELECT && [[ -f "$PARAM_JSON" ]]; then
    echo "═══ Stage 2: reusing $PARAM_JSON ═══"
else
    echo "═══ Stage 2: Parameter selection ═══"
    $PYTHON "${FOR_SUBIN_DIR}/select_parameters.py" \
        --conceptor-npz "$CONCEPTOR_NPZ" \
        --output-json "$PARAM_JSON"
fi

echo ""
echo "Sweep: layers=[${SWEEP_LAYERS}] alphas=[${SWEEP_ALPHAS}] betas=[${SWEEP_BETAS}] linear_alphas=[${LINEAR_ALPHAS}]"
echo "Strategies: ${STRATEGIES}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: one sbatch per task
# ══════════════════════════════════════════════════════════════════════════════
echo "═══ Stage 3: Submitting steering jobs ═══"
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))
    SCRIPT="${SCRIPT_DIR}/task_${i}.sh"

    cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
set -e
HEADER
    cat >> "$SCRIPT" << EOF
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Task ${i}: ${TASK:0:60} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
uv run python ${FOR_SUBIN_DIR}/conceptor_steering.py \\
    --task "${TASK}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --conceptor-npz ${CONCEPTOR_NPZ} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --output-dir ${REPO_ROOT}/${OUTPUT_ROOT} \\
    --strategies ${STRATEGIES} \\
    --layers ${SWEEP_LAYERS} \\
    --alphas ${SWEEP_ALPHAS} \\
    --betas ${SWEEP_BETAS} \\
    --linear-alphas ${LINEAR_ALPHAS}
echo "=== Done: ${TASK:0:60} ==="
EOF
    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] task_${i}: ${TASK:0:50}...  (port=${PORT})"
    else
        echo "Submitting task_${i}: ${TASK:0:50}..."
        sbatch \
            --job-name="lib-${i}" \
            --partition="${SLURM_PARTITION}" \
            --gres="${SLURM_GRES}" \
            --cpus-per-task="${SLURM_CPUS}" \
            --mem="${SLURM_MEM}" \
            --time="${SLURM_TIME}" \
            --output="${LOG_DIR}/task_${i}_%j.out" \
            --error="${LOG_DIR}/task_${i}_%j.err" \
            "$SCRIPT"
    fi
done

echo ""
echo "Total jobs: ${#TASKS[@]}"
echo "Results: ${OUTPUT_ROOT}/{task_short_name}/summary.json"
