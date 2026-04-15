#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Optimized Conceptor Steering Sweep for pi0.5 LIBERO
# ──────────────────────────────────────────────────────────────────────────────
# Two-stage pipeline:
#   Stage 1 (CPU-only):  select_parameters.py reads conceptors, picks
#                        best layer + sweet-spot alphas + safe betas.
#   Stage 2 (GPU):       run the EXISTING conceptor_steering.py with
#                        the narrowed parameter list.
#
# Usage:
#   bash experiments/pi05_libero/src/run_steering_optimized.sh
#   bash experiments/pi05_libero/src/run_steering_optimized.sh --dry-run
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Fixed params ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR="/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_libero/libero_b200_bs512/2000"
CONFIG="pi05_libero"
TASK_SUITE="libero_10"
NUM_EPISODES=15
OUTPUT_ROOT="experiments/pi05_libero/steering_results"
REPO_ROOT="/vast/projects/ungar/stellar/miaom/openpi-new"
PYTHON="${REPO_ROOT}/.venv/bin/python"

CONCEPTOR_NPZ="/vast/projects/ungar/stellar/miaom/.cache/openpi/libero_conceptors.npz"
PARAM_JSON="${REPO_ROOT}/experiments/pi05_libero/selected_params.json"

# ── All 10 LIBERO tasks ──────────────────────────────────────────────────────
TASKS=(
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
)

BASE_PORT=8000

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Parameter Selection (CPU-only, ~2 seconds)
# ══════════════════════════════════════════════════════════════════════════════
echo "═══ Stage 1: Parameter Selection ═══"
$PYTHON "${REPO_ROOT}/experiments/shared/select_parameters.py" \
    --conceptor-npz "$CONCEPTOR_NPZ" \
    --output-json "$PARAM_JSON"

# Parse the JSON to get the selected parameters
BEST_LAYER=$($PYTHON -c "import json; d=json.load(open('$PARAM_JSON')); print(d['best_layer'])")
SEL_ALPHAS=$($PYTHON -c "import json; d=json.load(open('$PARAM_JSON')); print(' '.join(str(a) for a in d['selected_alphas']))")
SEL_BETAS=$($PYTHON -c "import json; d=json.load(open('$PARAM_JSON')); print(' '.join(str(b) for b in d['selected_betas']))")

echo ""
echo "Selected: layer=${BEST_LAYER}, alphas=[${SEL_ALPHAS}], betas=[${SEL_BETAS}]"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Steering Sweep with Narrowed Parameters
# ══════════════════════════════════════════════════════════════════════════════
echo "═══ Stage 2: Submitting Steering Jobs ═══"

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    TASK_SHORT="${TASK:0:60}"
    PORT=$((BASE_PORT + i))

    SCRIPT="${SCRIPT_DIR}/task_opt_${i}.sh"
    cat > "$SCRIPT" << 'SLURM_HEADER'
#!/bin/bash
set -e
SLURM_HEADER
    cat >> "$SCRIPT" << EOF
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Task ${i}: ${TASK_SHORT} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
echo "=== Optimized sweep: layer=${BEST_LAYER}, alphas=[${SEL_ALPHAS}], betas=[${SEL_BETAS}] ==="
uv run experiments/pi05_libero/src/conceptor_steering.py \\
    --task "${TASK}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --task-suite-name ${TASK_SUITE} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --output-dir ${OUTPUT_ROOT} \\
    --layers ${BEST_LAYER} \\
    --alphas ${SEL_ALPHAS} \\
    --betas ${SEL_BETAS}
echo "=== Done: ${TASK_SHORT} ==="
EOF
    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] task_${i}: ${TASK_SHORT}  (port=${PORT})"
    else
        echo "Submitting task_${i}: ${TASK_SHORT}"
        sbatch \
            --job-name="steer-opt-${i}" \
            --partition=dgx-b200 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=0-06:00:00 \
            --output="${LOG_DIR}/task_opt_${i}_%j.out" \
            --error="${LOG_DIR}/task_opt_${i}_%j.err" \
            "$SCRIPT"
    fi
done

echo ""
echo "Total jobs: ${#TASKS[@]}"
echo "Results will appear in: ${OUTPUT_ROOT}/{task_name}/"
