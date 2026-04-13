#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Conceptor Steering Sweep for pi0.5 LIBERO — grouped by task
# ──────────────────────────────────────────────────────────────────────────────
# One SLURM job per task (10 jobs total).  Each job:
#   1. Loads the model ONCE on 1 GPU
#   2. Sweeps all (layer × alpha × beta × strategy) + baseline + random
#   3. Saves results under  steering_results/{task_short}/
#
# Per-job conditions:
#   1 baseline + 3 layers × 5 alphas × 3 betas × 3 strategies + 3×3 random = 145
#
# Now evaluates ONLY the target task (not all 10) → ~3 min/condition → ~8h total.
#
# Usage:
#   bash experiments/pi05_libero/src/run_steering.sh          # submit all
#   bash experiments/pi05_libero/src/run_steering.sh --dry-run # list without submitting
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

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    TASK_SHORT="${TASK:0:60}"
    PORT=$((BASE_PORT + i))

    SCRIPT="${SCRIPT_DIR}/task_${i}.sh"
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
uv run experiments/pi05_libero/src/conceptor_steering.py \\
    --task "${TASK}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --task-suite-name ${TASK_SUITE} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done: ${TASK_SHORT} ==="
EOF
    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] task_${i}: ${TASK_SHORT}  (port=${PORT})"
    else
        echo "Submitting task_${i}: ${TASK_SHORT}"
        sbatch \
            --job-name="steer-task${i}" \
            --partition=dgx-b200 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=0-15:00:00 \
            --output="${LOG_DIR}/task_${i}_%j.out" \
            --error="${LOG_DIR}/task_${i}_%j.err" \
            "$SCRIPT"
    fi
done

echo ""
echo "Total jobs: ${#TASKS[@]}"
echo "Results will appear in: ${OUTPUT_ROOT}/{task_name}/"
