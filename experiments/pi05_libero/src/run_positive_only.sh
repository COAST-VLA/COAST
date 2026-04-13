#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Positive-Only Conceptor Steering — 10 LIBERO tasks
# ──────────────────────────────────────────────────────────────────────────────
# Minimal sweep: 2 layers × 2 alphas × 2 betas = 8 conditions per task
# ~3 min/condition → ~25 min per task
#
# Usage:
#   bash experiments/pi05_libero/src/run_positive_only.sh          # submit all
#   bash experiments/pi05_libero/src/run_positive_only.sh --dry-run # list only
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

CHECKPOINT_DIR="/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_libero/libero_b200_bs512/2000"
CONFIG="pi05_libero"
TASK_SUITE="libero_10"
NUM_EPISODES=15
OUTPUT_ROOT="experiments/pi05_libero/steering_results"
REPO_ROOT="/vast/projects/ungar/stellar/miaom/openpi-new"

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

BASE_PORT=8100  # offset from main steering jobs to avoid port conflicts

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    TASK_SHORT="${TASK:0:60}"
    PORT=$((BASE_PORT + i))

    SCRIPT="${SCRIPT_DIR}/pos_only_task_${i}.sh"
    cat > "$SCRIPT" << 'SLURM_HEADER'
#!/bin/bash
set -e
SLURM_HEADER
    cat >> "$SCRIPT" << EOF
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Pos-Only Task ${i}: ${TASK_SHORT} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_libero/src/positive_only_steering.py \\
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
        echo "[DRY-RUN] pos_only_task_${i}: ${TASK_SHORT}  (port=${PORT})"
    else
        echo "Submitting pos_only_task_${i}: ${TASK_SHORT}"
        sbatch \
            --job-name="pos-task${i}" \
            --partition=dgx-b200 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=0-02:00:00 \
            --output="${LOG_DIR}/pos_only_task_${i}_%j.out" \
            --error="${LOG_DIR}/pos_only_task_${i}_%j.err" \
            "$SCRIPT"
    fi
done

echo ""
echo "Total jobs: ${#TASKS[@]}"
echo "Conditions per task: 8 (2 layers × 2 alphas × 2 betas)"
echo "Results will be appended to existing summary.json files"
