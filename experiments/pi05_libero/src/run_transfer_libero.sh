#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Transfer experiment sweep for pi0.5 LIBERO (Design B)
#
# One SLURM job per target task. Each job loads the model once, starts the
# WebSocket server, and iterates all 9 source tasks × (global + 2 per-step).
#
# Usage:
#   bash experiments/pi05_libero/src/run_transfer_libero.sh
#   bash experiments/pi05_libero/src/run_transfer_libero.sh --dry-run
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

CHECKPOINT_DIR="/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_libero/libero_b200_bs512/2000"
CONFIG="pi05_libero"
TASK_SUITE="libero_10"
NUM_EPISODES=15
OUTPUT_ROOT="experiments/pi05_libero/transfer_results"
RESULTS_ROOT="experiments/pi05_libero/steering_results"
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

# Distinct port range from the 8000-series steering sweep.
BASE_PORT=8100

SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

echo "═══ Submitting transfer jobs (${#TASKS[@]} targets) ═══"

for i in "${!TASKS[@]}"; do
    TARGET="${TASKS[$i]}"
    TARGET_SHORT="${TARGET:0:60}"
    PORT=$((BASE_PORT + i))

    SCRIPT="${SCRIPT_DIR}/transfer_${i}.sh"
    cat > "$SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}
echo "=== Transfer job ${i}: target=${TARGET_SHORT} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} ==="
uv run experiments/pi05_libero/src/transfer_steering.py \\
    --target-task "${TARGET}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --task-suite-name ${TASK_SUITE} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --results-root ${RESULTS_ROOT} \\
    --output-dir ${OUTPUT_ROOT}
echo "=== Done: target=${TARGET_SHORT} ==="
EOF
    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] transfer_${i}: target=${TARGET_SHORT} (port=${PORT})"
    else
        echo "Submitting transfer_${i}: target=${TARGET_SHORT}"
        sbatch \
            --job-name="transfer-${i}" \
            --partition=dgx-b200 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=0-08:00:00 \
            --output="${LOG_DIR}/transfer_${i}_%j.out" \
            --error="${LOG_DIR}/transfer_${i}_%j.err" \
            "$SCRIPT"
    fi
done

echo ""
echo "Total target jobs: ${#TASKS[@]}"
echo "Results will appear in: ${OUTPUT_ROOT}/target_{task_name}/"
