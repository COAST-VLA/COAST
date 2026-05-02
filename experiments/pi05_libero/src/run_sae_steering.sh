#!/bin/bash
# pi0.5 LIBERO SAE-ActAdd sweep — one slurm job per task.
# Reads v_sae from $OPENPI_DATA_HOME/libero_sae_vectors.npz.
# 10 tasks × 1 layer × 2 αs (mirrors run_linear_only_sweep.sh in pi05_robocasa).
set -euo pipefail
DRY_RUN=${DRY_RUN:-false}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pi05_libero/libero_b200_bs512/2000}"
CONFIG="pi05_libero"
NUM_EPISODES=15

OUTPUT_ROOT="experiments/pi05_libero/sae_steering_results"
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
BASE_PORT=8950
SWEEP_LAYER=11
SAE_ALPHAS="0.25 0.5 1.0 2.0"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))
    SCRIPT="${SCRIPT_DIR}/sae_task_${i}.sh"

    cat > "$SCRIPT" << HEADER
#!/bin/bash
#SBATCH --job-name=ll-sae-${i}
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/sae_task_${i}_%j.out
#SBATCH --error=${LOG_DIR}/sae_task_${i}_%j.err

set -e
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
cd ${REPO_ROOT}

echo "=== pi0.5 LIBERO SAE task ${i}: ${TASK:0:60} ==="
echo "Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES:-auto} | port: ${PORT}"
echo "Start: \$(date)"

uv run experiments/pi05_libero/src/sae_steering.py \\
    --task "${TASK}" \\
    --config ${CONFIG} \\
    --checkpoint-dir ${CHECKPOINT_DIR} \\
    --num-episodes ${NUM_EPISODES} \\
    --port ${PORT} \\
    --output-dir ${REPO_ROOT}/${OUTPUT_ROOT} \\
    --layer ${SWEEP_LAYER} \\
    --alphas ${SAE_ALPHAS}

echo "=== Done: \$(date) ==="
HEADER
    chmod +x "$SCRIPT"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] task_${i}: ${TASK:0:50}...  -> $SCRIPT"
    else
        sbatch "$SCRIPT"
    fi
done
echo ""
echo "${#TASKS[@]} tasks submitted, layer=$SWEEP_LAYER  αs=[$SAE_ALPHAS]"
