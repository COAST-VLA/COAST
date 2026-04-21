#!/bin/bash
#SBATCH --job-name=fast-lib-steer
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem=224G
#SBATCH --time=1-12:00:00
#SBATCH --output=experiments/pi0_fast_libero/steering_logs/steer_%A_%a.out
#SBATCH --error=experiments/pi0_fast_libero/steering_logs/steer_%A_%a.err
#SBATCH --array=0-9

set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
cd /vast/projects/ungar/stellar/miaom/openpi-new

# --- B200 / JAX tuning (single GPU) -------------------------------------
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TMPDIR="/vast/projects/ungar/stellar/miaom/.cache/openpi/tmp_xla"
mkdir -p "$TMPDIR"

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

TASK_NAME="${TASKS[${SLURM_ARRAY_TASK_ID}]}"

echo "=== pi0-fast LIBERO steering sweep ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}  Task: ${TASK_NAME}"
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "Start: $(date)"

# Unique port per array task so multiple array jobs on the same node cannot
# collide if they ever land together (and for readability in logs).
PORT=$((8100 + SLURM_ARRAY_TASK_ID))

uv run python experiments/pi0_fast_libero/src/conceptor_steering.py \
    --task "${TASK_NAME}" \
    --port "${PORT}"

echo "=== Done: $(date) ==="
