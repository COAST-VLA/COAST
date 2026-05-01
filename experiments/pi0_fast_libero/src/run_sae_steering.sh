#!/bin/bash
# Bulk pi0_fast LIBERO SAE-ActAdd sweep — one slurm job per task.
# Mirrors run_linear_final_sweep.sh; only difference is the steering-vector source.
set -euo pipefail
DRY_RUN=${DRY_RUN:-false}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

OUTPUT_ROOT="experiments/pi0_fast_libero/sae_steering_results"
LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/logs"
SCRIPT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/scripts"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR"

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
BASE_PORT=8700
SAE_ALPHAS="0.5 1.0"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))
    SCRIPT="${SCRIPT_DIR}/sae_task_${i}.sh"

    cat > "$SCRIPT" << HEADER
#!/bin/bash
#SBATCH --job-name=fl-sae-${i}
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem=224G
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/sae_task_${i}_%j.out
#SBATCH --error=${LOG_DIR}/sae_task_${i}_%j.err

set -e
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
export MUJOCO_GL=egl
cd ${REPO_ROOT}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TMPDIR="\${OPENPI_DATA_HOME}/tmp_xla"
mkdir -p "\$TMPDIR"

echo "=== pi0-fast LIBERO SAE task ${i}: ${TASK:0:60} ==="
echo "Node: \$(hostname) | port: ${PORT}"
echo "Start: \$(date)"

uv run python experiments/pi0_fast_libero/src/sae_steering.py \\
    --task "${TASK}" \\
    --checkpoint-dir checkpoints/pi0_fast_libero/pi0_fast_libero_b200_bs512/2000/ \\
    --port ${PORT} \\
    --alphas ${SAE_ALPHAS} \\
    --num-episodes 15 \\
    --output-dir ${OUTPUT_ROOT}

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
echo "${#TASKS[@]} tasks submitted, αs=[$SAE_ALPHAS]"
