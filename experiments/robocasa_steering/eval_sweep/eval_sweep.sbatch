#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Per-checkpoint eval + activation-collection sweep.
#
# Runs all 18 `atomic_seen` RoboCasa tasks for ONE checkpoint, with
#   --num_rollouts 30  --num_envs 15
# and the activation collector (so the same rollouts produce both success-rate
# data AND the suffix_residual / denoising / adarms_cond
# tensors used downstream by experiments/robocasa_steering/build_conceptors.py).
#
# Parameterised by env vars (set by submit_sweep.sh):
#   CHECKPOINT         Path to a .ckpt file.
#   ACTIVATIONS_ROOT   Root where activations land — the collector writes to
#                      $ACTIVATIONS_ROOT/<ckpt_stem>/<task>/episode_NNN_env_NNN/...
#   SPLIT              RoboCasa split (default: pretrain).
#
# Derived per-job outputs:
#   $ACTIVATIONS_ROOT/<ckpt_stem>/<task>/...          (one subtree per task)
#   $ACTIVATIONS_ROOT/<ckpt_stem>/success_rates.json  (written at end of job)
#   $ACTIVATIONS_ROOT/logs/<jobid>_<ckpt_stem>.out    (SLURM stdout/stderr)
#
# CPU / memory sizing for num_envs=15:
#   15 MuJoCo-EGL worker subprocesses + 1 main process → 20 CPU budget.
#   ~1.5 GB/worker MuJoCo VRAM buffers + policy ≈ 25-35 GB main RAM.
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=dp_robocasa_eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --time=23:00:00
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high

set -euo pipefail

# ── Config (inputs) ──────────────────────────────────────────────────────────
: "${CHECKPOINT:?need CHECKPOINT env var (path to .ckpt)}"
: "${ACTIVATIONS_ROOT:?need ACTIVATIONS_ROOT env var}"
SPLIT="${SPLIT:-pretrain}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-30}"
NUM_ENVS="${NUM_ENVS:-15}"

# ── Derived ─────────────────────────────────────────────────────────────────
: "${REPO_ROOT:?need REPO_ROOT env var (exported by submit_sweep.sh)}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
CKPT_STEM="$(basename "$CHECKPOINT" .ckpt)"
PER_CKPT_DIR="${ACTIVATIONS_ROOT}/${CKPT_STEM}"
mkdir -p "$PER_CKPT_DIR"

# ── Active task subset (6 of the 18 atomic_seen tasks) ─────────────────────
# To restore the full 18-task atomic_seen set, comment out this block and
# uncomment the block below.
TASKS=(
    "CoffeeSetupMug"            # "Coffee Mug"
    "OpenDrawer"                # "Open Drawer"
    "OpenStandMixerHead"        # "Stand Mixer"
    "PickPlaceCounterToCabinet" # "PP Cabinet"
    "PickPlaceCounterToStove"   # "PP Stove"
    "TurnOnElectricKettle"      # "Kettle"
)

# # ── Full 18 atomic_seen tasks (open-sourced DP policy's eval set) ──────────
# TASKS=(
#     "CloseBlenderLid"
#     "CloseFridge"
#     "CloseToasterOvenDoor"
#     "CoffeeSetupMug"
#     "NavigateKitchen"
#     "OpenCabinet"
#     "OpenDrawer"
#     "OpenStandMixerHead"
#     "PickPlaceCounterToCabinet"
#     "PickPlaceCounterToStove"
#     "PickPlaceDrawerToCounter"
#     "PickPlaceSinkToCounter"
#     "PickPlaceToasterToCounter"
#     "SlideDishwasherRack"
#     "TurnOffStove"
#     "TurnOnElectricKettle"
#     "TurnOnMicrowave"
#     "TurnOnSinkFaucet"
# )

# Optional: if TASKS_OVERRIDE is set, replace the default TASKS array with
# a comma-separated list (e.g. TASKS_OVERRIDE=CloseFridge,OpenDrawer).
TASKS_OVERRIDE="${TASKS_OVERRIDE:-}"
if [ -n "$TASKS_OVERRIDE" ]; then
    IFS=',' read -r -a TASKS <<< "$TASKS_OVERRIDE"
fi

# Optional: skip tasks before START_TASK (e.g. to resume a partial sweep).
START_TASK="${START_TASK:-}"
if [ -n "$START_TASK" ]; then
    start_idx=-1
    for i in "${!TASKS[@]}"; do
        if [ "${TASKS[$i]}" = "$START_TASK" ]; then start_idx=$i; break; fi
    done
    if [ $start_idx -lt 0 ]; then
        echo "ERROR: START_TASK='$START_TASK' not in TASKS array" >&2
        exit 2
    fi
    TASKS=( "${TASKS[@]:$start_idx}" )
fi

echo "======================================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "HOST:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CHECKPOINT:   $CHECKPOINT"
echo "CKPT_STEM:    $CKPT_STEM"
echo "SPLIT:        $SPLIT"
echo "NUM_ROLLOUTS: $NUM_ROLLOUTS   NUM_ENVS: $NUM_ENVS"
echo "OUTPUT:       $PER_CKPT_DIR"
echo "START_TASK:   ${START_TASK:-<none, from task 1>}"
echo "TASKS (${#TASKS[@]}):"
printf '  - %s\n' "${TASKS[@]}"
echo "======================================================================"

export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$REPO_ROOT"

# ── Run the 18 tasks sequentially (envs fan out inside each task) ───────────
for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    idx=$((i + 1))
    echo ""
    echo "[${idx}/${#TASKS[@]}] $TASK  @ $(date -Is)"

    "$PYTHON" collect_activations_robocasa.py \
        --checkpoint "$CHECKPOINT" \
        --activations_output_dir "$ACTIVATIONS_ROOT" \
        --task "$TASK" \
        --split "$SPLIT" \
        --num_rollouts "$NUM_ROLLOUTS" \
        --num_envs "$NUM_ENVS" \
        --device cuda:0

    echo "[${idx}/${#TASKS[@]}] $TASK  done @ $(date -Is)"
done

# ── Compute per-checkpoint success rates ────────────────────────────────────
echo ""
echo "Computing per-task success rates for $CKPT_STEM ..."
"$PYTHON" "$REPO_ROOT/experiments/robocasa_steering/eval_sweep/aggregate_results.py" \
    per-checkpoint \
    --checkpoint-dir "$PER_CKPT_DIR"

echo ""
echo "=== Job done at $(date -Is) ==="
echo "Activations + success_rates.json: $PER_CKPT_DIR"
