#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Per-checkpoint steering sweep.
#
# For ONE checkpoint:
#   * loads its conceptor .npz (with per-step alpha-aware matrices)
#   * iterates over the tasks listed in $RECIPE_JSON for that ckpt
#   * invokes experiments/robocasa_steering/steering.py with --strategies per_step
#     at the recipe's (layer, alpha, beta) per task
#   * 30 rollouts × 15 envs per task (matches the activation-collection sweep)
#   * NO activation collection during evaluation
#
# Per-task output written to:
#   $OUTPUT_ROOT/<ckpt_stem>/<task>/summary.json    (steering condition success rates)
#
# Required env vars (set by submit_steering_sweep.sh):
#   REPO_ROOT, CHECKPOINT, CONCEPTOR_NPZ, RECIPE_JSON, OUTPUT_ROOT
#
# Optional env vars with defaults:
#   SPLIT=pretrain  NUM_ROLLOUTS=30  NUM_ENVS=15
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=dp_steer
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --time=23:00:00
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high

set -euo pipefail

: "${REPO_ROOT:?need REPO_ROOT}"
: "${CHECKPOINT:?need CHECKPOINT}"
: "${CONCEPTOR_NPZ:?need CONCEPTOR_NPZ}"
: "${RECIPE_JSON:?need RECIPE_JSON}"
: "${OUTPUT_ROOT:?need OUTPUT_ROOT}"
SPLIT="${SPLIT:-pretrain}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-30}"
NUM_ENVS="${NUM_ENVS:-15}"
STRATEGIES="${STRATEGIES:-per_step}"      # space-separated, e.g. "per_step", "global", "per_step global"
SKIP_BASELINE="${SKIP_BASELINE:-0}"         # 1 to skip baseline
TASKS_OVERRIDE="${TASKS_OVERRIDE:-}"        # comma-separated; if set, restricts task loop to these
PER_STEP_STATIC_DS="${PER_STEP_STATIC_DS:-}"  # if set (int), per_step uses pi05-style static-at-ds semantics
ALPHAS_OVERRIDE="${ALPHAS_OVERRIDE:-}"        # space-separated; if set, overrides recipe per-task alpha
BETAS_OVERRIDE="${BETAS_OVERRIDE:-}"          # space-separated; if set, overrides recipe per-task beta

PYTHON="$REPO_ROOT/.venv/bin/python"
CKPT_STEM="$(basename "$CHECKPOINT" .ckpt)"
PER_CKPT_OUT="$OUTPUT_ROOT/$CKPT_STEM"
mkdir -p "$PER_CKPT_OUT"

# Pull tasks from the recipe (this ckpt's subdict). Optionally restrict via TASKS_OVERRIDE.
mapfile -t TASKS < <("$PYTHON" -c "
import json, sys
all_recipes = json.load(open('$RECIPE_JSON'))
ck = '$CKPT_STEM'
if ck not in all_recipes:
    sys.exit('ckpt $CKPT_STEM not in $RECIPE_JSON')
override = '$TASKS_OVERRIDE'.strip()
keep = [t for t in override.split(',') if t.strip()] if override else None
for t in all_recipes[ck]:
    if keep is None or t in keep:
        print(t)
")

echo "======================================================================"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "HOST:          $(hostname)"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CHECKPOINT:    $CHECKPOINT"
echo "CONCEPTOR_NPZ: $CONCEPTOR_NPZ"
echo "RECIPE_JSON:   $RECIPE_JSON"
echo "SPLIT:         $SPLIT  ROLLOUTS:$NUM_ROLLOUTS  ENVS:$NUM_ENVS"
echo "OUTPUT:        $PER_CKPT_OUT"
echo "TASKS (${#TASKS[@]}):"
printf '  - %s\n' "${TASKS[@]}"
echo "======================================================================"

export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# steering.py lives under experiments/, so REPO_ROOT must be on PYTHONPATH for
# `from diffusion_policy.workspace...` to resolve.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    idx=$((i + 1))

    # Pull recipe for this (ckpt, task). Supports both legacy single-value schema
    # (`{layer, alpha, beta}`) and list schema (`{layer, alphas: [...], betas: [...]}`).
    LAYER=$("$PYTHON" -c "import json; r=json.load(open('$RECIPE_JSON'))['$CKPT_STEM']['$TASK']; print(r['layer'])")
    ALPHA_STR=$("$PYTHON" -c "import json; r=json.load(open('$RECIPE_JSON'))['$CKPT_STEM']['$TASK']; A=r['alphas'] if isinstance(r.get('alphas'), list) else [r['alpha']]; print(' '.join(str(x) for x in A))")
    BETA_STR=$("$PYTHON" -c "import json; r=json.load(open('$RECIPE_JSON'))['$CKPT_STEM']['$TASK']; B=r['betas']  if isinstance(r.get('betas'),  list) else [r['beta']]; print(' '.join(str(x) for x in B))")
    read -r -a ALPHA_LIST <<< "$ALPHA_STR"
    read -r -a BETA_LIST  <<< "$BETA_STR"
    TASK_OUT="$PER_CKPT_OUT/$TASK"

    echo ""
    echo "[${idx}/${#TASKS[@]}] $TASK  L=$LAYER alphas=[${ALPHA_LIST[*]}] betas=[${BETA_LIST[*]}]  @ $(date -Is)"

    SKIP_BASELINE_FLAG=()
    if [ "$SKIP_BASELINE" = "1" ]; then SKIP_BASELINE_FLAG=( "--skip-baseline" ); fi
    PER_STEP_STATIC_DS_FLAG=()
    if [ -n "$PER_STEP_STATIC_DS" ]; then PER_STEP_STATIC_DS_FLAG=( "--per-step-static-ds" "$PER_STEP_STATIC_DS" ); fi

    # Override recipe alpha/beta lists if env vars set (space-separated).
    if [ -n "$ALPHAS_OVERRIDE" ]; then read -r -a ALPHA_LIST <<< "$ALPHAS_OVERRIDE"; fi
    if [ -n "$BETAS_OVERRIDE"  ]; then read -r -a BETA_LIST  <<< "$BETAS_OVERRIDE";  fi

    "$PYTHON" experiments/robocasa_steering/steering.py \
        --checkpoint "$CHECKPOINT" \
        --conceptor_npz "$CONCEPTOR_NPZ" \
        --task "$TASK" \
        --split "$SPLIT" \
        --num_rollouts "$NUM_ROLLOUTS" \
        --num_envs "$NUM_ENVS" \
        --layers "$LAYER" \
        --alphas "${ALPHA_LIST[@]}" \
        --betas "${BETA_LIST[@]}" \
        --strategies $STRATEGIES \
        --output_dir "$PER_CKPT_OUT" \
        "${SKIP_BASELINE_FLAG[@]}" \
        "${PER_STEP_STATIC_DS_FLAG[@]}"

    echo "[${idx}/${#TASKS[@]}] $TASK  done @ $(date -Is)"
done

echo ""
echo "=== Job done at $(date -Is) ==="
echo "Per-task summaries under: $PER_CKPT_OUT/<task>/summary.json"
