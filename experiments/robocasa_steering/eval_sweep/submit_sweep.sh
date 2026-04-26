#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Top-level launcher.
#
# Submits one eval_sweep.sbatch job per .ckpt file in CHECKPOINTS_DIR (each
# gets its own GPU), then submits aggregate_results.sbatch with a dependency
# on all of them. The aggregator fires once every per-checkpoint job finishes
# (afterany, so partial results still get combined on failures).
#
# Overrides via env vars before invocation:
#   CHECKPOINTS_DIR    Directory of .ckpt files (default: repo/checkpoints).
#   ACTIVATIONS_ROOT   Where activations + results.json land
#                      (default: repo/eval_sweep_results).
#   SPLIT              RoboCasa split (default: pretrain).
#   NUM_ROLLOUTS       Rollouts per task (default: 30).
#   NUM_ENVS           Parallel envs per task (default: 15).
#   DRY_RUN=1          Print what would be submitted without running sbatch.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWEEP_DIR="$REPO_ROOT/experiments/robocasa_steering/eval_sweep"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$REPO_ROOT/checkpoints}"
ACTIVATIONS_ROOT="${ACTIVATIONS_ROOT:-/mnt/bird_home/kim34/eval_sweep_results}"
SPLIT="${SPLIT:-pretrain}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-30}"
NUM_ENVS="${NUM_ENVS:-15}"
START_TASK="${START_TASK:-}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$ACTIVATIONS_ROOT" "$REPO_ROOT/slurm_logs"

shopt -s nullglob
CKPTS=( "$CHECKPOINTS_DIR"/*.ckpt )
if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "No .ckpt files found in $CHECKPOINTS_DIR" >&2
    exit 1
fi

echo "=== diffusion_policy / RoboCasa 18-task eval sweep ===================="
echo "REPO_ROOT:         $REPO_ROOT"
echo "CHECKPOINTS_DIR:   $CHECKPOINTS_DIR"
echo "ACTIVATIONS_ROOT:  $ACTIVATIONS_ROOT"
echo "SPLIT / rollouts / envs:  $SPLIT / $NUM_ROLLOUTS / $NUM_ENVS"
echo "Checkpoints (${#CKPTS[@]}):"
printf '  - %s\n' "${CKPTS[@]}"
echo "========================================================================"

JOB_IDS=()
for CKPT in "${CKPTS[@]}"; do
    CKPT_STEM="$(basename "$CKPT" .ckpt)"
    CMD=(
        sbatch --parsable
        --job-name="dp_eval_${CKPT_STEM}"
        --export="ALL,REPO_ROOT=$REPO_ROOT,CHECKPOINT=$CKPT,ACTIVATIONS_ROOT=$ACTIVATIONS_ROOT,SPLIT=$SPLIT,NUM_ROLLOUTS=$NUM_ROLLOUTS,NUM_ENVS=$NUM_ENVS,START_TASK=$START_TASK"
        "$SWEEP_DIR/eval_sweep.sbatch"
    )
    echo ""
    echo "Submitting eval for $CKPT_STEM:"
    printf '  %q ' "${CMD[@]}"; echo
    if [ "$DRY_RUN" = "1" ]; then
        JID="DRYRUN_$CKPT_STEM"
    else
        JID="$("${CMD[@]}")"
    fi
    echo "  -> job id: $JID"
    JOB_IDS+=( "$JID" )
done

# Join JOB_IDS with ':' for --dependency=afterany:A:B:C
DEPS="$(IFS=:; echo "${JOB_IDS[*]}")"

AGG_CMD=(
    sbatch --parsable
    --dependency="afterany:$DEPS"
    --export="ALL,REPO_ROOT=$REPO_ROOT,ACTIVATIONS_ROOT=$ACTIVATIONS_ROOT"
    "$SWEEP_DIR/aggregate_results.sbatch"
)
echo ""
echo "Submitting aggregator (runs after all eval jobs finish):"
printf '  %q ' "${AGG_CMD[@]}"; echo
if [ "$DRY_RUN" = "1" ]; then
    AGG_JID="DRYRUN_aggregate"
else
    AGG_JID="$("${AGG_CMD[@]}")"
fi
echo "  -> job id: $AGG_JID"

echo ""
echo "=== All jobs queued =================================================="
echo "Per-checkpoint JIDs: ${JOB_IDS[*]}"
echo "Aggregator JID:      $AGG_JID"
echo "Track:       squeue -u \$USER"
echo "Final file:  $ACTIVATIONS_ROOT/results.json"
