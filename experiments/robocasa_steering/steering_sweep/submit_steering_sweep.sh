#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Top-level launcher for the per_step steering sweep.
#
# For each ckpt subdir under $CONCEPTORS_ROOT (matching $CHECKPOINTS_DIR):
#   submits one steering_sweep.sbatch job pinned to dj-a40-1 with the
#   appropriate CHECKPOINT, CONCEPTOR_NPZ, RECIPE_JSON, OUTPUT_ROOT.
#
# Output JIDs printed; aggregator queued with afterany on all of them.
#
# Env-var overrides:
#   CHECKPOINTS_DIR  default: repo/checkpoints
#   CONCEPTORS_ROOT  default: repo/experiments/robocasa_steering/conceptors
#   RECIPE_JSON      default: repo/experiments/robocasa_steering/steering_sweep/recipes.json
#   OUTPUT_ROOT      default: /mnt/bird_home/kim34/steering_sweep_results
#   NODELIST         default: dj-a40-1.grasp.maas
#   DRY_RUN=1        print sbatch invocations without submitting
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SWEEP_DIR="$REPO_ROOT/experiments/robocasa_steering/steering_sweep"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$REPO_ROOT/checkpoints}"
CONCEPTORS_ROOT="${CONCEPTORS_ROOT:-$REPO_ROOT/experiments/robocasa_steering/conceptors}"
RECIPE_JSON="${RECIPE_JSON:-$SWEEP_DIR/recipes.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/bird_home/kim34/steering_sweep_results}"
NODELIST="${NODELIST:-dj-a40-1.grasp.maas}"
DRY_RUN="${DRY_RUN:-0}"

[ -f "$RECIPE_JSON" ] || { echo "missing recipe: $RECIPE_JSON" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT" "$REPO_ROOT/slurm_logs"

shopt -s nullglob
CKPTS=( "$CHECKPOINTS_DIR"/*.ckpt )
[ ${#CKPTS[@]} -gt 0 ] || { echo "no .ckpt found in $CHECKPOINTS_DIR" >&2; exit 1; }

echo "=== diffusion_policy / RoboCasa steering sweep =========================="
echo "REPO_ROOT:       $REPO_ROOT"
echo "CHECKPOINTS_DIR: $CHECKPOINTS_DIR"
echo "CONCEPTORS_ROOT: $CONCEPTORS_ROOT"
echo "RECIPE_JSON:     $RECIPE_JSON"
echo "OUTPUT_ROOT:     $OUTPUT_ROOT"
echo "NODELIST:        $NODELIST"
echo "Checkpoints (${#CKPTS[@]}):"
printf '  - %s\n' "${CKPTS[@]}"
echo "========================================================================="

JIDS=()
for CKPT in "${CKPTS[@]}"; do
    STEM="$(basename "$CKPT" .ckpt)"
    CONCEPTOR_NPZ="$CONCEPTORS_ROOT/$STEM/conceptors.npz"
    if [ ! -f "$CONCEPTOR_NPZ" ]; then
        echo "[skip] $STEM: missing $CONCEPTOR_NPZ"
        continue
    fi
    # If recipes have no entry for this ckpt, skip (e.g. all-zero-success ckpt).
    HAS_ENTRY=$("$REPO_ROOT/.venv/bin/python" -c "
import json, sys
sys.exit(0 if '$STEM' in json.load(open('$RECIPE_JSON')) else 1)
" && echo yes || echo no)
    if [ "$HAS_ENTRY" != "yes" ]; then
        echo "[skip] $STEM: not in recipe"
        continue
    fi

    CMD=(
        sbatch --parsable
        --nodelist="$NODELIST"
        --job-name="dp_steer_${STEM}"
        --export="ALL,REPO_ROOT=$REPO_ROOT,CHECKPOINT=$CKPT,CONCEPTOR_NPZ=$CONCEPTOR_NPZ,RECIPE_JSON=$RECIPE_JSON,OUTPUT_ROOT=$OUTPUT_ROOT"
        "$SWEEP_DIR/steering_sweep.sbatch"
    )
    echo ""
    echo "Submitting steering for $STEM:"
    printf '  %q ' "${CMD[@]}"; echo
    if [ "$DRY_RUN" = "1" ]; then
        JID="DRYRUN_$STEM"
    else
        JID="$("${CMD[@]}")"
    fi
    echo "  -> job id: $JID"
    JIDS+=( "$JID" )
done

echo ""
echo "=== Steering jobs queued ==============================================="
echo "Per-checkpoint JIDs: ${JIDS[*]}"
echo "Output:              $OUTPUT_ROOT/<ckpt_stem>/<task>/summary.json"
echo "Track:               squeue -u \$USER"
