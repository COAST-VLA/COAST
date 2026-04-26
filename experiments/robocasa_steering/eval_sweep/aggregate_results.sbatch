#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Final aggregator — combines every per-checkpoint success_rates.json under
# $ACTIVATIONS_ROOT into a single results.json.
#
# Submitted by submit_sweep.sh with --dependency=afterany:<jid1>:<jid2>:... so
# it fires once all per-checkpoint eval jobs have finished (success or fail).
# CPU-only.
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=dp_robocasa_aggregate
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dj-high

set -euo pipefail

: "${ACTIVATIONS_ROOT:?need ACTIVATIONS_ROOT env var}"
: "${REPO_ROOT:?need REPO_ROOT env var (exported by submit_sweep.sh)}"
PYTHON="${REPO_ROOT}/.venv/bin/python"

echo "Aggregating results under: $ACTIVATIONS_ROOT"
"$PYTHON" "$REPO_ROOT/experiments/robocasa_steering/eval_sweep/aggregate_results.py" \
    all --activations-root "$ACTIVATIONS_ROOT"
echo "Done. Top-level results: $ACTIVATIONS_ROOT/results.json"
