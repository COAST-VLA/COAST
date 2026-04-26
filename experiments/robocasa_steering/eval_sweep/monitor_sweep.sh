#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# One-shot status report for a running / recently-completed eval sweep.
#
# Prints:
#   1. SLURM queue state for this user's dp_eval_*/dp_robocasa_aggregate jobs.
#   2. Recent terminal states (sacct since midnight).
#   3. Per-checkpoint latest task tick (from `[i/18] <Task>` markers).
#   4. Per-checkpoint bytes written (activations growing on disk).
#   5. Per-checkpoint success_rates.json status + top-level results.json.
#   6. Any traceback / error / OOM signatures in the logs (last line only).
#
# Overrides:
#   ACTIVATIONS_ROOT   Default: repo/eval_sweep_results
#   SLURM_LOGS_DIR     Default: repo/slurm_logs
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ACTIVATIONS_ROOT="${ACTIVATIONS_ROOT:-/mnt/bird_home/kim34/eval_sweep_results}"
SLURM_LOGS_DIR="${SLURM_LOGS_DIR:-$REPO_ROOT/slurm_logs}"

hr() { printf -- '─%.0s' $(seq 1 72); echo; }

echo "=== eval sweep status @ $(date -Is) ========================================"
echo "REPO_ROOT:        $REPO_ROOT"
echo "ACTIVATIONS_ROOT: $ACTIVATIONS_ROOT"
echo "SLURM_LOGS_DIR:   $SLURM_LOGS_DIR"
hr

# 1. Queue --------------------------------------------------------------------
echo "[queue]"
squeue -u "$USER" -o "%.10i %.42j %.2t %.10M %.12L %R" \
    | awk 'NR==1 || $2 ~ /^dp_(eval|robocasa)/' || true
hr

# 2. Recent terminal states ---------------------------------------------------
echo "[sacct since 00:00 — dp_eval_* / dp_robocasa_aggregate]"
sacct -u "$USER" --starttime=today \
    --format=JobID,JobName%42,State,Elapsed,ExitCode \
    --noheader \
    | awk '$2 ~ /^dp_(eval|robocasa)/ {print}' \
    | sort -k1n | tail -20 || true
hr

# 3. Per-job latest task tick (only jobs currently in queue, or newest log if
#    no queued jobs, so stale logs from older failed submissions don't clutter).
# ---------------------------------------------------------------------------
echo "[per-job latest task tick]"
shopt -s nullglob
LOGS=( "$SLURM_LOGS_DIR"/dp_eval_*.out )

QUEUED_JIDS=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null \
    | awk '$2 ~ /^dp_eval_/ {print $1}' || true)

PICK=()
if [ -n "$QUEUED_JIDS" ]; then
    for jid in $QUEUED_JIDS; do
        for f in "${LOGS[@]}"; do
            case "$f" in *_"$jid".out) PICK+=( "$f" );; esac
        done
    done
elif [ ${#LOGS[@]} -gt 0 ]; then
    # No queued jobs — show the newest log per ckpt stem.
    declare -A NEWEST
    for f in "${LOGS[@]}"; do
        b=$(basename "$f" .out); stem=${b#dp_eval_}; stem=${stem%_*}
        if [ -z "${NEWEST[$stem]:-}" ] || [ "$f" -nt "${NEWEST[$stem]}" ]; then
            NEWEST[$stem]="$f"
        fi
    done
    PICK=( "${NEWEST[@]}" )
fi

if [ ${#PICK[@]} -eq 0 ]; then
    echo "  (no dp_eval_*.out logs found)"
else
    for f in "${PICK[@]}"; do
        b=$(basename "$f" .out); jid=${b##*_}
        stem=${b#dp_eval_}; stem=${stem%_*}
        tick=$(grep -E "^\[[0-9]+/18\]" "$f" 2>/dev/null | tail -1 || true)
        printf "  %-8s %-44s %s\n" "$jid" "$stem" "${tick:-<no task tick yet>}"
    done
fi
hr

# 4. Disk usage per ckpt ------------------------------------------------------
echo "[activations on disk]"
if [ -d "$ACTIVATIONS_ROOT" ]; then
    du -sh "$ACTIVATIONS_ROOT"/*/ 2>/dev/null \
        | sort -k2 || echo "  (empty)"
else
    echo "  (ACTIVATIONS_ROOT does not exist yet)"
fi
hr

# 5. Results files ------------------------------------------------------------
echo "[results files]"
FOUND_ANY=0
if [ -d "$ACTIVATIONS_ROOT" ]; then
    while IFS= read -r -d '' sr; do
        FOUND_ANY=1
        rel=${sr#"$ACTIVATIONS_ROOT"/}
        size=$(stat -c %s "$sr")
        printf "  %-60s %s bytes\n" "$rel" "$size"
    done < <(find "$ACTIVATIONS_ROOT" -maxdepth 2 -name success_rates.json -print0 2>/dev/null)
fi
if [ $FOUND_ANY -eq 0 ]; then
    echo "  (no per-ckpt success_rates.json yet)"
fi
if [ -f "$ACTIVATIONS_ROOT/results.json" ]; then
    echo "  FINAL: $ACTIVATIONS_ROOT/results.json ($(stat -c %s "$ACTIVATIONS_ROOT/results.json") bytes)"
else
    echo "  (no final results.json yet)"
fi
hr

# 6. Errors — limited to the currently-selected logs (section 3 above) so
#    old failed-submission tracebacks don't drown out the current run.
# ---------------------------------------------------------------------------
echo "[first error-ish line per active log, if any]"
ERR_FOUND=0
for f in "${PICK[@]:-}"; do
    [ -z "$f" ] && continue
    b=$(basename "$f" .out)
    hit=$(grep -nE "Traceback|^[A-Z][a-zA-Z]*Error|FAILED|CUDA out of memory|^Killed|assert" "$f" 2>/dev/null | head -1 || true)
    if [ -n "$hit" ]; then
        ERR_FOUND=1
        printf "  %s: %s\n" "$b" "$hit"
    fi
done
if [ $ERR_FOUND -eq 0 ]; then
    echo "  (no error signatures in the active logs)"
fi
hr
