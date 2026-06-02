#!/usr/bin/env bash
# Full LIBERO-10 steering pipeline, end-to-end.
#
# Runs all 7 stages from experiments/libero/README.md in sequence:
#   (a) start collection server
#   (b) collect activations  — uses $SEED_COLLECT
#   (c) kill collection server
#   (d) build conceptor NPZ
#   (e) run find_best_configs.py sweep — uses $SEED_SWEEP
#   (f) start steering server for held-out final eval
#   (g) final eval × 2 (baseline, then steered)  — uses $SEED_EVAL
#
# Defaults give three disjoint 15-state windows per task (collect 0..14,
# sweep 15..29, eval 30..44). Override via env vars.
#
# Usage (from repo root):
#   bash experiments/libero/run_end_to_end.sh
#   GPU=1 SEED_COLLECT=0 SEED_SWEEP=20 SEED_EVAL=40 bash experiments/libero/run_end_to_end.sh
#
# Required env:
#   GPU            which CUDA device to use (default 0)
#   CHECKPOINT_DIR path to the pi0.5 LIBERO checkpoint (default checkpoints/coast-libero-2000)
#   NUM_EPISODES   eps per task in each stage (default 15)
#   SEED_COLLECT   seed for collection  (default 0)
#   SEED_SWEEP     seed for sweep       (default 15)
#   SEED_EVAL      seed for final eval  (default 30)
#   NUM_WORKERS    subprocess concurrency for eval_all.py (default 10)
#   COLLECT_PORT   port for the collection server (default 8100)
#   EVAL_PORT      port for the final-eval steering server (default 8101)

set -euo pipefail

GPU="${GPU:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/coast-libero-2000}"
NUM_EPISODES="${NUM_EPISODES:-15}"
SEED_COLLECT="${SEED_COLLECT:-0}"
SEED_SWEEP="${SEED_SWEEP:-15}"
SEED_EVAL="${SEED_EVAL:-30}"
NUM_WORKERS="${NUM_WORKERS:-10}"
COLLECT_PORT="${COLLECT_PORT:-8100}"
EVAL_PORT="${EVAL_PORT:-8101}"

# Everything below this line is derived — no need to touch.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVATIONS_DIR="$REPO_ROOT/activations/libero"
NPZ_PATH="$REPO_ROOT/conceptors/libero_conceptors.npz"
LOG_DIR="$REPO_ROOT/experiments/libero/run_logs"
mkdir -p "$LOG_DIR"

# Clean up any background servers on exit / interrupt.
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[cleanup] stopping server PID=$SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 3
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_port() {
    local port="$1" timeout=120 elapsed=0
    until ss -tnl 2>/dev/null | grep -q ":$port "; do
        sleep 3
        elapsed=$((elapsed + 3))
        if (( elapsed > timeout )); then
            echo "[error] server did not bind to port $port within ${timeout}s" >&2
            exit 1
        fi
    done
}

banner() { echo; echo "======= $1 ======="; }

cd "$REPO_ROOT"

# ── (a) Start collection server ───────────────────────────────────────────────
banner "(a) Start collection server on GPU=$GPU port=$COLLECT_PORT"
CUDA_VISIBLE_DEVICES="$GPU" uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output_dir "$ACTIVATIONS_DIR" --port "$COLLECT_PORT" \
    policy:checkpoint --policy.config pi05_libero \
    --policy.dir "$CHECKPOINT_DIR" \
    > "$LOG_DIR/01_collect_server.log" 2>&1 &
SERVER_PID=$!
wait_for_port "$COLLECT_PORT"
echo "[ok] collection server PID=$SERVER_PID bound on port $COLLECT_PORT"

# ── (b) Collect activations ───────────────────────────────────────────────────
banner "(b) Collect activations (seed=$SEED_COLLECT, num_episodes=$NUM_EPISODES)"
(
    cd examples/libero_env
    MUJOCO_GL=egl uv run python eval_all.py \
        --task_suite_name libero_10 \
        --num_episodes "$NUM_EPISODES" --seed "$SEED_COLLECT" --collect --port "$COLLECT_PORT" \
        --num_workers "$NUM_WORKERS"
) 2>&1 | tee "$LOG_DIR/02_collect_client.log"

# ── (c) Kill collection server ────────────────────────────────────────────────
banner "(c) Kill collection server"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
sleep 3

# ── (d) Build conceptor NPZ ───────────────────────────────────────────────────
banner "(d) Build conceptor NPZ → $NPZ_PATH"
CUDA_VISIBLE_DEVICES="" uv run python experiments/libero/compute_conceptors.py \
    --activation_root "$ACTIVATIONS_DIR" \
    --output_path "$NPZ_PATH" \
    2>&1 | tee "$LOG_DIR/03_compute_conceptors.log"

# ── (e) Sweep ─────────────────────────────────────────────────────────────────
banner "(e) Sweep hyperparameters (seed=$SEED_SWEEP)"
CUDA_VISIBLE_DEVICES="$GPU" uv run python experiments/libero/find_best_configs.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --num_episodes "$NUM_EPISODES" --seed "$SEED_SWEEP" \
    2>&1 | tee "$LOG_DIR/04_sweep.log"

# ── (f) Start steering server for final eval ──────────────────────────────────
banner "(f) Start steering server on GPU=$GPU port=$EVAL_PORT"
CUDA_VISIBLE_DEVICES="$GPU" uv run scripts/serve_policy.py --pytorch --steer \
    --conceptor_npz "$NPZ_PATH" --port "$EVAL_PORT" \
    policy:checkpoint --policy.config pi05_libero \
    --policy.dir "$CHECKPOINT_DIR" \
    > "$LOG_DIR/05_eval_server.log" 2>&1 &
SERVER_PID=$!
wait_for_port "$EVAL_PORT"
echo "[ok] steering server PID=$SERVER_PID bound on port $EVAL_PORT"

# ── (g) Final held-out eval × 2 (baseline, steered) ───────────────────────────
banner "(g1) Baseline eval (seed=$SEED_EVAL, no --steer)"
(
    cd examples/libero_env
    MUJOCO_GL=egl uv run python eval_all.py \
        --task_suite_name libero_10 \
        --num_episodes "$NUM_EPISODES" --seed "$SEED_EVAL" --port "$EVAL_PORT" \
        --num_workers "$NUM_WORKERS"
) 2>&1 | tee "$LOG_DIR/06_eval_baseline.log"

# eval_all.py overwrites results.json on each run — preserve the baseline
BASELINE_RESULTS="$REPO_ROOT/examples/libero_env/output/libero_10/results.json"
if [[ -f "$BASELINE_RESULTS" ]]; then
    cp "$BASELINE_RESULTS" "$LOG_DIR/results_baseline.json"
    echo "[ok] saved baseline results → $LOG_DIR/results_baseline.json"
fi

banner "(g2) Steered eval (seed=$SEED_EVAL, --steering_config)"
(
    cd examples/libero_env
    MUJOCO_GL=egl uv run python eval_all.py \
        --task_suite_name libero_10 \
        --num_episodes "$NUM_EPISODES" --seed "$SEED_EVAL" --port "$EVAL_PORT" \
        --num_workers "$NUM_WORKERS" \
        --steer --steering_config "$REPO_ROOT/experiments/libero/best_configs.json"
) 2>&1 | tee "$LOG_DIR/07_eval_steered.log"

if [[ -f "$BASELINE_RESULTS" ]]; then
    cp "$BASELINE_RESULTS" "$LOG_DIR/results_steered.json"
    echo "[ok] saved steered results → $LOG_DIR/results_steered.json"
fi

banner "DONE"
echo "logs:            $LOG_DIR/"
echo "best_configs:    $REPO_ROOT/experiments/libero/best_configs.json"
echo "baseline SR:     $(python3 -c "import json; d=json.load(open('$LOG_DIR/results_baseline.json')); print(f\"{d['mean_success_rate']:.3f}\")" 2>/dev/null || echo "n/a")"
echo "steered SR:      $(python3 -c "import json; d=json.load(open('$LOG_DIR/results_steered.json')); print(f\"{d['mean_success_rate']:.3f}\")" 2>/dev/null || echo "n/a")"
