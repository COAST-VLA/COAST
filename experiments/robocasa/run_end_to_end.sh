#!/usr/bin/env bash
# Full RoboCasa steering pipeline, end-to-end.
#
# Mirrors the 7-stage flow from experiments/robocasa/README.md. RoboCasa
# stepping is ~2× slower than LIBERO, so plan for a longer run.
#
# Usage (from repo root):
#   bash experiments/robocasa/run_end_to_end.sh
#   GPU=1 TASK_SET=atomic_seen SEED_COLLECT=0 SEED_SWEEP=15 SEED_EVAL=30 \
#       bash experiments/robocasa/run_end_to_end.sh
#
# Defaults: TASK_SET=subset (curated list, matches examples/robocasa_env
# eval_all.py's default), NUM_EPISODES=15.

set -euo pipefail

GPU="${GPU:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pi05_pretrain_human300/multitask_learning/75000}"
TASK_SET="${TASK_SET:-subset}"
SPLIT="${SPLIT:-pretrain}"
NUM_EPISODES="${NUM_EPISODES:-15}"
SEED_COLLECT="${SEED_COLLECT:-0}"
SEED_SWEEP="${SEED_SWEEP:-15}"
SEED_EVAL="${SEED_EVAL:-30}"
NUM_WORKERS="${NUM_WORKERS:-5}"
COLLECT_PORT="${COLLECT_PORT:-8200}"
EVAL_PORT="${EVAL_PORT:-8201}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVATIONS_DIR="$REPO_ROOT/activations/robocasa"
NPZ_PATH="$REPO_ROOT/conceptors/robocasa_conceptors.npz"
LOG_DIR="$REPO_ROOT/experiments/robocasa/run_logs"
mkdir -p "$LOG_DIR"

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
        (( elapsed > timeout )) && { echo "[error] port $port never bound" >&2; exit 1; }
    done
}

banner() { echo; echo "======= $1 ======="; }

cd "$REPO_ROOT"

banner "(a) Start collection server on GPU=$GPU port=$COLLECT_PORT"
CUDA_VISIBLE_DEVICES="$GPU" uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output_dir "$ACTIVATIONS_DIR" --port "$COLLECT_PORT" \
    policy:checkpoint --policy.config pi05_robocasa \
    --policy.dir "$CHECKPOINT_DIR" \
    > "$LOG_DIR/01_collect_server.log" 2>&1 &
SERVER_PID=$!
wait_for_port "$COLLECT_PORT"
echo "[ok] collection server PID=$SERVER_PID"

banner "(b) Collect activations (task_set=$TASK_SET, seed=$SEED_COLLECT)"
(
    cd examples/robocasa_env
    MUJOCO_GL=egl uv run python eval_all.py \
        --task_set "$TASK_SET" \
        --num_episodes "$NUM_EPISODES" --seed "$SEED_COLLECT" --collect --port "$COLLECT_PORT" \
        --num_workers "$NUM_WORKERS"
) 2>&1 | tee "$LOG_DIR/02_collect_client.log"

banner "(c) Kill collection server"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
sleep 3

banner "(d) Build conceptor NPZ → $NPZ_PATH"
CUDA_VISIBLE_DEVICES="" uv run python experiments/robocasa/compute_conceptors.py \
    --activation_root "$ACTIVATIONS_DIR" \
    --output_path "$NPZ_PATH" \
    2>&1 | tee "$LOG_DIR/03_compute_conceptors.log"

banner "(e) Sweep hyperparameters (seed=$SEED_SWEEP)"
CUDA_VISIBLE_DEVICES="$GPU" uv run python experiments/robocasa/find_best_configs.py \
    --num_episodes "$NUM_EPISODES" --seed "$SEED_SWEEP" --split "$SPLIT" \
    2>&1 | tee "$LOG_DIR/04_sweep.log"

banner "(f) Start steering server on GPU=$GPU port=$EVAL_PORT"
CUDA_VISIBLE_DEVICES="$GPU" uv run scripts/serve_policy.py --pytorch --steer \
    --conceptor_npz "$NPZ_PATH" --port "$EVAL_PORT" \
    policy:checkpoint --policy.config pi05_robocasa \
    --policy.dir "$CHECKPOINT_DIR" \
    > "$LOG_DIR/05_eval_server.log" 2>&1 &
SERVER_PID=$!
wait_for_port "$EVAL_PORT"
echo "[ok] steering server PID=$SERVER_PID"

banner "(g1) Baseline eval (seed=$SEED_EVAL)"
(
    cd examples/robocasa_env
    MUJOCO_GL=egl uv run python eval_all.py \
        --task_set "$TASK_SET" \
        --num_episodes "$NUM_EPISODES" --seed "$SEED_EVAL" --port "$EVAL_PORT" \
        --num_workers "$NUM_WORKERS"
) 2>&1 | tee "$LOG_DIR/06_eval_baseline.log"

BASELINE_RESULTS="$REPO_ROOT/examples/robocasa_env/output/${TASK_SET}-${SPLIT}/results.json"
[[ -f "$BASELINE_RESULTS" ]] && cp "$BASELINE_RESULTS" "$LOG_DIR/results_baseline.json"

banner "(g2) Steered eval (seed=$SEED_EVAL, --steering_config)"
(
    cd examples/robocasa_env
    MUJOCO_GL=egl uv run python eval_all.py \
        --task_set "$TASK_SET" \
        --num_episodes "$NUM_EPISODES" --seed "$SEED_EVAL" --port "$EVAL_PORT" \
        --num_workers "$NUM_WORKERS" \
        --steer --steering_config experiments/robocasa/best_configs.json
) 2>&1 | tee "$LOG_DIR/07_eval_steered.log"

[[ -f "$BASELINE_RESULTS" ]] && cp "$BASELINE_RESULTS" "$LOG_DIR/results_steered.json"

banner "DONE"
echo "logs:            $LOG_DIR/"
echo "best_configs:    $REPO_ROOT/experiments/robocasa/best_configs.json"
echo "baseline SR:     $(python3 -c "import json; d=json.load(open('$LOG_DIR/results_baseline.json')); print(f\"{d['mean_success_rate']:.3f}\")" 2>/dev/null || echo "n/a")"
echo "steered SR:      $(python3 -c "import json; d=json.load(open('$LOG_DIR/results_steered.json')); print(f\"{d['mean_success_rate']:.3f}\")" 2>/dev/null || echo "n/a")"
