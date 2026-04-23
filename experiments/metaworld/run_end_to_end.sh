#!/usr/bin/env bash
# Full MetaWorld steering pipeline, end-to-end.
#
# Mirrors the 5-stage flow from experiments/metaworld/README.md.
# MetaWorld collects IN-PROCESS (no server) via `eval_all.py --collect`,
# then switches to the WebSocket server path for the sweep + final eval.
#
# Usage (from repo root):
#   bash experiments/metaworld/run_end_to_end.sh
#   GPU=1 SPLIT=train SEED_COLLECT=0 SEED_SWEEP=15 SEED_EVAL=30 \
#       bash experiments/metaworld/run_end_to_end.sh
#
# Defaults: SPLIT=subset (curated 26-task set, matches examples/metaworld
# eval_all.py's default), NUM_EPISODES=15, NUM_ENVS=16.

set -euo pipefail

GPU="${GPU:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/openpi-metaworld-5000}"
SPLIT="${SPLIT:-subset}"
NUM_ENVS="${NUM_ENVS:-16}"
NUM_EPISODES="${NUM_EPISODES:-15}"
SEED_COLLECT="${SEED_COLLECT:-0}"
SEED_SWEEP="${SEED_SWEEP:-15}"
SEED_EVAL="${SEED_EVAL:-30}"
EVAL_PORT="${EVAL_PORT:-8301}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVATIONS_DIR="$REPO_ROOT/activations/metaworld"
NPZ_PATH="$REPO_ROOT/conceptors/metaworld_conceptors.npz"
LOG_DIR="$REPO_ROOT/experiments/metaworld/run_logs"
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

banner "(a) Collect activations IN-PROCESS (split=$SPLIT, seed=$SEED_COLLECT)"
CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --collect --split "$SPLIT" --num_envs "$NUM_ENVS" --seed "$SEED_COLLECT" \
    --policy.config=pi05_metaworld \
    --policy.dir="$CHECKPOINT_DIR" \
    --collect_output_dir "$ACTIVATIONS_DIR" \
    2>&1 | tee "$LOG_DIR/01_collect.log"

banner "(b) Build conceptor NPZ → $NPZ_PATH"
CUDA_VISIBLE_DEVICES="" uv run python experiments/metaworld/compute_conceptors.py \
    --activation_root "$ACTIVATIONS_DIR" \
    --output_path "$NPZ_PATH" \
    2>&1 | tee "$LOG_DIR/02_compute_conceptors.log"

banner "(c) Sweep hyperparameters (seed=$SEED_SWEEP)"
CUDA_VISIBLE_DEVICES="$GPU" uv run python experiments/metaworld/find_best_configs.py \
    --num_episodes "$NUM_EPISODES" --num_envs "$NUM_ENVS" --seed "$SEED_SWEEP" \
    2>&1 | tee "$LOG_DIR/03_sweep.log"

banner "(d) Start steering server on GPU=$GPU port=$EVAL_PORT"
CUDA_VISIBLE_DEVICES="$GPU" uv run scripts/serve_policy.py --pytorch --steer \
    --conceptor_npz "$NPZ_PATH" --port "$EVAL_PORT" \
    policy:checkpoint --policy.config pi05_metaworld \
    --policy.dir "$CHECKPOINT_DIR" \
    > "$LOG_DIR/04_eval_server.log" 2>&1 &
SERVER_PID=$!
wait_for_port "$EVAL_PORT"
echo "[ok] steering server PID=$SERVER_PID"

banner "(e1) Baseline eval (seed=$SEED_EVAL)"
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --split "$SPLIT" --num_episodes "$NUM_EPISODES" --seed "$SEED_EVAL" --port "$EVAL_PORT" \
    2>&1 | tee "$LOG_DIR/05_eval_baseline.log"

BASELINE_RESULTS="$REPO_ROOT/examples/metaworld/output/ML45-${SPLIT}/results.json"
[[ -f "$BASELINE_RESULTS" ]] && cp "$BASELINE_RESULTS" "$LOG_DIR/results_baseline.json"

banner "(e2) Steered eval (seed=$SEED_EVAL, --steering_config)"
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --split "$SPLIT" --num_episodes "$NUM_EPISODES" --seed "$SEED_EVAL" --port "$EVAL_PORT" \
    --steer --steering_config experiments/metaworld/best_configs.json \
    2>&1 | tee "$LOG_DIR/06_eval_steered.log"

[[ -f "$BASELINE_RESULTS" ]] && cp "$BASELINE_RESULTS" "$LOG_DIR/results_steered.json"

banner "DONE"
echo "logs:            $LOG_DIR/"
echo "best_configs:    $REPO_ROOT/experiments/metaworld/best_configs.json"
echo "baseline SR:     $(python3 -c "import json; d=json.load(open('$LOG_DIR/results_baseline.json')); print(f\"{d['mean_success_rate']:.3f}\")" 2>/dev/null || echo "n/a")"
echo "steered SR:      $(python3 -c "import json; d=json.load(open('$LOG_DIR/results_steered.json')); print(f\"{d['mean_success_rate']:.3f}\")" 2>/dev/null || echo "n/a")"
