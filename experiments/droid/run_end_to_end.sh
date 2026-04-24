#!/usr/bin/env bash
# Partial DROID steering pipeline, end-to-end.
#
# DROID is a real-robot harness, so the rollout steps (collection and final
# eval) are manual — the operator runs main.py on the DROID control laptop,
# enters free-form instructions, and labels success per rollout. This script
# automates what CAN be automated on the GPU host:
#
#   (a) start collection server
#       [manual: operator runs main.py --collect on the DROID laptop]
#   (c) kill collection server   (prompts the user to confirm collection is done)
#   (d) build conceptor NPZ
#   (e) run select_parameters.py diagnostic narrower
#   (f) start steering server
#       [manual: operator runs main.py --steer per shortlisted (layer,α,β)]
#
# Usage (on the GPU host, from repo root):
#   bash experiments/droid/run_end_to_end.sh

set -euo pipefail

GPU="${GPU:-0}"
CHECKPOINT="${CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_droid}"
COLLECT_PORT="${COLLECT_PORT:-8000}"
EVAL_PORT="${EVAL_PORT:-8001}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTIVATIONS_DIR="$REPO_ROOT/activations/droid"
NPZ_PATH="$REPO_ROOT/conceptors/droid_conceptors.npz"
SELECTED_PARAMS="$REPO_ROOT/experiments/droid/selected_params.json"
LOG_DIR="$REPO_ROOT/experiments/droid/run_logs"
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

wait_for_operator() {
    echo
    echo ">>>>>>> $1"
    echo ">>>>>>> Press ENTER here on the GPU host when the operator is done."
    read -r
}

cd "$REPO_ROOT"

banner "(a) Start collection server on GPU=$GPU port=$COLLECT_PORT"
CUDA_VISIBLE_DEVICES="$GPU" uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output_dir "$ACTIVATIONS_DIR" --port "$COLLECT_PORT" \
    policy:checkpoint --policy.config pi05_droid \
    --policy.dir "$CHECKPOINT" \
    > "$LOG_DIR/01_collect_server.log" 2>&1 &
SERVER_PID=$!
wait_for_port "$COLLECT_PORT"
echo "[ok] collection server PID=$SERVER_PID bound on port $COLLECT_PORT"

wait_for_operator "(b) MANUAL: on the DROID laptop, run main.py --collect per instruction. Aim for 15-30 rollouts per instruction for stable conceptors. Connect to this server at <server_ip>:$COLLECT_PORT."

banner "(c) Kill collection server"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""
sleep 3

banner "(d) Build conceptor NPZ → $NPZ_PATH"
CUDA_VISIBLE_DEVICES="" uv run python experiments/droid/compute_conceptors.py \
    --activation_root "$ACTIVATIONS_DIR" \
    --output_path "$NPZ_PATH" \
    2>&1 | tee "$LOG_DIR/02_compute_conceptors.log"

banner "(e) Run select_parameters.py diagnostic → $SELECTED_PARAMS"
CUDA_VISIBLE_DEVICES="" uv run python experiments/droid/select_parameters.py \
    --conceptor-npz "$NPZ_PATH" \
    --output-json "$SELECTED_PARAMS" \
    2>&1 | tee "$LOG_DIR/03_select_parameters.log"

echo
echo "Shortlisted config:"
cat "$SELECTED_PARAMS"
echo

banner "(f) Start steering server on GPU=$GPU port=$EVAL_PORT"
CUDA_VISIBLE_DEVICES="$GPU" uv run scripts/serve_policy.py --pytorch --steer \
    --conceptor_npz "$NPZ_PATH" --port "$EVAL_PORT" \
    policy:checkpoint --policy.config pi05_droid \
    --policy.dir "$CHECKPOINT" \
    > "$LOG_DIR/04_eval_server.log" 2>&1 &
SERVER_PID=$!
wait_for_port "$EVAL_PORT"
echo "[ok] steering server PID=$SERVER_PID bound on port $EVAL_PORT"

wait_for_operator "(g) MANUAL: on the DROID laptop, run main.py --steer for each (layer, α, β) in $SELECTED_PARAMS. Also run an unsteered baseline for comparison. Operator labels success per rollout (main.py writes results/eval_<ts>.csv)."

banner "DONE"
echo "logs:            $LOG_DIR/"
echo "npz:             $NPZ_PATH"
echo "selected_params: $SELECTED_PARAMS"
echo "Baseline vs steered comparison is on the DROID laptop (results/eval_*.csv)."
