#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Re-collect activations for the 3 missing/corrupted RoboCasa tasks on GR00T
# N1.5. One SLURM job per task — each job:
#   1. Launches `groot_env/serve.py --collect-activations` in the background.
#   2. Waits for the server port to bind.
#   3. Runs `examples/robocasa_env/main.py --env_name <task> --collect`.
#   4. Tears the server down.
# The server writes directly into the HF cache path so the new tasks land
# alongside the 4 existing good tasks with matching directory layout.
#
# Usage (from any directory):
#   bash experiments/groot_robocasa/collect_missing_activations.sh
#   bash experiments/groot_robocasa/collect_missing_activations.sh --dry-run
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
ACT_OUTPUT_ROOT="${OPENPI_DATA_HOME}/huggingface/lerobot/brandonyang/groot_n15-robocasa-activations-v1-15env"
CHECKPOINT_DIR="../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000"

TASKS=(
    "TurnOnElectricKettle"
    "PickPlaceCounterToCabinet"
    "PickPlaceCounterToStove"
)

BASE_PORT=8500
NUM_EPISODES=15

SCRIPT_DIR="${REPO_ROOT}/experiments/groot_robocasa/steering_results/scripts"
LOG_DIR="${REPO_ROOT}/experiments/groot_robocasa/steering_results/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

declare -a COLLECT_JIDS=()

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    PORT=$((BASE_PORT + i))
    JOB_SCRIPT="${SCRIPT_DIR}/collect_${TASK}.sh"

    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
export MUJOCO_GL=egl
export TORCH_COMPILE_DISABLE=1
export HF_HUB_DOWNLOAD_TIMEOUT=600

PORT=${PORT}
TASK=${TASK}
SERVER_LOG="${LOG_DIR}/collect_\${TASK}_server_\${SLURM_JOB_ID}.log"
echo "=== Collect \${TASK} | Node: \$(hostname) | GPU: \${CUDA_VISIBLE_DEVICES} | Port: \${PORT} ==="

# ── Start GR00T N1.5 server (collection mode) in background ──
cd ${REPO_ROOT}/groot_env
uv run python serve.py \\
    --port \${PORT} \\
    --model-path ${CHECKPOINT_DIR} \\
    --collect-activations \\
    --output-dir ${ACT_OUTPUT_ROOT} \\
    > "\${SERVER_LOG}" 2>&1 &
SERVER_PID=\$!
echo "Server PID: \${SERVER_PID}, logs -> \${SERVER_LOG}"

# Shut the server down on any exit (success, failure, or client kill).
cleanup() {
    if kill -0 \${SERVER_PID} 2>/dev/null; then
        echo "Stopping server PID \${SERVER_PID}"
        kill \${SERVER_PID} || true
        wait \${SERVER_PID} 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Wait for server port to bind (policy loading takes ~60-90 s) ──
for attempt in \$(seq 1 60); do
    if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', \${PORT}))" 2>/dev/null; then
        echo "Server ready after \$((attempt * 5)) s"
        break
    fi
    if ! kill -0 \${SERVER_PID} 2>/dev/null; then
        echo "ERROR: server died before binding. Tail of server log:"
        tail -60 "\${SERVER_LOG}" || true
        exit 1
    fi
    sleep 5
done

# ── Run the robocasa client with --collect ──
cd ${REPO_ROOT}/examples/robocasa_env
uv run python main.py \\
    --env_name \${TASK} \\
    --num_episodes ${NUM_EPISODES} \\
    --port \${PORT} \\
    --collect

echo "=== Done: \${TASK} ==="
EOF
    chmod +x "$JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] ${TASK} -> ${JOB_SCRIPT} (port=${PORT})"
    else
        JID=$(sbatch \
            --parsable \
            --job-name="grc-collect-${TASK}" \
            --partition=dgx-b200 \
            --exclude=dgx016 \
            --gres=gpu:1 \
            --cpus-per-task=12 \
            --mem=96G \
            --time=8:00:00 \
            --output="${LOG_DIR}/collect_${TASK}_%j.out" \
            --error="${LOG_DIR}/collect_${TASK}_%j.err" \
            "$JOB_SCRIPT")
        echo "Submitted ${TASK} as job ${JID}"
        COLLECT_JIDS+=("$JID")
    fi
done

# ── Chain conceptor rebuild afterok:all-3 ─────────────────────────────────────
REBUILD_SCRIPT="${SCRIPT_DIR}/rebuild_conceptors.sh"
cat > "$REBUILD_SCRIPT" << EOF
#!/bin/bash
set -e
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
cd ${REPO_ROOT}/groot_env
echo "=== Rebuilding conceptors (all 7 tasks) ==="
uv run python ${REPO_ROOT}/experiments/groot_robocasa/src/build_conceptors.py
echo "=== Rebuilt: \${OPENPI_DATA_HOME}/groot_n15_robocasa_conceptors.npz ==="
python3 -c "
import numpy as np
npz = np.load('${OPENPI_DATA_HOME}/groot_n15_robocasa_conceptors.npz', allow_pickle=False)
tasks = sorted({k.split('__')[0] for k in npz.files if '__' in k})
print(f'Tasks in npz: {tasks}')
print(f'Total keys: {len(npz.files)}')
"
EOF
chmod +x "$REBUILD_SCRIPT"

if ! $DRY_RUN && [[ ${#COLLECT_JIDS[@]} -gt 0 ]]; then
    DEP=$(IFS=: ; echo "${COLLECT_JIDS[*]}")
    REBUILD_JID=$(sbatch \
        --parsable \
        --job-name="grc-rebuild-conceptors" \
        --partition=dgx-b200 \
        --gres=gpu:0 \
        --cpus-per-task=8 \
        --mem=128G \
        --time=4:00:00 \
        --dependency="afterok:${DEP}" \
        --output="${LOG_DIR}/rebuild_conceptors_%j.out" \
        --error="${LOG_DIR}/rebuild_conceptors_%j.err" \
        "$REBUILD_SCRIPT")
    echo ""
    echo "Rebuild job ${REBUILD_JID} scheduled with afterok dependency on [${DEP}]"
fi

echo ""
echo "Collection jobs: ${COLLECT_JIDS[*]:-<dry-run>}"
echo "Logs: ${LOG_DIR}/collect_<TASK>_<JID>.{out,err} + collect_<TASK>_server_<JID>.log"
