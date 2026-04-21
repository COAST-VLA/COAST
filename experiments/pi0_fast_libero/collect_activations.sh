#!/bin/bash
#SBATCH --job-name=fast-lib-collect
#SBATCH --partition=dgx-b200
#SBATCH --account=ungar-stellar
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem=224G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/pi0_fast_libero/collect_logs/collect_%j.out
#SBATCH --error=experiments/pi0_fast_libero/collect_logs/collect_%j.err

set -e
export OPENPI_DATA_HOME=/vast/projects/ungar/stellar/miaom/.cache/openpi
export MUJOCO_GL=egl
cd /vast/projects/ungar/stellar/miaom/openpi-new

# --- B200 / JAX tuning (single GPU) -------------------------------------
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Redirect XLA PTX cache off /tmp to avoid "No space left on device".
export TMPDIR="/vast/projects/ungar/stellar/miaom/.cache/openpi/tmp_xla"
mkdir -p "$TMPDIR"

echo "=== pi0-fast LIBERO-10 activation collection ==="
echo "Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?} | Mem: ${SLURM_MEM_PER_NODE:-?} MB"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo "Start: $(date)"

# --- Step 1: Start the pi0-fast collection server in the background ---
echo "Starting pi0-fast collection server..."
uv run scripts/serve_policy.py --collect_activations \
    --output-dir /vast/projects/ungar/stellar/miaom/.cache/openpi/activations_fast_libero \
    --port 8000 \
    policy:checkpoint \
    --policy.config=pi0_fast_libero \
    --policy.dir=checkpoints/pi0_fast_libero/pi0_fast_libero_b200_bs512/1000/ &
SERVER_PID=$!

# Wait for server to be ready (poll the port)
echo "Waiting for server (PID=$SERVER_PID) to be ready on port 8000..."
for i in $(seq 1 120); do
    if python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost', 8000)); s.close()" 2>/dev/null; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server exited prematurely"
        exit 1
    fi
    sleep 1
done

# --- Step 2: Run LIBERO client with --collect for libero_10 ---
echo "Running LIBERO-10 evaluation with activation collection..."
cd examples/libero_env
MUJOCO_GL=egl uv run python eval_all.py \
    --task_suite_name libero_10 \
    --collect \
    --resume \
    --activation_dir /vast/projects/ungar/stellar/miaom/.cache/openpi/activations_fast_libero \
    --num_episodes 15 \
    --num_workers 10 \
    --port 8000

echo "LIBERO-10 collection complete."

# --- Step 3: Clean up ---
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "=== Done: $(date) ==="
