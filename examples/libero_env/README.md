# LIBERO

[LIBERO](https://libero-project.github.io/) is a lifelong-robot-learning benchmark of four task suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`). LIBERO needs Python 3.8, so this example lives in its own venv; the sim runs here and talks to the policy server (which stays in the root venv) over WebSocket.

- `main.py` evaluates one LIBERO task (`--task_suite_name` + `--task_id`).
- `eval_all.py` evaluates every task in one LIBERO suite, launching one `main.py` subprocess per `task_id` for parallel execution.

## Installation

```bash
git submodule update --init --recursive

cd examples/libero_env
uv sync
uv run python setup_libero_config.py
```

`~/.libero/config.yaml` is LIBERO's default config file — it tells LIBERO where to find the benchmark, assets, init states, and datasets. Rerun `setup_libero_config.py` if this checkout moves. If EGL gives MuJoCo rendering issues, use `MUJOCO_GL=glx` instead.

## Dataset & Training

Training uses the [`physical-intelligence/libero`](https://huggingface.co/datasets/physical-intelligence/libero) LeRobot dataset. Compute norm stats once, then train (both from the repo root in the root venv):

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero \
    --exp-name pi05_libero_test \
    --overwrite \
    --num_train_steps 30_000
```

`pi05_libero` and `pi0_fast_libero` are registered in `src/openpi/training/config.py`.

### Released checkpoints

| Config | Checkpoints |
|---|---|
| `pi05_libero`      | [`brandonyang/openpi-libero-2000`](https://huggingface.co/brandonyang/openpi-libero-2000), [`brandonyang/openpi-libero-3000`](https://huggingface.co/brandonyang/openpi-libero-3000), [`brandonyang/openpi-libero-9000`](https://huggingface.co/brandonyang/openpi-libero-9000) |
| `pi0_fast_libero`  | [`brandonyang/pi0fast-libero-checkpoints`](https://huggingface.co/brandonyang/pi0fast-libero-checkpoints) (1000, 2000 steps) |

```bash
# pi0-FAST (pick a step):
hf download brandonyang/pi0fast-libero-checkpoints \
    --include "pi0_fast_libero_b200_bs512/2000/*" \
    --local-dir checkpoints/pi0_fast_libero
```

## Serving the policy

Start the policy server from the repo root (root venv):

```bash
# Default LIBERO policy:
uv run scripts/serve_policy.py --env LIBERO

# Specific checkpoint — pi0.5 (JAX by default; add --pytorch for PyTorch):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=path/to/checkpoint

# pi0-FAST (JAX only — no PyTorch port):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_libero \
    --policy.dir=checkpoints/pi0_fast_libero/pi0_fast_libero_b200_bs512/2000
```

## Evaluation

### Single task

```bash
cd examples/libero_env
# Defaults to --task_suite_name libero_10 --task_id 0; override either/both:
MUJOCO_GL=egl uv run python main.py --task_suite_name libero_spatial --task_id 0
```

Default output: `examples/libero_env/output/<task_suite_name>-task<task_id:02d>/`. Override with `--output_dir`.

### Full suite

`eval_all.py` runs every task in one LIBERO suite by launching one `main.py` subprocess per task_id — each subprocess has its own MuJoCo/EGL context (which is why in-process parallelism isn't possible).

```bash
cd examples/libero_env
# Default suite: libero_10
MUJOCO_GL=egl uv run python eval_all.py

# Another suite, with more episodes per task and a concurrency cap:
MUJOCO_GL=egl uv run python eval_all.py \
    --task_suite_name libero_spatial --num_episodes 15 --num_workers 5

# Sequential execution (inline stack traces on crash):
MUJOCO_GL=egl uv run python eval_all.py --num_workers 1
```

A full run produces a single directory containing everything:

```
examples/libero_env/output/<task_suite_name>/
├── results.json                             # aggregated, incrementally saved
├── parallel_logs/task_NN.log                # per-subprocess stdout + stderr
└── <task_id:02d>-<task_name>/episode_NNN.mp4
```

## Activation collection

LIBERO collects **server-side**: a collection-mode server wraps the policy in `CollectingPolicy` and writes intermediates to the **server's** filesystem while the client runs a rollout. Protocol, output layout, schema per model family, and verification are covered in the canonical reference — see **[`docs/activation_collection.md`](../../docs/activation_collection.md)**.

Server (root venv). pi0.5 collection requires `--pytorch` (forward hooks); pi0-FAST requires JAX:

```bash
# pi0.5 diffusion (PyTorch):
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi05_libero \
    --policy.dir=/path/to/checkpoint

# pi0-FAST autoregressive (JAX):
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi0_fast_libero \
    --policy.dir=checkpoints/pi0_fast_libero/pi0_fast_libero_b200_bs512/2000
```

Client (this venv):

```bash
cd examples/libero_env
MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_spatial --collect --num_workers 5
# or a single task:
MUJOCO_GL=egl uv run python main.py --task_suite_name libero_spatial --task_id 0 --collect
```

Pre-collected datasets:

- [`brandonyang/pi05-libero-activations-v1-2000-15env`](https://huggingface.co/datasets/brandonyang/pi05-libero-activations-v1-2000-15env) — pi0.5, `v1` schema, 2000-step checkpoint, 10 tasks × 15 episodes.
- [`brandonyang/pi0fast-libero-activations-v1-2000-15env`](https://huggingface.co/datasets/brandonyang/pi0fast-libero-activations-v1-2000-15env) — pi0-FAST, `fast_v1` schema, libero_10, 2000-step, 1.1 GB, mean success 0.65.

## Results

![Comparison of Mean Performance](figures/compare_means_2000_vs_3000_vs_9000.png)
![Per-task comparison](figures/compare_per_task_2000_vs_3000_vs_9000.png)
