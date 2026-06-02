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
| `pi0_fast_libero`  | [`1000`](https://huggingface.co/brandonyang/pi0fast-libero-checkpoints/tree/main/pi0_fast_libero_b200_bs512/1000), [`2000`](https://huggingface.co/brandonyang/pi0fast-libero-checkpoints/tree/main/pi0_fast_libero_b200_bs512/2000) (subdirs of [`brandonyang/pi0fast-libero-checkpoints`](https://huggingface.co/brandonyang/pi0fast-libero-checkpoints)) |

## Serving the policy

Start the policy server from the repo root (root venv):

```bash
# pi0.5 (JAX by default; add --pytorch for PyTorch):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=/path/to/checkpoint

# pi0-FAST (JAX only — no PyTorch port):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_libero \
    --policy.dir=/path/to/checkpoint
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

# Another suite, with a concurrency cap (--num_episodes defaults to 15):
MUJOCO_GL=egl uv run python eval_all.py \
    --task_suite_name libero_spatial --num_workers 5

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
    --policy.dir=/path/to/checkpoint
```

Client (this venv):

```bash
cd examples/libero_env

# Single task (main.py defaults --num_episodes=1; bump to 15 for real runs):
MUJOCO_GL=egl uv run python main.py --task_suite_name libero_10 --task_id 0 --collect --num_episodes 15

# Full suite — parallelized across tasks (eval_all.py defaults --num_episodes=15):
MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_10 --collect --num_workers 5
```

Pre-collected datasets:

- [`brandonyang/pi05-libero-activations-v1-2000-15env`](https://huggingface.co/datasets/brandonyang/pi05-libero-activations-v1-2000-15env) — pi0.5, `v1` schema, 2000-step checkpoint, 10 tasks × 15 episodes.
- [`brandonyang/pi0fast-libero-activations-v1-2000-15env`](https://huggingface.co/datasets/brandonyang/pi0fast-libero-activations-v1-2000-15env) — pi0-FAST, `fast_v1` schema, libero_10, 2000-step, 1.1 GB, mean success 0.65.

## Results

![Comparison of Mean Performance](figures/compare_means_2000_vs_3000_vs_9000.png)
![Per-task comparison](figures/compare_per_task_2000_vs_3000_vs_9000.png)

## Testing

Run from this directory (libero_env Python 3.8 venv). LIBERO env tests need EGL rendering and are marked `manual` (skipped in CI).

```bash
cd examples/libero_env

# Pure-logic tests only (no LIBERO/MuJoCo):
uv run pytest tests/ -v -m "not manual"

# Full suite including env rollouts (GPU + EGL):
MUJOCO_GL=egl uv run pytest tests/ -v
```
## Running with Steering

Conceptor-based activation steering nudges the action expert's hidden state
toward the subspace of successful rollouts. The end-user surface is one flag:
`--steer`. Tuned per-task hyperparameters are committed at
`experiments/libero/best_configs.json`; producing new ones is a research task
(see `experiments/libero/README.md`).

### Prereqs

1. Download the conceptor NPZ:

   ```bash
   hf download brandonyang/libero-conceptors libero_conceptors.npz \
       --repo-type dataset --local-dir conceptors/
   ```

2. Start a steering-capable server. `--steer` always requires
   `--conceptor_npz`. For pi0.5, add `--pytorch` because steering uses forward
   hooks. For pi0-fast, do not add `--pytorch`; steering is JAX-only and uses
   fast conceptors built from `token_pre_logits`.

   ```bash
   # pi0.5:
   uv run scripts/serve_policy.py --pytorch --steer \
       --conceptor_npz conceptors/libero_conceptors.npz \
       policy:checkpoint \
       --policy.config pi05_libero --policy.dir checkpoints/coast-libero-2000

   # pi0-FAST:
   uv run scripts/serve_policy.py --steer \
       --conceptor_npz conceptors/pi0fast_libero_conceptors.npz \
       policy:checkpoint \
       --policy.config pi0_fast_libero --policy.dir checkpoints/pi0_fast_libero/pi0_fast_libero_b200_bs512/2000
   ```

   The same `--steer` server happily serves baseline (unsteered) clients too —
   an obs without an `__steering__` key passes straight through.

### Single task, default steering params

```bash
cd examples/libero_env
MUJOCO_GL=egl uv run python main.py \
    --task_suite_name libero_10 --task_id 2 --steer
```

Defaults (duplicated from `src/openpi/serving/steering.py`):
`--steering_layer 11 --steering_alpha 0.1 --steering_beta 0.3 --steering_strategy global`.
For pi0-fast, `--steering_layer` is accepted for wire compatibility but ignored;
the fast NPZ has no layer axis.

### Single task, explicit params

```bash
MUJOCO_GL=egl uv run python main.py \
    --task_suite_name libero_10 --task_id 2 --steer \
    --steering_layer 17 --steering_alpha 0.5 --steering_beta 0.1 \
    --steering_strategy per_step
```

### Full suite, uniform defaults

```bash
MUJOCO_GL=egl uv run python eval_all.py \
    --task_suite_name libero_10 --num_episodes 10 --steer
```

### Full suite, per-task tuned configs

This is the standard way to reproduce tuned steering results:

```bash
MUJOCO_GL=egl uv run python eval_all.py \
    --task_suite_name libero_10 --num_episodes 10 \
    --steer --steering_config ../../experiments/libero/best_configs.json
```

Tasks not present in `best_configs.json` fall back to the config's `defaults`
block if present, or the CLI scalar flags otherwise.

### All steering flags

`main.py`:

| Flag                   | Default   | Notes                                     |
|------------------------|-----------|-------------------------------------------|
| `--steer`              | False     | Toggle steering on                        |
| `--steering_layer`     | 11        | Action-expert transformer layer index; ignored by pi0-fast |
| `--steering_alpha`     | 0.1       | Conceptor aperture                        |
| `--steering_beta`      | 0.3       | Interpolation weight (β=0 is no-op)       |
| `--steering_strategy`  | "global"  | See strategy table below                  |
| `--steering_task`      | None      | Override NPZ task key (default: LIBERO task name) |

`eval_all.py` adds:

| Flag                 | Default | Notes                                            |
|----------------------|---------|--------------------------------------------------|
| `--steering_config`  | None    | Path to `best_configs.json` (per-task overrides) |

### Steering strategies

| Strategy          | Math                                               | Params used                  | Notes |
|-------------------|----------------------------------------------------|------------------------------|-------|
| `global`          | `h' = (1−β)h + β(h @ C_contrastive.T)`             | `layer`, `alpha`, `beta`     | Default. Contrastive conceptor `C_s ∧ NOT(C_f)` at aperture α. |
| `per_step`        | Same as `global` but with a DIFFERENT conceptor by position | pi0.5: `layer`, `beta`; pi0-fast: `alpha`, `beta` | pi0.5 NPZ must contain per_step_0..per_step_9 keys. pi0-fast NPZ must contain per_token_first/mid/last keys. |
| `positive_only`   | `h' = (1−β)h + β(h @ C_success.T)`                 | `layer`, `alpha`, `beta`     | Ablation dropping the `NOT(C_failure)` term. |
| `random_matched`  | Same as `global` but with a random-eigenvector conceptor whose spectrum matches `C_contrastive` | pi0.5: `layer`, `alpha`, `beta`; pi0-fast: `alpha`, `beta` | Control. If this helps, the benefit wasn't from the learned direction. |
| `linear`          | `h' = h + α · v`, where `v` = unit(μ_success − μ_failure) | pi0.5: `layer`, `alpha` (β ignored) | ActAdd-style additive baseline for pi0.5. pi0-fast rejects this strategy unless a separate fast linear implementation is added. |

### Error behavior

- Server started without `--steer` receiving an `__steering__` payload: hard
  error back to the client ("CollectingPolicy requires..."/wrong wrapper).
- `task` not present in the conceptor NPZ: `ValueError` listing available task keys.
- Malformed `best_configs.json`: `eval_all.py` fails before spawning any
  subprocesses and points at the offending field.

### Producing new tuned configs

Not a normal-user task. See `experiments/libero/README.md`.
