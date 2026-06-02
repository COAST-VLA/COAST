# RoboCasa

[RoboCasa](https://robocasa.ai/docs/build/html/index.html) is a kitchen-environment benchmark built on robosuite. Its dependencies conflict with the root `openpi` venv, so this directory is a **separate venv**. The sim runs here and talks to the policy server (in the root venv, or in [`groot_env/`](../../groot_env/README.md)) over WebSocket.

The client is **backend-agnostic** — `main.py` / `eval_all.py` target either a **pi0.5 server** (`scripts/serve_policy.py`) or an **NVIDIA GR00T N1.5 server** (`groot_env/serve.py`) with no client-side changes; just point `--host` / `--port` at whichever is running.

## Installation

Adapted from the [RoboCasa setup guide](https://robocasa.ai/docs/build/html/introduction/installation.html):

```bash
cd examples/robocasa_env
uv sync
uv run python -m robocasa.scripts.setup_macros
uv run python -m robocasa.scripts.download_kitchen_assets   # ~10GB
```

## Dataset & Training

We do not train RoboCasa in-repo — evaluate against upstream checkpoints directly.

```bash
# pi0.5 (root venv):
hf download robocasa/robocasa365_checkpoints \
    --include "pi05_pretrain_human300/multitask_learning/75000/*" --local-dir checkpoints
mkdir -p checkpoints/pi05_pretrain_human300/multitask_learning/75000/assets/robocasa
mv checkpoints/pi05_pretrain_human300/multitask_learning/75000/assets/norm_stats.json \
   checkpoints/pi05_pretrain_human300/multitask_learning/75000/assets/robocasa

# GR00T N1.5 (groot_env venv — see groot_env/README.md for setup):
uv run hf download robocasa/robocasa365_checkpoints \
    --include "gr00t_n1-5/multitask_learning/checkpoint-120000/*" \
    --local-dir checkpoints/groot_n15
```

## Serving the policy

### pi0.5 (root venv)

Add `--pytorch` to use the Torch backend; the first run converts the JAX checkpoint to `model.safetensors` and caches it.

```bash
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_robocasa \
    --policy.dir=checkpoints/pi05_pretrain_human300/multitask_learning/75000
```

### GR00T N1.5 (groot_env venv)

N1.5 pins `torch==2.5.1`, so it lives in its own venv. Full setup in [`groot_env/README.md`](../../groot_env/README.md); minimum to serve:

```bash
cd groot_env && GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install --no-build-isolation flash-attn==2.7.1.post4

export CUDA_VISIBLE_DEVICES=0
uv run python serve.py --port 8000     # defaults to multitask_learning/checkpoint-120000
```

### Camera payload (pi0.5 vs GR00T)

The client always emits three camera keys; each server reads what it needs:

| Key | pi0.5 | GR00T N1.5 |
|---|:-:|:-:|
| `observation/image` (agentview_left) | ✓ | ✓ |
| `observation/image2` (agentview_right) | ignored | ✓ (trained with it) |
| `observation/wrist_image` (eye_in_hand) | ✓ | ✓ |

Same payload, same client, either server.

## Evaluation

Two entry points, identical regardless of backend:

1. **`main.py`** — one task (one `env_name`), current process.
2. **`eval_all.py`** — every task in a task set (`atomic_seen`, `composite_seen`, `composite_unseen`, `pretrain50`), one `main.py` subprocess per env via a `ThreadPoolExecutor`.

RoboCasa can't share EGL contexts across threads in one process, so in-process parallelism isn't possible; `eval_all.py` gives each task its own subprocess. Env stepping is ~400ms/step, so 5–10 workers gives a real wall-clock win. `--num_workers 1` runs sequentially with inline tracebacks.

### Single task

```bash
cd examples/robocasa_env
MUJOCO_GL=egl uv run python main.py --env_name CloseBlenderLid
```

Default output: `examples/robocasa_env/output/<env_name>/`. Override with `--output_dir`.

### Full task set

```bash
cd examples/robocasa_env

# Curated 7-task subset (default; the tasks we have published pi0.5 + GR00T N1.5
# results for — faster iteration than the full 18-task atomic_seen set).
# eval_all.py defaults --num_episodes=15; drop to --num_episodes 1 for smoke tests.
MUJOCO_GL=egl uv run python eval_all.py --num_workers 5

# A full RoboCasa task set (see TASK_SET_REGISTRY: atomic_seen, composite_seen,
# composite_unseen, target50, pretrain50 / 100 / 200 / 300)
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen --num_workers 5

# Explicit task list (overrides --task_set)
MUJOCO_GL=egl uv run python eval_all.py --tasks OpenDrawer CloseFridge
```

Default output layout (default `output/<task_set>-<split>/`, override with `--output_dir`):

```
<output_dir>/
├── results.json                        # per-task + mean success rate, written incrementally
├── parallel_logs/task_NN_<env>.log     # per-subprocess stdout/stderr
└── <env_name>/episode_NNN.mp4          # per-episode video (tiles agentview_left/right + eye_in_hand)
```

## Activation collection

RoboCasa collects **server-side** — the client's `--collect` flag is identical for both pi0.5 and GR00T backends (same wire protocol, same `CollectionSession` helper); only the server command differs. Protocol, output layout, schema per model family, and verification are covered in the canonical reference — see **[`docs/activation_collection.md`](../../docs/activation_collection.md)**.

Client (either backend):

```bash
cd examples/robocasa_env

# Single task (main.py defaults --num_episodes=1; bump to 15 for real runs):
MUJOCO_GL=egl uv run python main.py --env_name CloseBlenderLid --collect --num_episodes 15

# Full task set (eval_all.py defaults --num_episodes=15):
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen --collect --num_workers 5
```

Server — pick one:

```bash
# pi0.5 (root venv). --pytorch required (forward hooks are Torch-only).
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi05_robocasa \
    --policy.dir=checkpoints/pi05_pretrain_human300/multitask_learning/75000

# GR00T (groot_env venv). serve.py is PyTorch-only; no --pytorch flag exists.
export CUDA_VISIBLE_DEVICES=0
cd groot_env && uv run python serve.py --port 8000 --collect_activations \
    --model-path /path/to/checkpoint \
    --output-dir ../activations
```

Pre-collected datasets (both cover the same 7 tasks × 15 episodes: `CloseFridge`, `CoffeeSetupMug`, `OpenDrawer`, `OpenStandMixerHead`, `PickPlaceCounterToCabinet`, `PickPlaceCounterToStove`, `TurnOnElectricKettle`):

- [`ksb21st/robocasa-activations-75000`](https://huggingface.co/datasets/ksb21st/robocasa-activations-75000) — pi0.5, `v1` schema, sourced from [`pi05_pretrain_human300/multitask_learning/75000`](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi05_pretrain_human300/multitask_learning/75000).
- [`brandonyang/groot_n15-robocasa-activations-v1-15env`](https://huggingface.co/datasets/brandonyang/groot_n15-robocasa-activations-v1-15env) — GR00T N1.5, `groot_v1` schema, sourced from [`gr00t_n1-5/multitask_learning/checkpoint-120000`](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/gr00t_n1-5/multitask_learning/checkpoint-120000).

### Verifying collected activations

Env-var-driven pytest suite (skipped in CI when `ACTIVATIONS_DIR` is unset):

```bash
# pi0.5 (repo root):
ACTIVATIONS_DIR=./activations/75000/CloseBlenderLid \
    uv run pytest tests/test_activations.py -v

# GR00T N1.5 (groot_env):
cd groot_env
ACTIVATIONS_DIR=../activations/groot_n15-robocasa-activations-v1-15env/checkpoint-120000/OpenDrawer \
    uv run pytest tests/test_groot_activations.py -v
```

## Results

### pi0.5 (15 ep/task, `pi05_pretrain_human300/multitask_learning/75000`, `pretrain` split)

Raw numbers: [`figures/results_75000.json`](figures/results_75000.json). Upstream RoboCasa numbers: [multitask_learning page](https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html#benchmark-results-and-checkpoints).

![Mean success rate per task set](figures/compare_means_75000.png)
![Per-task success rates](figures/compare_per_task_75000.png)

### GR00T N1.5

Published comparisons: https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html

## Testing

Run from this directory (robocasa_env venv). RoboCasa env tests need EGL rendering and are marked `manual` (skipped in CI).

```bash
cd examples/robocasa_env

# Pure-logic tests only (no RoboCasa/MuJoCo):
uv run pytest tests/ -v -m "not manual"

# Full suite including env rollouts (GPU + EGL):
MUJOCO_GL=egl uv run pytest tests/ -v
```

Each `eval_all.py` subprocess creates its own `CollectionSession` keyed on its distinct `env_name`, so the shared collection-mode server writes activations to disjoint output directories with no cross-subprocess coordination. The server's single-threaded asyncio dispatch serializes the underlying hook-based `infer_with_intermediates` call automatically, and `CollectingPolicy`'s explicit lock documents the invariant for future executor-based optimizations.

Notes:
- Collection mode requires `--pytorch` on the server. `infer_with_intermediates`
  is implemented for the PyTorch backend only.
- A collection-mode server **rejects** plain inference requests. If you want to
  also run regular eval, start a separate non-collection server on a different
  port.
- The server's `--output-dir` is on the **server's** filesystem. With
  `--output-dir ./activations`, files land at
  `./activations/<checkpoint_step>/<env_name>/episode_NNN_env_000/step_NNNN/`
  relative to wherever the server was launched from.
- The robocasa client uses `env_name` (e.g. `CloseBlenderLid`) as the
  `task_name` in the collection metadata. The `task_id` field is fixed at 0
  since each robocasa env is its own standalone task. The `episode_id` cycles
  through `0..num_episodes-1` per env.
- See `examples/libero_env/README.md` (the **Protocol** section under
  Activation Collection) for the full wire-level spec of the `__collect__`
  and `__finalize_episode__` payloads. The same `openpi_client.collection_session.CollectionSession`
  helper handles the bookkeeping for libero, robocasa, and any future client.

## Running with Steering

Same flag surface as libero — the end-user entry point is `--steer`. The NPZ
key is the RoboCasa env name directly (e.g., `CloseFridge`). RoboCasa steering
is supported for `pi05_robocasa` and GR00T N1.5. There is no
`pi0_fast_robocasa` config/checkpoint path in this branch.

### Prereqs

```bash
hf download brandonyang/robocasa-conceptors robocasa_conceptors.npz \
    --repo-type dataset --local-dir conceptors/

# pi0.5 server (from repo root)
uv run scripts/serve_policy.py --pytorch --steer \
    --conceptor_npz conceptors/robocasa_conceptors.npz \
    policy:checkpoint \
    --policy.config pi05_robocasa \
    --policy.dir checkpoints/pi05_pretrain_human300/multitask_learning/75000

# GR00T conceptors are built from groot_v1 activations.
uv run python experiments/robocasa/compute_groot_conceptors.py \
    --activation_root activations/groot_n15-robocasa-activations-v1-15env \
    --output_path conceptors/groot_robocasa_conceptors.npz

# GR00T server (from groot_env/)
cd groot_env
uv run python serve.py --port 8000 --steer \
    --model-path ../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000 \
    --conceptor-npz ../conceptors/groot_robocasa_conceptors.npz
```

### Single env, default steering

```bash
cd examples/robocasa_env
MUJOCO_GL=egl uv run python main.py --env_name CloseFridge --steer
```

### Single env, explicit params

```bash
MUJOCO_GL=egl uv run python main.py --env_name CloseFridge --steer \
    --steering_layer 11 --steering_alpha 0.5 --steering_beta 0.1 \
    --steering_strategy per_step
```

### Full task_set with per-env tuned configs

```bash
MUJOCO_GL=egl uv run python eval_all.py \
    --task_set atomic_seen --num_episodes 10 \
    --steer --steering_config ../../experiments/robocasa/best_configs.json
```

Flag names match libero; see `examples/libero_env/README.md#running-with-steering`
for the full table. The only difference is the NPZ task-key source: robocasa
uses `args.env_name` directly, so `--steering_task` is rarely needed.
For GR00T, `per_step` expects `per_step_0..per_step_3` conceptors by default
because `groot_env/serve.py` runs 4 denoising steps unless you override
`--denoising-steps`.

To produce new tuned configs, see `experiments/robocasa/README.md`.
