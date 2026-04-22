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

### Diffusion Policy (released checkpoint — use upstream's repo, not ours)

The Transformer-Hybrid DP architecture is vendored from [`robocasa-benchmark/diffusion_policy`](https://github.com/robocasa-benchmark/diffusion_policy) (Apache 2.0) under `src/openpi/models_pytorch/diffusion_policy/vendored/` and used here only for **training** the DP baseline on MetaWorld / LIBERO (see those READMEs). We verify the port's bit-compatibility with the released RoboCasa `pretrain_human300` checkpoint via `tests/robocasa/test_dp_e2e.py::test_dp_robocasa_loads_ckpt_and_samples_actions` — strict load of 589 keys / 106M params, 0 missing / 0 unexpected.

**This repo does not ship a DP inference server for RoboCasa.** We tried; the mechanical pipeline worked (actions flowed, envs stepped) but eval success was 0/N against the released checkpoint even after porting upstream's text encoder (`CLIPTextModelWithProjection` + `text_embeds` + `max_length=25`), action slice (`action_pred[:, n_obs_steps-1:...]`), obs stacking (zero-padded first step), state slice (`pos[0:3], quat[3:7], gripper[14:16]`), image preprocessing, and env seed 1111111. Further debugging requires a live per-step action diff against upstream's own `eval_robocasa.py` — which needs the `NVlabs/sage` robomimic fork + Python 3.9 + PyTorch 1.12 pin. Rather than maintain that adapter here, the recommended path is:

```bash
# Clone upstream in a separate repo (NOT a submodule of openpi-metaworld)
git clone https://github.com/robocasa-benchmark/diffusion_policy ~/diffusion_policy_robocasa
cd ~/diffusion_policy_robocasa && pip install -e .  # follow their conda_environment.yaml
# Then run their eval directly against the released .ckpt:
python eval_robocasa.py --checkpoint <ckpt_path> --task_set atomic_seen --split pretrain
```

The `dp_robocasa` **config** in `src/openpi/training/config.py` stays in the repo — its defaults match the released checkpoint's architecture exactly, which is what makes the strict-load test in `tests/robocasa/test_dp_e2e.py` possible.

DP training is **not** supported for `dp_robocasa` in-repo (we don't host the RoboCasa LeRobot dataset pipeline); that config exists only as the inference-side architecture spec that matches the released `.ckpt`.

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
