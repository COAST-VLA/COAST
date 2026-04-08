# MetaWorld Example

[MetaWorld](https://meta-world.github.io/) is a benchmark of 50 simulated robotic manipulation tasks built on MuJoCo. This directory contains every metaworld-specific entry point: the dataset generator (`generate_dataset.py`), the eval clients (`main.py`, `eval_all.py`), and the in-process activation collectors (`collect_activations.py`, `collect_activations_v2.py`).

## Installation

No separate venv is required.

## Generating the Dataset for Training

`generate_dataset.py` rolls out [MetaWorld's scripted policies](https://github.com/Farama-Foundation/Metaworld) (`metaworld.policies.ENV_POLICY_MAP`) across all ML45 train tasks, records per-step observations and three camera views, and pushes the result to the HuggingFace Hub as a LeRobot dataset.

```bash
MUJOCO_GL=egl uv run examples/metaworld/generate_dataset.py \
    --repo_id <hf-username>/metaworld_ml45 \
    --num_envs 50 \
    --num_episodes 2
```

Use `--help` for common flags.

You must be authenticated with `hf auth login` before running, since the script ends with `dataset.push_to_hub()`.

We have pre-generated the ML45 dataset with 100 demonstrations per task at [`brandonyang/metaworld_ml45`](https://huggingface.co/datasets/brandonyang/metaworld_ml45).

## Training

Compute normalization stats once before the first training run, then launch training. Both commands run from the repo root:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_metaworld

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_metaworld \
    --exp-name pi05_metaworld_test \
    --overwrite \
    --num_train_steps 30_000
```

The `pi05_metaworld` config is registered in `src/openpi/training/config.py`.

We have released two checkpoints trained on `brandonyang/metaworld_ml45`:
- [`brandonyang/openpi-metaworld-5000`](https://huggingface.co/brandonyang/openpi-metaworld-5000)
- [`brandonyang/openpi-metaworld-25000`](https://huggingface.co/brandonyang/openpi-metaworld-25000)

## Evaluation

We evalute with server-client architecture: the policy server hosts the model and serves actions over WebSocket, while the eval clients run the envs and query the policy for actions at each step. 

### Serving the Policy

Serving runs from the repo root and uses the shared `scripts/serve_policy.py` entry point.

#### JAX (default)

```bash
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_metaworld \
    --policy.dir=/path/to/checkpoint
```

#### PyTorch

Add `--pytorch`. The first run patches the transformers library and converts the JAX checkpoint to `model.safetensors` (cached afterwards).

```bash
uv run scripts/serve_policy.py --pytorch policy:checkpoint \
    --policy.config=pi05_metaworld \
    --policy.dir=/path/to/checkpoint
```

**Performance notes (PyTorch backend):**

- **First inference call takes ~6 minutes** due to `torch.compile(mode="max-autotune")` benchmarking Triton kernels. This is a one-time cost per process launch.
- After warmup, inference runs at ~3 calls/sec (comparable to JAX after JIT compilation).
- GPU memory usage is ~70 GB during warmup, settling to ~10 GB for steady-state inference.

### Run Evaluation

### Single task

```bash
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3
```

See common flags with `--help`. 

### All tasks (ML45 split)

```bash
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split test
```

`--split train` evaluates the 45 ML45 train tasks; `--split test` evaluates the 5 held-out test tasks.

## Evaluation Results

### ML45 Train Tasks — `pi05_metaworld_test/5000` (15 envs per task)

**Mean success rate: 74.1%** (500/675 episodes)

| Task | Success | Rate |
|---|---|---|
| button-press-topdown-v3 | 15/15 | 100% |
| button-press-topdown-wall-v3 | 15/15 | 100% |
| button-press-v3 | 15/15 | 100% |
| button-press-wall-v3 | 15/15 | 100% |
| coffee-button-v3 | 15/15 | 100% |
| door-close-v3 | 15/15 | 100% |
| drawer-close-v3 | 15/15 | 100% |
| drawer-open-v3 | 15/15 | 100% |
| faucet-open-v3 | 15/15 | 100% |
| handle-press-side-v3 | 15/15 | 100% |
| handle-press-v3 | 15/15 | 100% |
| peg-unplug-side-v3 | 15/15 | 100% |
| plate-slide-side-v3 | 15/15 | 100% |
| plate-slide-v3 | 15/15 | 100% |
| push-wall-v3 | 15/15 | 100% |
| reach-wall-v3 | 15/15 | 100% |
| window-close-v3 | 15/15 | 100% |
| window-open-v3 | 15/15 | 100% |
| coffee-pull-v3 | 14/15 | 93% |
| door-open-v3 | 14/15 | 93% |
| reach-v3 | 14/15 | 93% |
| faucet-close-v3 | 12/15 | 80% |
| pick-place-v3 | 12/15 | 80% |
| plate-slide-back-side-v3 | 12/15 | 80% |
| push-v3 | 12/15 | 80% |
| shelf-place-v3 | 12/15 | 80% |
| sweep-into-v3 | 12/15 | 80% |
| sweep-v3 | 12/15 | 80% |
| lever-pull-v3 | 11/15 | 73% |
| push-back-v3 | 11/15 | 73% |
| coffee-push-v3 | 10/15 | 67% |
| pick-place-wall-v3 | 10/15 | 67% |
| peg-insert-side-v3 | 9/15 | 60% |
| stick-pull-v3 | 9/15 | 60% |
| disassemble-v3 | 7/15 | 47% |
| handle-pull-v3 | 7/15 | 47% |
| plate-slide-back-v3 | 7/15 | 47% |
| basketball-v3 | 5/15 | 33% |
| hammer-v3 | 4/15 | 27% |
| pick-out-of-hole-v3 | 4/15 | 27% |
| soccer-v3 | 4/15 | 27% |
| assembly-v3 | 3/15 | 20% |
| handle-pull-side-v3 | 2/15 | 13% |
| stick-push-v3 | 1/15 | 7% |
| dial-turn-v3 | 0/15 | 0% |

## Collecting Activations for Mechanistic Interpretability

We provide two activation collection scripts. **V2 is recommended** — it collects richer data (attention weights, adaRMS gates, proprioceptive state) in 65% less storage.

Unlike `examples/libero/` and `examples/robocasa_env/`, metaworld activation collection runs **in-process**: the collection scripts load the policy and the env in the same Python process.

### Downloading Pre-Collected Activations

Pre-collected activation datasets are available on HuggingFace. Download them to local directories using this naming convention: `pi05-metaworld-activations-v{version}-{num_envs}env`.

```bash
# V2 activations (recommended) — 126 GB, 15 envs per task
hf download brandonyang/pi05-metaworld-activations-v2-15env --repo-type dataset --local-dir pi05-metaworld-activations-v2-15env

# V1 activations (15 envs per task) — 357 GB
hf download brandonyang/pi05-metaworld-activations-v1-15env --repo-type dataset --local-dir pi05-metaworld-activations-v1-15env

# V1 activations (2 envs per task) — 20 GB
hf download brandonyang/pi05-metaworld-activations-v1-2env --repo-type dataset --local-dir pi05-metaworld-activations-v1-2env
```

> **Note:** The V2 dataset is uploaded as per-task `.tar` files. After downloading, extract them:
> ```bash
> cd pi05-metaworld-activations-v2-15env
> for f in *.tar; do tar xf "$f"; done
> ```

### V2 Collection

Dataset: [brandonyang/pi05-metaworld-activations-v2-15env](https://huggingface.co/datasets/brandonyang/pi05-metaworld-activations-v2-15env)

```bash
# Single task
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl uv run examples/metaworld/collect_activations_v2.py \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
    --tasks reach-v3 --num_envs 2

# All 45 ML45 train tasks (multi-GPU)
MUJOCO_GL=egl uv run examples/metaworld/collect_activations_v2.py \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
    --split train --num_envs 15 --gpus 1 6
```

Check `--help` for common flags.

#### V2 Output Structure

```
activations_v2/{checkpoint_step}/
  adarms_cond_global.npz                  # Saved ONCE (deterministic across all episodes)
    adarms_cond_global: (num_envs, 1024)  #   1024-dim conditioning vector (all rows identical)
  {task_name}/
    episode_000_env_000/
      metadata.json                       # Episode-level metadata (see below)
      rewards.npz                         # Reward trajectory
        per_step_reward: (300,) float32   #   Per-environment-step reward
        cumulative_reward: (300,) float32 #   Running sum
        success_at_step: (300,) bool      #   Whether task succeeded by this step
      step_0000/
        denoising.npz                     # Denoising trajectory (3 of 10 steps: 0, 4, 9)
          all_x_t: (3, 32, 32) float32   #   Noisy actions at collected denoising steps
          all_v_t: (3, 32, 32) float32   #   Velocity predictions (model output)
        suffix_residual.npz               # Action Expert residual streams
          all_suffix_residual: (3, 2, 32, 1024) float32
            # (denoise_steps, layers[5,11], action_tokens, hidden_dim)
        suffix_mlp_hidden.npz             # Action Expert MLP hidden states
          all_suffix_mlp_hidden: (3, 1, 32, 4096) float32
            # (denoise_steps, layers[11], action_tokens, mlp_dim)
        attention_weights.npz             # Action Expert attention patterns
          all_attention_weights: (3, 2, 8, 32, ~1000) float32
            # (denoise_steps, layers[5,11], heads, action_tokens, prefix_seq_len)
            # Shows which image/language tokens each action token attends to
        adarms_gates.npz                  # Adaptive RMSNorm gate values
          all_adarms_gates: (3, 18, 2, 1, 1024) float32
            # (denoise_steps, all_18_layers, 2[attn_gate,mlp_gate], batch, hidden_dim)
            # Controls per-layer contribution at each denoising timestep
        metadata.json                     # Step-level metadata (see below)
      step_0010/
        ...
```

#### V2 Episode Metadata (`episode_*/metadata.json`)

```json
{
  "task_name": "reach-v3",
  "episode_id": 0,
  "env_id": 0,
  "episode_success": true,
  "total_reward": 2574.66,
  "steps_to_success": 34,
  "total_env_steps": 300,
  "total_inference_steps": 30,
  "prompt": "reach the goal position",
  "checkpoint_dir": "checkpoints/pi05_metaworld/pi05_metaworld_test/5000/",
  "config_name": "pi05_metaworld",
  "collection_version": "v2",
  "collected_denoise_steps": [0, 4, 9],
  "collected_residual_layers": [5, 11],
  "collected_mlp_layers": [11],
  "collected_attention_layers": [5, 11]
}
```

#### V2 Step Metadata (`step_*/metadata.json`)

```json
{
  "task_name": "reach-v3",
  "episode_id": 0,
  "env_id": 0,
  "step": 0,
  "inference_step": 0,
  "prompt": "reach the goal position",
  "cumulative_reward": 0.0,
  "success_so_far": false,
  "reward_since_last_inference": 0.0,
  "proprio_state": [0.005, 0.601, 0.195, 1.0],
  "object_positions": [0.002, 0.683, 0.02],
  "predicted_actions": [-0.04, 1.0, -0.29, 0.006]
}
```

| Field | Description |
|---|---|
| `cumulative_reward` | Total reward accumulated from episode start to this step |
| `success_so_far` | Whether the task has been completed by this step |
| `reward_since_last_inference` | Reward accumulated during the 10 env steps since last inference call |
| `proprio_state` | Robot proprioceptive state: `[hand_x, hand_y, hand_z, gripper_angle]` |
| `object_positions` | Task object position from observation: `[obj_x, obj_y, obj_z]` |
| `predicted_actions` | Actions sent to the environment: `[dx, dy, dz, gripper]` |

#### V2 Storage

~8.7 MB per inference step (vs ~26 MB for V1). Total for 45 tasks × 15 envs: **~126 GB** (vs 357 GB for V1).

#### Validate V2 Activations

```bash
# Validate a single task
ACTIVATIONS_V2_DIR=activations_v2/5000/reach-v3 \
ACTIVATIONS_V2_BASE=activations_v2/5000 \
  uv run pytest tests/test_activations_v2.py -v

# Validate a different task
ACTIVATIONS_V2_DIR=activations_v2/5000/pick-place-v3 \
ACTIVATIONS_V2_BASE=activations_v2/5000 \
  uv run pytest tests/test_activations_v2.py -v
```

---

### V1 Collection (Original)

Datasets:
- [brandonyang/pi05-metaworld-activations-v1-2env](https://huggingface.co/datasets/brandonyang/pi05-metaworld-activations-v1-2env) — 2 envs per task
- [brandonyang/pi05-metaworld-activations-v1-15env](https://huggingface.co/datasets/brandonyang/pi05-metaworld-activations-v1-15env) — 15 envs per task

```bash
# Single task
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl uv run examples/metaworld/collect_activations.py \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
    --tasks reach-v3 --num_envs 2

# All 45 ML45 train tasks
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl uv run examples/metaworld/collect_activations.py \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
    --split train --num_envs 2

# Multi-GPU
MUJOCO_GL=egl uv run examples/metaworld/collect_activations.py \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
    --split train --num_envs 2 --gpus 0 1
```

#### V1 Output Structure

```
activations/{checkpoint_step}/{task_name}/
  episode_000_env_000/
    metadata.json                         # Episode-level metadata
      # task_name, episode_id, env_id, episode_success, total_reward,
      # steps_to_success, total_env_steps, total_inference_steps, prompt,
      # checkpoint_dir, config_name
    rewards.npz                           # Reward trajectory
      per_step_reward: (300,) float32
      cumulative_reward: (300,) float32
      success_at_step: (300,) bool
    step_0000/
      denoising.npz                       # All 10 denoising steps
        all_x_t: (10, 32, 32) float32    #   Noisy action states
        all_v_t: (10, 32, 32) float32    #   Velocity predictions
      adarms_cond.npz                     # Timestep conditioning (per step)
        all_adarms_cond: (10, 1024) float32
      suffix_residual.npz                 # 4 Action Expert layers
        all_suffix_residual: (10, 4, 32, 1024) float32
          # (denoise_steps, layers[0,5,11,17], action_tokens, hidden_dim)
      suffix_mlp_hidden.npz               # 4 Action Expert MLP layers
        all_suffix_mlp_hidden: (10, 4, 32, 4096) float32
          # (denoise_steps, layers[0,5,11,17], action_tokens, mlp_dim)
      metadata.json                       # Step-level metadata
        # task_name, episode_id, env_id, step, inference_step, prompt,
        # cumulative_reward, success_so_far, reward_since_last_inference
    step_0010/
      ...
```

#### V1 Storage

~26 MB per inference step. Total for 45 tasks × 15 envs: **~357 GB**.

#### Validate V1 Activations

The schema validators are env-agnostic — they work for any directory written by these scripts (or by any other env that follows the same on-disk format).

```bash
# Validate a single task
ACTIVATIONS_DIR=activations/5000/reach-v3 uv run pytest tests/test_activations.py -v

# Validate a different task
ACTIVATIONS_DIR=activations/5000/pick-place-v3 uv run pytest tests/test_activations.py -v
```

## Testing

MetaWorld environment tests require a GPU with EGL rendering support. They are marked as `manual` and skipped in CI.

```bash
# Run all MetaWorld tests locally (requires GPU + EGL):
MUJOCO_GL=egl uv run pytest tests/metaworld/test_metaworld_envs.py -v

# Run only pure-logic tests (no GPU / rendering required):
uv run pytest tests/metaworld/test_metaworld_envs.py -v -m "not manual"
```
