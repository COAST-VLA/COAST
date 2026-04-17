# MetaWorld Example

[MetaWorld](https://meta-world.github.io/) is a benchmark of 50 simulated robotic manipulation tasks built on MuJoCo. This directory contains every metaworld-specific entry point: the dataset generator (`generate_dataset.py`) and the eval clients (`main.py`, `eval_all.py` — both support an optional `--collect` flag for in-process activation collection).

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

You must be authenticated with `hf auth login` before running, since the script ends with `dataset.push_to_hub()`.

We have pre-generated the ML45 dataset with ~100 demonstrations per task at [`brandonyang/metaworld_ml45`](https://huggingface.co/datasets/brandonyang/metaworld_ml45).

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

We have released two checkpoints trained with the following config:
```python
TrainConfig(
    name="pi05_metaworld",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=32, discrete_state_input=False),
    data=LeRobotMetaworldDataConfig(
        repo_id="brandonyang/metaworld_ml45",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=128,  # 256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,
        decay_steps=29_000,
        decay_lr=5e-6,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    pytorch_weight_path="/path/to/your/pytorch_weight_path",
    num_train_steps=30_000,
),
```

- [`brandonyang/openpi-metaworld-5000`](https://huggingface.co/brandonyang/openpi-metaworld-5000)
- [`brandonyang/openpi-metaworld-25000`](https://huggingface.co/brandonyang/openpi-metaworld-25000)

## Evaluation

Normal evaluation uses a server-client architecture: `scripts/serve_policy.py` hosts the model and serves actions over WebSocket; `main.py` / `eval_all.py` run the envs and query the server at each step. Both run from the repo root.

### Download a checkpoint

```bash
hf download brandonyang/openpi-metaworld-5000 --local-dir checkpoints/openpi-metaworld-5000
# also available: brandonyang/openpi-metaworld-25000
```

### Serve the policy (Terminal 1)

```bash
export CUDA_VISIBLE_DEVICES=0

# JAX (default):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/openpi-metaworld-5000

# PyTorch (add --pytorch; first run auto-converts the JAX checkpoint to model.safetensors):
uv run scripts/serve_policy.py --pytorch policy:checkpoint \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/openpi-metaworld-5000
```

### Run evaluation (Terminal 2)

**Single task** — `main.py`, runs one `env_name` with `num_envs` parallel envs:

```bash
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3 --num_envs 16
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3 --output_dir /tmp/reach_debug
```

**Full eval** — `eval_all.py`, sweeps all tasks in the ML45 split (train=45 tasks, test=5 held-out tasks):

```bash
# All 45 train tasks
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train

# All 5 held-out test tasks
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split test

# Subset of tasks (skips --split)
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --tasks reach-v3 push-v3 pick-place-v3

# Custom output directory (results.json, per-task videos)
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train --output_dir /tmp/ml45_run1

# Larger batch per task
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train --num_envs 16
```

`--output_dir` follows the same contract as the libero and robocasa examples: if set, every artifact (`results.json`, per-task video dirs) lands directly under that path; if omitted, the default is `examples/metaworld/output/ML45-{split}/`. `eval_all.py` writes `results.json` incrementally after each task so progress isn't lost on early exit.

### Parallelism model

Unlike the libero and robocasa examples, metaworld uses **in-process** parallelism via `gym.vector.AsyncVectorEnv(context="spawn")` rather than spawning one subprocess per task. The `--num_envs N` flag (default 15 in `eval_all.py`, 10 in `main.py`) runs N parallel envs of the same task inside a single process and batches their observations into one policy call. This is more efficient for metaworld because:

- Metaworld's MuJoCo + EGL setup is multiprocess-safe in this configuration (see PR #9 — `AsyncVectorEnv(spawn)` gave an 8.6× speedup over sequential env stepping)
- Batched inference amortizes the policy call cost across all envs in a task
- Metaworld env steps are fast (~5 ms), so the WebSocket policy server becomes the bottleneck well before subprocess-level parallelism would help

Libero and robocasa cannot use this approach because their envs can't share EGL contexts in-process, so they fall back to one subprocess per task (see `examples/libero_env/eval_all.py` and `examples/robocasa_env/eval_all.py`, both of which expose a `--num_workers` flag). Metaworld does not have a `--num_workers` flag and does not need one — tune `--num_envs` instead.

## Evaluation Results

![Comparison](figures/compare_means_5000_vs_25000.png)
![Comparison Per Task](figures/compare_per_task_5000_vs_25000.png)

## Collecting Activations for Mechanistic Interpretability

The same `main.py` and `eval_all.py` accept a `--collect` flag. Unlike the libero and robocasa examples (which route activations through a dedicated collection server), metaworld activation collection runs **in-process**: when `--collect` is set, the script loads the PyTorch policy directly, bypassing the WebSocket server, and saves intermediate activations during rollout. This preserves metaworld's batched `AsyncVectorEnv` inference and enables per-GPU task sharding via `eval_all.py --gpus`.

**No server needed for `--collect`** — do not run `serve_policy.py`. The script loads the policy itself from `--policy.dir`. (The first run auto-converts the JAX checkpoint to `model.safetensors` via `ensure_pytorch_checkpoint`; in multi-GPU mode the conversion happens once in the parent before any subprocess spawns, so concurrent jobs don't race.)

Activations are written to `--collect_output_dir` (default `./activations`) using the same on-disk schema as libero and robocasa collection. Videos and `results.json` are still written to the eval artifact directory (`--output_dir`), unaffected by `--collect`. `--gpus` is only valid together with `--collect`; for normal (WebSocket-served) eval, pin a single GPU with `CUDA_VISIBLE_DEVICES`.

### Downloading Pre-Collected Activations

Pre-collected activation datasets are available on HuggingFace if you want to skip collection:

```bash
# 15 envs per task — 357 GB
hf download brandonyang/pi05-metaworld-activations-v1-15env --repo-type dataset --local-dir pi05-metaworld-activations-v1-15env

# 2 envs per task — 20 GB
hf download brandonyang/pi05-metaworld-activations-v1-2env --repo-type dataset --local-dir pi05-metaworld-activations-v1-2env
```

### Running Collection

**Single task** — `main.py --collect`:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/metaworld/main.py \
    --collect --env_name reach-v3 --num_envs 16 \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/openpi-metaworld-5000
```

**Full collection** — `eval_all.py --collect`, sweeps all 45 (or 5) tasks:

```bash
# All 45 ML45 train tasks, single GPU
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --collect --split train --num_envs 16 \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/openpi-metaworld-5000

# Test split (5 held-out tasks)
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --collect --split test --num_envs 16 \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/openpi-metaworld-5000

# Subset of tasks
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --collect --tasks reach-v3 push-v3 pick-place-v3 --num_envs 16 \
    --policy.dir=checkpoints/openpi-metaworld-5000

# Multi-GPU: round-robin task sharding across GPUs 0 and 1
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --collect --split train --num_envs 16 --gpus 0 1 \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/openpi-metaworld-5000

# Custom activation root
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --collect --split train --num_envs 16 \
    --collect_output_dir /scratch/my_activations \
    --policy.dir=checkpoints/openpi-metaworld-5000
```

### Tuning `--num_envs`

`--num_envs 16` is a good default for a 46 GB GPU with the 5000-step checkpoint. Each env contributes a batch element to the captured activation tensors; memory pressure scales roughly linearly with `num_envs`. If you OOM, drop to 8 or 4. The per-inference-step memory peak is dominated by the captured `suffix_mlp_hidden` tensor (shape `(10, 4, num_envs, 32, 4096)` float32) plus the attention kernels during the denoising loop.

`--collect` automatically sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX doesn't grab most of the GPU on import (openpi pulls in JAX even in PyTorch-inference mode). If you want to override this, export the env var yourself before launching.

### Output Structure

```
{collect_output_dir}/{checkpoint_step}/{task_name}/
  episode_NNN_env_NNN/
    metadata.json                         # Episode-level metadata
      # task_name, episode_id, env_id, episode_success, total_reward,
      # steps_to_success, total_env_steps, total_inference_steps, prompt,
      # checkpoint_dir, config_name
    rewards.npz                           # Reward trajectory
      per_step_reward: (N,) float32
      cumulative_reward: (N,) float32
      success_at_step: (N,) bool
    step_NNNN/
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
    step_NNNN+replan_steps/
      ...
```

Storage: ~26 MB per inference step. Total for 45 tasks × 15 envs: **~357 GB**.

### Validate Activations

The schema validators are env-agnostic — they work on any directory matching the layout above.

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
