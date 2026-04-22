# MetaWorld

[MetaWorld](https://meta-world.github.io/) is a benchmark of 50 simulated robotic manipulation tasks built on MuJoCo. This directory contains every metaworld-specific entry point: the dataset generator (`generate_dataset.py`) and the eval clients (`main.py`, `eval_all.py` — both support `--collect` for in-process activation collection).

## Installation

MetaWorld uses the **root openpi venv** — no separate environment needed. Initialize submodules and sync once:

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

## Dataset & Training

`generate_dataset.py` rolls out [MetaWorld's scripted policies](https://github.com/Farama-Foundation/Metaworld) (`metaworld.policies.ENV_POLICY_MAP`) across all ML45 train tasks, records per-step observations and three camera views, and pushes the result to the HuggingFace Hub as a LeRobot dataset.

```bash
MUJOCO_GL=egl uv run examples/metaworld/generate_dataset.py \
    --repo_id <hf-username>/metaworld_ml45 \
    --num_envs 50 \
    --num_episodes 2
```

Log in with `hf auth login` first — the script ends with `dataset.push_to_hub()`. A pre-generated ML45 dataset with ~100 demonstrations per task is available at [`brandonyang/metaworld_ml45`](https://huggingface.co/datasets/brandonyang/metaworld_ml45).

Compute norm stats once, then train (both from the repo root):

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_metaworld

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_metaworld \
    --exp-name pi05_metaworld_test \
    --overwrite \
    --num_train_steps 30_000
```

The `pi05_metaworld` and `pi0_fast_metaworld` configs are registered in `src/openpi/training/config.py`.

### Released checkpoints

| Config | Checkpoint |
|---|---|
| `pi05_metaworld`       | [`brandonyang/openpi-metaworld-5000`](https://huggingface.co/brandonyang/openpi-metaworld-5000), [`brandonyang/openpi-metaworld-25000`](https://huggingface.co/brandonyang/openpi-metaworld-25000) |
| `pi0_fast_metaworld`   | [`1000`](https://huggingface.co/brandonyang/pi0fast-metaworld-checkpoints/tree/main/pi0_fast_metaworld_b200_bs512/1000), [`2000`](https://huggingface.co/brandonyang/pi0fast-metaworld-checkpoints/tree/main/pi0_fast_metaworld_b200_bs512/2000), [`2500`](https://huggingface.co/brandonyang/pi0fast-metaworld-checkpoints/tree/main/pi0_fast_metaworld_b200_bs512/2500) (subdirs of [`brandonyang/pi0fast-metaworld-checkpoints`](https://huggingface.co/brandonyang/pi0fast-metaworld-checkpoints)) |

## Serving the policy

Normal evaluation uses a server-client architecture: `scripts/serve_policy.py` hosts the model and serves actions over WebSocket; the clients below query it each step. Run from the repo root:

```bash
export CUDA_VISIBLE_DEVICES=0

# pi0.5 (JAX by default; add --pytorch for the PyTorch backend):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_metaworld \
    --policy.dir=/path/to/checkpoint

# pi0-FAST (JAX only — no PyTorch port):
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_metaworld \
    --policy.dir=/path/to/checkpoint
```

### Diffusion Policy baseline (Transformer-Hybrid, PyTorch-only)

`dp_metaworld` is a Diffusion Policy baseline using the Transformer-Hybrid variant from Chi et al. 2023, vendored from [`robocasa-benchmark/diffusion_policy`](https://github.com/robocasa-benchmark/diffusion_policy) (Apache 2.0) into `src/openpi/models_pytorch/diffusion_policy/vendored/` so the same model class can load the released RoboCasa checkpoint and be trained from scratch on MetaWorld. Training uses the PyTorch entry point; eval reuses the openpi server/client flow but `serve_policy.py` **must** be launched with `--pytorch` (DP has no JAX path). Activation collection (`--collect`) is not supported for DP — the collection path is pi0 / pi0-FAST / pi0.5 only.

**Language-conditioned multi-task baseline.** `dp_metaworld` trains on all 44 ML45 tasks with the per-task prompt routed through `ComputeLangEmb` (CLIP ViT-L/14 `text_embeds` projection, 768-d, `padding="max_length", max_length=25`) into a `lang_emb` obs field. The DP model's `VisualCoreLanguageConditioned` branch consumes it both via FiLM modulation of the ResNet18 image features and as a raw concatenated feature — same language-conditioning path as `dp_robocasa`. CLIP is forced to CPU in the model_transform so each of the `num_workers × world_size` DataLoader workers caches encodings locally without fragmenting GPU memory.

```bash
# 1. Norm stats (once per dataset).
uv run scripts/compute_norm_stats.py --config-name dp_metaworld

# 2a. Train — single-GPU.
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_pytorch.py dp_metaworld \
    --exp-name dp_metaworld_test --overwrite

# 2b. Train — multi-GPU via torchrun (DDP; batch_size is the total across GPUs).
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py dp_metaworld --exp-name dp_metaworld_test --overwrite

# 3. Serve the resulting checkpoint. --pytorch is required.
uv run scripts/serve_policy.py --pytorch policy:checkpoint \
    --policy.config=dp_metaworld \
    --policy.dir=/path/to/checkpoint
```

The `dp_metaworld_lang_v1` config reproduces the 4× L40 reference training run (`batch_size=256`, `keep_period=10_000`, all other hyperparameters inherited from `dp_metaworld`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py dp_metaworld_lang_v1
```

## Evaluation

MetaWorld parallelizes **in-process**: `--num_envs N` runs N envs of the same task in one process and batches their observations into a single policy call. (This differs from LIBERO / RoboCasa, which spawn one subprocess per task because their envs can't share an EGL context in-process.) Tune `--num_envs` to trade off batch efficiency against memory.

### Single task

```bash
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3
```

Default output: `examples/metaworld/output/<env_name>/`. Override with `--output_dir`.

### Full sweep

Pick a split or a task subset:

```bash
# Curated 26-task subset (default; the tasks with the most success-rate
# variation across training checkpoints)
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split subset

# ML45 train split (45 tasks) or test split (5 held-out tasks)
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split test

# Specific tasks (overrides --split)
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --tasks reach-v3 push-v3 pick-place-v3
```

Default output: `examples/metaworld/output/ML45-<split>/`. `results.json` is written incrementally after each task.

## Activation collection

MetaWorld collects **in-process** (no server needed): `--collect` makes the script load the policy directly from `--policy.dir` and write intermediates to `--collect_output_dir`. Schema, output layout, and verification are covered in the canonical reference — see **[`docs/activation_collection.md`](../../docs/activation_collection.md)**.

```bash
# Single task — pi0.5 (PyTorch auto-detected):
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/metaworld/main.py \
    --collect --env_name reach-v3 --num_envs 16 \
    --policy.config=pi05_metaworld \
    --policy.dir=/path/to/checkpoint \
    --collect_output_dir ./activations

# Full sweep — pi0-FAST (JAX):
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --collect --split subset --num_envs 16 \
    --policy.config=pi0_fast_metaworld \
    --policy.dir=/path/to/checkpoint \
    --collect_output_dir ./activations
```

Start `--num_envs` at 16 and halve it if you OOM — memory scales linearly.

Pre-collected datasets:

- [`brandonyang/pi05-metaworld-activations-v1-ml45train-16env`](https://huggingface.co/datasets/brandonyang/pi05-metaworld-activations-v1-ml45train-16env) — pi0.5, `v1` schema, 16 envs × 45 ML45-train tasks.
- [`brandonyang/pi0fast-metaworld-activations-v1-ml45train-16env`](https://huggingface.co/datasets/brandonyang/pi0fast-metaworld-activations-v1-ml45train-16env) — pi0-FAST, `fast_v1` schema, 16 envs × 45 ML45-train tasks.

## Results

Mean success rate and per-task comparisons across released checkpoints:

![Comparison](figures/compare_means_5000_vs_25000.png)
![Comparison Per Task](figures/compare_per_task_5000_vs_25000.png)

## Testing

Run from the repo root (root venv). MetaWorld env tests need EGL rendering and are marked `manual` (GPU-required, skipped in CI).

```bash
# Pure-logic tests only (no GPU):
uv run pytest tests/metaworld/ -v -m "not manual"

# Full suite including env rollouts (GPU + EGL):
MUJOCO_GL=egl uv run pytest tests/metaworld/ -v
```
