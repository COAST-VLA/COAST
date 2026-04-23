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
| `dp_libero_lang_v1` | [`30000`](https://huggingface.co/brandonyang/dp-libero-checkpoints/tree/main/dp_libero_lang_v1/30000) (subdir of [`brandonyang/dp-libero-checkpoints`](https://huggingface.co/brandonyang/dp-libero-checkpoints); training crashed at step 30k, partial run) |

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

### Diffusion Policy baseline (Transformer-Hybrid, PyTorch-only)

`dp_libero` is a Diffusion Policy baseline using the Transformer-Hybrid variant (Chi et al. 2023), vendored from [`robocasa-benchmark/diffusion_policy`](https://github.com/robocasa-benchmark/diffusion_policy) (Apache 2.0) into `src/openpi/models_pytorch/diffusion_policy/vendored/` so the same model class can load the released RoboCasa checkpoint and be trained from scratch on LIBERO. Training uses the PyTorch entry point; eval reuses the openpi server/client flow but `serve_policy.py` **must** be launched with `--pytorch` (DP has no JAX path). Activation collection (`--collect`) is not supported for DP — the collection path is pi0 / pi0-FAST / pi0.5 only.

**Language-conditioned multi-task baseline.** `dp_libero` trains on the `physical-intelligence/libero` dataset with the per-task prompt routed through `ComputeLangEmb` (CLIP ViT-L/14 `text_embeds` projection, 768-d, `padding="max_length", max_length=25`) into a `lang_emb` obs field. The DP model's `VisualCoreLanguageConditioned` branch consumes it both via FiLM modulation of the ResNet18 image features and as a raw concatenated feature — same language-conditioning path as `dp_robocasa` / `dp_metaworld`. CLIP is forced to CPU in the model_transform so each of the `num_workers × world_size` DataLoader workers caches encodings locally without fragmenting GPU memory.

```bash
# 1. Norm stats (once per dataset).
uv run scripts/compute_norm_stats.py --config-name dp_libero

# 2a. Train — single-GPU.
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_pytorch.py dp_libero \
    --exp-name dp_libero_test --overwrite

# 2b. Train — multi-GPU via torchrun (DDP; batch_size is the total across GPUs).
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py dp_libero --exp-name dp_libero_test --overwrite

# 3. Serve the resulting checkpoint. --pytorch is required.
uv run scripts/serve_policy.py --pytorch policy:checkpoint \
    --policy.config=dp_libero \
    --policy.dir=/path/to/checkpoint
```

The `dp_libero_lang_v1` config reproduces the 4× L40 reference training run (`batch_size=256`, `keep_period=10_000`, all other hyperparameters inherited from `dp_libero`). Norm stats are looked up by config name, so run `compute_norm_stats` once for `dp_libero_lang_v1` as well:

```bash
uv run scripts/compute_norm_stats.py --config-name dp_libero_lang_v1

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py dp_libero_lang_v1
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

### pi0.5 training curve (`pi05_libero` @ 2k vs 3k vs 9k)

![Comparison of Mean Performance](figures/compare_means_2000_vs_3000_vs_9000.png)
![Per-task comparison](figures/compare_per_task_2000_vs_3000_vs_9000.png)

Raw numbers: [`figures/results_2000_3000_9000.json`](figures/results_2000_3000_9000.json).

### Diffusion Policy (`dp_libero_lang_v1` @ 30k)

Eval of the step-30000 checkpoint on `libero_10`, 15 episodes per task (150 total). Training crashed at step 30k before the 100k-step target, so this is a partial-run snapshot — not a final head-to-head with the pi0.5 rows above (those are from `physical-intelligence/libero`-trained runs on the same suite). Mean success rate: **0.067 (7%)**.

| task_id | task | success rate |
|---------|------|--------------|
| 3 | KITCHEN_SCENE4 put the black bowl in the bottom drawer of the cabinet and close it | 0.27 (4/15) |
| 2 | KITCHEN_SCENE3 turn on the stove and put the moka pot on it | 0.13 (2/15) |
| 9 | KITCHEN_SCENE6 put the yellow and white mug in the microwave and close it | 0.13 (2/15) |
| 1 | LIVING_ROOM_SCENE2 put both the cream cheese box and the butter in the basket | 0.07 (1/15) |
| 8 | KITCHEN_SCENE8 put both moka pots on the stove | 0.07 (1/15) |
| 0 | LIVING_ROOM_SCENE2 put both the alphabet soup and the tomato sauce in the basket | 0.00 |
| 4 | LIVING_ROOM_SCENE5 put the white mug on the left plate and put the yellow and white mug on the right plate | 0.00 |
| 5 | STUDY_SCENE1 pick up the book and place it in the back compartment of the caddy | 0.00 |
| 6 | LIVING_ROOM_SCENE6 put the white mug on the plate and put the chocolate pudding to the right of the plate | 0.00 |
| 7 | LIVING_ROOM_SCENE1 put both the alphabet soup and the cream cheese box in the basket | 0.00 |

Raw numbers: [`figures/results_dp_libero_30000.json`](figures/results_dp_libero_30000.json).

## Testing

Run from this directory (libero_env Python 3.8 venv). LIBERO env tests need EGL rendering and are marked `manual` (skipped in CI).

```bash
cd examples/libero_env

# Pure-logic tests only (no LIBERO/MuJoCo):
uv run pytest tests/ -v -m "not manual"

# Full suite including env rollouts (GPU + EGL):
MUJOCO_GL=egl uv run pytest tests/ -v
```
