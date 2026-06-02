# MetaWorld

[MetaWorld](https://meta-world.github.io/) is a benchmark of 50 simulated robotic manipulation tasks built on MuJoCo. This directory contains every metaworld-specific entry point: the dataset generator (`generate_dataset.py`) and the eval clients (`main.py`, `eval_all.py` — both support `--collect` to forward activation-collection metadata to a `--collect_activations` server).

## Installation

MetaWorld uses the **root COAST venv** — no separate environment needed. Initialize submodules and sync once:

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
| `pi05_metaworld`       | `5000`, `25000` intermediate checkpoints. Download into local COAST directories such as `checkpoints/coast-metaworld-5000` and `checkpoints/coast-metaworld-25000`. |
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

MetaWorld uses the same server-side collection pattern as LIBERO / RoboCasa — start a `--collect_activations` policy server, then run the client with `--collect`. The metaworld client batches `num_envs` parallel rollouts into one inference call and sends a list-shaped `__collect__` payload (one entry per env), so the server saves N step dirs from a single forward pass. Schema, output layout, and verification are covered in the canonical reference — see **[`docs/activation_collection.md`](../../docs/activation_collection.md)**.

```bash
# Terminal 1 — pi0.5 (PyTorch required for diffusion intermediates):
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi05_metaworld \
    --policy.dir=/path/to/checkpoint

# Terminal 1 (alternative) — pi0-FAST (JAX only):
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi0_fast_metaworld \
    --policy.dir=/path/to/checkpoint

# Terminal 2 — single task (16 envs in one forward pass):
MUJOCO_GL=egl uv run examples/metaworld/main.py \
    --collect --env_name reach-v3 --num_envs 16

# Terminal 2 — full sweep:
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --collect --split subset --num_envs 16
```

Start `--num_envs` at 16 and halve it if the server OOMs — server-side memory scales linearly with batch size.

Pre-collected datasets:

- [`brandonyang/pi05-metaworld-activations-v1-ml45train-16env`](https://huggingface.co/datasets/brandonyang/pi05-metaworld-activations-v1-ml45train-16env) — pi0.5, `v1` schema, 16 envs × 45 ML45-train tasks.
- [`brandonyang/pi0fast-metaworld-activations-v1-ml45train-16env`](https://huggingface.co/datasets/brandonyang/pi0fast-metaworld-activations-v1-ml45train-16env) — pi0-FAST, `fast_v1` schema, 16 envs × 45 ML45-train tasks.

## Running with Steering

Conceptor-based activation steering nudges the action expert's hidden state toward the subspace of successful rollouts. End-user surface is one flag: `--steer`. Tuned per-task hyperparameters (once produced by the research sweep) live at `experiments/metaworld/best_configs.json`; producing new ones is a research task (see `experiments/metaworld/README.md`).

**Only the steering WebSocket server supports steering** — `--collect` targets a collection-mode server, while `--steer` targets a steering-mode server. Run collection and steering as separate passes, each with the matching server mode.

Both `pi05_metaworld` and `pi0_fast_metaworld` are supported. pi0.5 steering
uses PyTorch hooks on denoising activations. pi0-fast steering is JAX-only and
applies conceptor matrices to autoregressive token `pre_logits` before the LM
head, using Miranda-v2-style fast conceptor keys.

### Prereqs

1. Download the conceptor NPZ:

   ```bash
   hf download brandonyang/metaworld-conceptors metaworld_conceptors.npz \
       --repo-type dataset --local-dir conceptors/
   ```

2. Start a steering-capable server. `--steer` always requires
   `--conceptor_npz`. For pi0.5, add `--pytorch` because steering uses hooks.
   For pi0-fast, do not add `--pytorch`; the fast path is JAX-only:

   ```bash
   # pi0.5:
   uv run scripts/serve_policy.py --pytorch --steer \
       --conceptor_npz conceptors/metaworld_conceptors.npz \
       policy:checkpoint \
       --policy.config pi05_metaworld --policy.dir checkpoints/coast-metaworld-5000

   # pi0-FAST:
   uv run scripts/serve_policy.py --steer \
       --conceptor_npz conceptors/pi0fast_metaworld_conceptors.npz \
       policy:checkpoint \
       --policy.config pi0_fast_metaworld --policy.dir checkpoints/pi0_fast_metaworld/pi0_fast_metaworld_b200_bs512/2500
   ```

   The same `--steer` server happily serves baseline (unsteered) clients too — an obs without an `__steering__` key passes straight through.

### Single task, default steering params

```bash
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3 --steer
```

Defaults (duplicated from `src/openpi/serving/steering.py`): `--steering_layer 11 --steering_alpha 0.1 --steering_beta 0.3 --steering_strategy global`. For pi0-fast, `--steering_layer` is accepted for wire compatibility but ignored because the fast NPZ has no layer axis.

### Single task, explicit params

```bash
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3 --steer \
    --steering_layer 17 --steering_alpha 0.5 --steering_beta 0.1 \
    --steering_strategy per_step
```

### Full sweep, uniform defaults

```bash
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train --steer
```

### Full sweep, per-task tuned configs

```bash
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train \
    --steer --steering_config experiments/metaworld/best_configs.json
```

Tasks not present in `best_configs.json` fall back to the config's `defaults` block if present, or the CLI scalar flags otherwise.

### All steering flags

`main.py`:

| Flag                   | Default   | Notes                                     |
|------------------------|-----------|-------------------------------------------|
| `--steer`              | False     | Toggle steering on                        |
| `--steering_layer`     | 11        | Action-expert transformer layer index     |
| `--steering_alpha`     | 0.1       | Conceptor aperture                        |
| `--steering_beta`      | 0.3       | Interpolation weight (β=0 is no-op)       |
| `--steering_strategy`  | "global"  | See strategy table in `examples/libero_env/README.md` |
| `--steering_task`      | None      | Override NPZ task key (default: `--env_name`) |

`eval_all.py` adds `--steering_config <path>` for per-task overrides.

See `examples/libero_env/README.md` for the full strategy table (`global`, `per_step`, `positive_only`, `random_matched`, `linear`) and error-behavior notes. For pi0-fast, `per_step` maps to first/mid/last token-position conceptors, and `linear` is rejected unless a separate fast linear/ActAdd implementation is added.

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
