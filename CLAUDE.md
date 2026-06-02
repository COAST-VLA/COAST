# CLAUDE.md

## Project Overview

This is a fork of Physical Intelligence's openpi repository for **activation collection and mechanistic interpretability** on Vision-Language-Action (VLA) models (pi0, pi0-FAST, pi0.5) plus NVIDIA GR00T N1.5. The models are implemented in JAX (primary) and PyTorch, with fine-tuning, evaluation, and activation collection pipelines across four robot environments.

## Evaluation & Collection Clients

| Client | Architecture | Venv | Naive eval | Activation collection | Steering (`--steer`) |
|--------|-------------|------|------------|----------------------|----------------------|
| **MetaWorld** | Server-client over WebSocket; client runs vectorized envs | Root venv | pi0.5, pi0-fast | pi0.5 `v1`, pi0-fast `fast_v1` server-side batched collection | pi0.5 PyTorch hooks; pi0-fast JAX pre-logit steering |
| **LIBERO** | Server-client over WebSocket | **Separate venv (Python 3.8)** in `examples/libero_env/` | pi0.5, pi0-fast | pi0.5 `v1`, pi0-fast `fast_v1` server-side collection | pi0.5 PyTorch hooks; pi0-fast JAX pre-logit steering |
| **RoboCasa** | Server-client over WebSocket | **Separate venv (Python 3.11)** in `examples/robocasa_env/` | pi0.5, GR00T N1.5 | pi0.5 `v1`, GR00T `groot_v1` server-side collection | pi0.5 PyTorch hooks; GR00T N1.5 DiT hooks; no pi0-fast RoboCasa path |
| **DROID** | Server-client over WebSocket; real-robot control laptop | Root venv (server), DROID conda env + `openpi-client` (laptop) | See `examples/droid/README.md` | See `examples/droid/README.md` | Real-robot/manual path; outside the simulator support matrix |

For steering configuration, see `src/openpi/serving/steering.py` (root runtime), `groot_env/groot_steering.py` (GR00T runtime), `src/openpi/serving/conceptors.py` (offline NPZ builder), and `packages/openpi-client/src/openpi_client/steering.py` (wire protocol). Per-env tuning lives under `experiments/{libero,robocasa,metaworld,droid}/`. pi0/pi0.5 steering uses PyTorch hooks on the action expert; pi0-fast steering is JAX-only and applies Miranda-v2-style conceptor matrices to autoregressive token `pre_logits` before the LM head; GR00T N1.5 RoboCasa steering uses PyTorch hooks on the action DiT residual stream from `groot_env/`. There is no pi0-fast RoboCasa config/checkpoint path in this branch.

Canonical activation-collection reference: [`docs/activation_collection.md`](docs/activation_collection.md). For per-client workflow details (dataset generation, training configs, eval commands), read the respective `examples/{client}/README.md`.

## Multi-Venv Setup (IMPORTANT)

This repo has **four separate Python environments**. Running commands from the wrong directory will use the wrong venv and fail.

| Venv | Python | Directory | When to use |
|------|--------|-----------|-------------|
| **Root** | 3.11 | Repo root | Training, serving (pi0/pi0.5/pi0-FAST), MetaWorld eval/collection, DROID eval server, tests |
| **libero_env** | 3.8 | `examples/libero_env/` | LIBERO client eval/collection |
| **robocasa_env** | 3.11 | `examples/robocasa_env/` | RoboCasa client eval/collection |
| **groot_env** | 3.10 | `groot_env/` | GR00T N1.5 policy server (separate due to `torch==2.5.1` pin) |

```bash
# Root venv (training, serving, metaworld)
uv run scripts/train.py ...
uv run scripts/serve_policy.py ...
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py ...

# LIBERO client — MUST cd first
cd examples/libero_env
MUJOCO_GL=egl uv run python main.py ...

# RoboCasa client — MUST cd first
cd examples/robocasa_env
MUJOCO_GL=egl uv run python main.py ...
```

First-time setup for each venv:
```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync                      # root
cd examples/libero_env && uv sync                    # libero (Python 3.8)
cd examples/robocasa_env && uv sync                  # robocasa
```

The `openpi-client` package (`packages/openpi-client/`) is an editable dependency in all three venvs. After modifying it, re-sync the example venvs.

## GPU Selection (IMPORTANT)

Before running **any** GPU script, check availability with `nvidia-smi` and set `CUDA_VISIBLE_DEVICES`:

- **Inference/serving/collection**: **1 GPU**. Pick the one with lowest memory usage. If all occupied, **stop and tell the user**.

```bash
export CUDA_VISIBLE_DEVICES=2              # single GPU
```

## Common Commands

### Serving & Evaluation
```bash
# Serve (from root)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_metaworld --policy.dir=/path/to/checkpoint

# MetaWorld eval (from root)
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train

# LIBERO eval (from examples/libero_env/)
cd examples/libero_env
MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_spatial

# RoboCasa eval (from examples/robocasa_env/)
cd examples/robocasa_env
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen
```

### Testing
```bash
uv run pytest --strict-markers -m "not manual"           # CI default (no GPU)
MUJOCO_GL=egl uv run pytest tests/metaworld/ -v          # GPU tests
```

### Linting & Formatting
```bash
uv run ruff check .          # lint
uv run ruff check --fix .    # lint with autofix
uv run ruff format .         # format
```

## Architecture

### Model Layer (`src/openpi/models/`)
Three VLA model variants on a Gemma LLM backbone + SigLip vision encoder:
- **pi0**: Flow-based action generation (continuous diffusion)
- **pi0-FAST**: Autoregressive via FAST tokenizer
- **pi0.5**: Improved pi0 with knowledge insulation and AdaRMS

JAX/Flax NNX is primary. PyTorch versions in `src/openpi/models_pytorch/`.

### Policy Layer (`src/openpi/policies/`)
Robot-specific wrappers translating between env observations and model I/O. Key file: `policy_config.py`.

### Training Config (`src/openpi/training/config.py`)
Central config file. All named configs (e.g., `pi05_metaworld`, `pi05_libero`) registered in `_CONFIGS` list. Use `get_config(name)` to retrieve.

### Data Pipeline (`src/openpi/transforms.py`, `src/openpi/training/data_loader.py`)
Three-stage: **repack** (dataset-specific -> common) -> **normalize** (z-score/quantile) -> **model transforms** (tokenization, resizing). Data from LeRobot-format datasets.

### Serving & Collection (`src/openpi/serving/`)
WebSocket policy server. Collection-mode server (`--collect_activations`) saves per-step activations to disk: pi0/pi0.5 use PyTorch hooks, pi0-fast uses JAX intermediates, and GR00T uses its own collector in `groot_env/`. Clients use the `openpi-client` package.

## HuggingFace Downloads

Always use the `hf download` CLI and download into the project tree — never into `~/.cache/huggingface` or any user-global cache.

```bash
# Activation datasets — land under activations/<dataset-name>/ (one
# canonical root so mech-interp tooling + .gitignore can use a single rule)
hf download brandonyang/pi05-metaworld-activations-v1-ml45train-16env \
    --repo-type dataset --local-dir activations/pi05-metaworld-activations-v1-ml45train-16env

# Checkpoints — place under checkpoints/, optionally --include a subpath
hf download robocasa/robocasa365_checkpoints \
    --include "pi05_pretrain_human300/multitask_learning/75000/*" \
    --local-dir checkpoints

# Discover flags / subcommands
hf download --help
```

Rules:
- Always pass `--local-dir <path>`.
- Datasets require `--repo-type dataset`; models are the default.
- Run from the directory where the asset will be consumed (repo root for shared assets, `examples/{client}/` for client-specific ones).
- When unsure about flags, run `hf download --help` rather than guessing.

## HuggingFace Uploads (activation datasets)

`hf upload <root>` rejects a commit once it crosses the **25k file limit**, which activation datasets routinely exceed. Use [`scripts/upload_activations_by_task.sh`](scripts/upload_activations_by_task.sh) — it creates the HF repo idempotently and splits the upload into one commit per task.

```bash
# Upload every step subdir of the local dataset (auto-detect steps):
scripts/upload_activations_by_task.sh \
    brandonyang/pi0fast-metaworld-activations-v1-ml45train-16env \
    activations/pi0fast-metaworld-activations-v1-ml45train-16env

# Upload one step only (argv[3]):
scripts/upload_activations_by_task.sh \
    brandonyang/pi0fast-metaworld-activations-v1-ml45train-16env \
    activations/pi0fast-metaworld-activations-v1-ml45train-16env \
    2500
```

Expected local layout: `<local_root>/<step>/<task>/...`. Each `<task>` becomes a separate HF commit at `path_in_repo=<step>/<task>`. Do not use bare `hf upload` on a multi-task activation tree — it will fail with `413 Payload Too Large`.

## Key Conventions

- **Python 3.11+** (except libero_env: 3.8), managed by `uv`
- **Line length**: 120 characters
- **Tests**: `tests/` directory, `test_*.py`. Marker `manual` = GPU-required
- **Config names**: `{model}_{robot}` (e.g., `pi05_metaworld`, `pi05_libero`)
- **Ruff rules**: see `pyproject.toml`; `F722` ignored (array typing), `T201` ignored (print allowed)
- `third_party/` and `src/openpi/models_pytorch/transformers_replace/` excluded from linting
