# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

openpi-metaworld is a fork of Physical Intelligence's openpi repository, focused on MetaWorld robot environment integration. It implements Vision-Language-Action (VLA) models (pi0, pi0-FAST, pi0.5) in JAX (primary) and PyTorch, with fine-tuning and evaluation pipelines for robotic manipulation tasks.

## GPU Selection (IMPORTANT)

Before running **any** script (training, serving, evaluation, etc.), you MUST check GPU availability by running `nvidia-smi` and set `CUDA_VISIBLE_DEVICES` accordingly:

- **Inference/serving**: Requires **1 GPU**. Pick the GPU with the lowest utilization/memory usage. If ALL GPUs are fully occupied, **stop and tell the user** that no GPUs are available to serve a policy.
- **Training**: Requires **4 GPUs** to be available. If fewer than 4 are free, **stop and tell the user**.

Always `export CUDA_VISIBLE_DEVICES=<id(s)>` before running commands, e.g.:
```bash
export CUDA_VISIBLE_DEVICES=2              # single GPU for inference
export CUDA_VISIBLE_DEVICES=0,1,2,3        # 4 GPUs for training
```

## Common Commands

### Setup
```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Training
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_metaworld --exp-name <name> --overwrite --num_train_steps 30000
```

Compute normalization stats before first training run:
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_metaworld
```

### Serving & Evaluation
```bash
# Serve policy
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_metaworld --policy.dir=/path/to/checkpoint

# Evaluate (separate terminal)
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split test
```

See `examples/metaworld/README.md` for the metaworld-specific dataset generation, training, and activation collection workflows.

### Testing
```bash
# Run all non-manual tests (CI default)
uv run pytest --strict-markers -m "not manual"

# Run a single test file
uv run pytest tests/models/test_pi0.py -v

# Run tests in a subdirectory
uv run pytest tests/shared/ -v

# MetaWorld env tests (requires GPU + EGL rendering)
MUJOCO_GL=egl uv run pytest tests/metaworld/test_metaworld_envs.py -v
```

### Linting & Formatting
```bash
uv run ruff check .          # lint
uv run ruff check --fix .    # lint with autofix
uv run ruff format .         # format
```

Pre-commit hooks (ruff lint, ruff format, uv-lock) run automatically. Install with `pre-commit install`.

## Unified Cache & Model Storage

All downloaded models, HuggingFace caches, and GCS assets are consolidated under a **single root directory** controlled by the `OPENPI_DATA_HOME` environment variable (default: `~/.cache/openpi`).

On this machine the physical location is `/vast/projects/ungar/stellar/miaom/.cache/openpi` (symlinked from `~/.cache/openpi`).

```
$OPENPI_DATA_HOME/                          # ~/.cache/openpi (default)
├── huggingface/                            # HF_HOME — auto-set by download.py at import time
│   ├── hub/                                # HuggingFace model hub cache
│   ├── datasets/                           # HuggingFace datasets parquet cache (~33G)
│   ├── lerobot/                            # LeRobot datasets (e.g. brandonyang/metaworld_ml45)
│   ├── xet/                                # XET storage cache
│   ├── token                               # HF auth token
│   └── stored_tokens                       # HF stored tokens
├── openpi-assets/                          # Base model weights downloaded from gs://openpi-assets/
│   └── checkpoints/
│       └── pi05_base/                      # (~12G) params, assets, etc.
└── big_vision/                             # PaliGemma tokenizer downloaded from gs://big_vision/
    └── paligemma_tokenizer.model           # (~4MB)
```

**How it works** (see `src/openpi/shared/download.py`):
- `get_data_home()` returns the root. `get_cache_dir()` is an alias.
- `_configure_hf_home()` runs at import time: if `HF_HOME` is not already set, it sets `HF_HOME=$OPENPI_DATA_HOME/huggingface/`. This ensures all HuggingFace downloads (transformers, datasets, LeRobot, XET) land under the same root.
- `maybe_download(url)` downloads GCS URLs (e.g. `gs://openpi-assets/...`) into `$OPENPI_DATA_HOME/{netloc}/{path}`.

**Training outputs** (checkpoints, assets, activations) default to **relative paths** from the working directory:
- Checkpoints: `./checkpoints/{config_name}/{exp_name}/{step}/` (configurable via `--checkpoint_base_dir`)
- Assets/norm stats: `./assets/{config_name}/` (configurable via `--assets_base_dir`)
- Activations: `./activations/` (configurable via `--output_dir`)

**To override the cache root**, set `OPENPI_DATA_HOME` before running any script:
```bash
export OPENPI_DATA_HOME=/path/to/unified/cache
```

## Architecture

### Model Layer (`src/openpi/models/`)
Three VLA model variants, all built on a Gemma LLM backbone + SigLip vision encoder:
- **pi0**: Flow-based action generation (continuous diffusion)
- **pi0-FAST**: Autoregressive action generation via FAST tokenizer
- **pi0.5**: Improved pi0 with knowledge insulation and AdaRMS

JAX/Flax NNX is the primary implementation. PyTorch versions live in `src/openpi/models_pytorch/`.

### Policy Layer (`src/openpi/policies/`)
Robot-specific wrappers that translate between raw environment observations and model inputs/outputs. Each policy defines `Inputs` (env -> model) and `Outputs` (model -> env) dataclasses. Key file: `policy_config.py` creates policies from training configs.

### Training Config (`src/openpi/training/config.py`)
Central configuration file. All named training configs (e.g., `pi05_metaworld`, `pi05_libero`) are registered in `_CONFIGS` dict at the bottom. Each config ties together: model type, data config (with transforms), optimizer, weight loader, and hyperparameters. Use `get_config(name)` to retrieve.

### Data Pipeline (`src/openpi/transforms.py`, `src/openpi/training/data_loader.py`)
Three-stage transform pipeline: **repack** (dataset-specific format -> common format) -> **normalize** (z-score or quantile) -> **model transforms** (tokenization, image resizing). Data is loaded from LeRobot-format datasets.

### Serving (`src/openpi/serving/`)
WebSocket-based policy server for remote inference. Clients use the `openpi-client` package (`packages/openpi-client/`).

## Key Conventions

- **Python 3.11+**, managed by `uv` (not pip/conda)
- **Line length**: 120 characters
- **Imports**: Single-line, sorted by ruff isort
- **Tests**: All in `tests/` directory, named `test_*.py`. Subdirectories mirror source structure (models, shared, training, etc.). pytest markers: `manual` for GPU-required tests
- **Config names** follow pattern: `{model}_{robot}` (e.g., `pi05_metaworld`, `pi0_aloha_sim`)
- **Ruff rules** are extensive (see `pyproject.toml`); `F722` ignored for array typing, `T201` ignored (print allowed)
- `third_party/` and `src/openpi/models_pytorch/transformers_replace/` are excluded from linting