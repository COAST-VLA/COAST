# GR00T N1.5 RoboCasa: Conceptor Steering Experiments

Conceptor-based representation steering for the GR00T N1.5 vision-language-action
model on RoboCasa manipulation tasks. Evaluates whether soft projection via
conceptors can improve task success rates at inference time, without retraining.

## Overview

The pipeline has three stages:

1. **Activation collection** -- roll out the base policy, recording hidden-state
   activations for success and failure episodes.
2. **Conceptor construction** -- compute per-layer, per-alpha conceptor matrices
   (contrastive, positive-only, and linear directions) from the collected
   activations.
3. **Steered evaluation** -- sweep over strategies, layers, alphas, and betas,
   injecting the conceptor (or linear direction) at inference time and measuring
   success rate changes.

## Tasks

| Task | Description |
|------|-------------|
| CloseFridge | Close an open refrigerator door |
| CoffeeSetupMug | Place a mug under the coffee machine |
| OpenDrawer | Pull open a kitchen drawer |
| OpenStandMixerHead | Lift the head of a stand mixer |
| PickPlaceCounterToCabinet | Pick object from counter, place in cabinet |
| PickPlaceCounterToStove | Pick object from counter, place on stove |
| TurnOnElectricKettle | Toggle the electric kettle switch |

## Prerequisites

- **GR00T N1.5 checkpoint** at `../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000`
  (relative to repo root)
- **groot_env** virtual environment (Python 3.10) with torch 2.7.0+cu128
  and flash-attn compiled from source (required for B200/Blackwell GPUs)
- **robocasa_env** virtual environment (Python 3.11) with torch 2.7.0+cu128
- **Pre-built conceptor NPZ** at `$OPENPI_DATA_HOME/groot_n15_robocasa_conceptors.npz`
- SLURM cluster with B200 GPUs (`dgx-b200` partition)

Set `OPENPI_DATA_HOME` to the directory containing the NPZ file (defaults to
`$HOME/.cache/openpi` if unset).

## Directory Structure

```
experiments/groot_robocasa/
  README.md                          # this file
  collect_missing_activations.sh     # SLURM: collect activations for tasks missing data
  run_steering_optimized.sh          # SLURM: auto-select params then sweep
  run_steering_sweep.sh              # SLURM: manual full sweep (current)
  selected_params/
    selected_params.json             # output of select_parameters.py
  src/
    build_conceptors.py              # stage 2: build conceptor NPZ from activations
    conceptor_steering.py            # stage 3: steered evaluation sweep
    select_parameters.py             # auto-pick layer, alpha, beta from conceptor spectra
  steering_results/                  # (gitignored) runtime outputs, logs, videos
```

## Quick Start

### 1. Collect activations (if needed)

```bash
# Collect activations for missing tasks (submits SLURM jobs)
bash experiments/groot_robocasa/collect_missing_activations.sh

# Dry-run to see what would be submitted
bash experiments/groot_robocasa/collect_missing_activations.sh --dry-run
```

### 2. Build conceptors

```bash
cd groot_env
.venv/bin/python ../experiments/groot_robocasa/src/build_conceptors.py
```

This reads activations from `$OPENPI_DATA_HOME/huggingface/lerobot/brandonyang/groot_n15-robocasa-activations-v1-15env`
and writes `$OPENPI_DATA_HOME/groot_n15_robocasa_conceptors.npz`.

### 3. Select parameters

```bash
cd groot_env
.venv/bin/python ../experiments/groot_robocasa/src/select_parameters.py
```

Outputs `experiments/groot_robocasa/selected_params/selected_params.json` with
the best layer, alpha, and beta values.

### 4. Run steering sweep

```bash
# Submit one SLURM job per task (7 jobs total)
bash experiments/groot_robocasa/run_steering_sweep.sh

# Dry-run
bash experiments/groot_robocasa/run_steering_sweep.sh --dry-run
```

Or use the optimized script that chains parameter selection and sweeping:

```bash
bash experiments/groot_robocasa/run_steering_optimized.sh
bash experiments/groot_robocasa/run_steering_optimized.sh --skip-select  # reuse existing params
```

### 5. Single-task manual run

```bash
cd groot_env
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
    ../experiments/groot_robocasa/src/conceptor_steering.py \
    --task CloseFridge \
    --layers 10 \
    --alphas 0.1 0.5 1.0 \
    --betas 0.1 0.3 \
    --strategies global per_step \
    --num-episodes 15
```

## Steering Strategies

| Strategy | Description |
|----------|-------------|
| `global` | Single contrastive conceptor applied at all 4 denoising steps |
| `per_step` | Different contrastive conceptor per denoising step |
| `positive_only` | Uses C_success directly (no contrastive NOT operation) |
| `linear` | ActAdd-style h' = h + alpha * v, where v = unit(mean_success - mean_failure) |
| `random` | Random PSD matrix baseline (control for any-projection effect) |

## Conceptor Math

The contrastive conceptor is:

```
C_s = R_s (R_s + alpha^{-2} I)^{-1}     # success conceptor
C_f = R_f (R_f + alpha^{-2} I)^{-1}     # failure conceptor
C   = C_s (I - C_f)                       # contrastive: amplify success, suppress failure
```

At inference, the hidden state h at the selected layer is projected:

```
h' = (1 - beta) * h + beta * C @ h
```

where beta controls steering strength (0 = no steering, 1 = full projection).

## Important Notes

- **Do not use `uv run`** for steering scripts. The gr00t[base] package pins
  torch==2.5.1+cu124 in pyproject.toml, and `uv run` will revert manually
  installed torch upgrades. Always use `.venv/bin/python` directly.
- **flash-attn must be built from source** on a GPU node against torch 2.7.0
  (pre-built wheels are compiled against older torch ABI).
- Results are written incrementally -- if a job is interrupted, restarting it
  will skip already-completed conditions.
