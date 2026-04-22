---
name: serve
description: Check GPU availability and serve a trained policy via WebSocket for inference
disable-model-invocation: true
argument-hint: <checkpoint-path> [--config CONFIG] [--port PORT]
allowed-tools: Bash(uv run:*) Bash(export:*) Bash(nvidia-smi:*) Read Glob
---

# Serve a Policy Model

<command-name>serve</command-name>

You are starting a WebSocket policy server for a trained checkpoint. Follow these steps in order.

## Step 1: Check GPU Availability

Run `nvidia-smi` and analyze the output. Inference requires **1 free GPU** (< 1 GB memory used).

```!
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

- If 1+ GPUs are free: pick the one with the lowest memory usage and proceed.
- If all GPUs are occupied: **STOP** and tell the user. Show them the current GPU utilization table. Do not attempt to serve.

## Step 2: Parse Arguments

Arguments: `$ARGUMENTS`

Defaults (override with flags):
- **Config**: `pi05_metaworld` (override with `--config <name>`)
- **Checkpoint path**: first positional argument (required)

If no checkpoint path is provided, list recent checkpoints and ask the user to choose:

```bash
ls -dt checkpoints/*/ 2>/dev/null | head -10
```

## Step 3: Validate Checkpoint

Verify the checkpoint directory exists and contains expected files:

```bash
ls <checkpoint-path>/
```

If the path doesn't exist, show available checkpoints and ask the user.

## Step 4: Launch Server

Write session state for context preservation across compaction:

```bash
echo "Serving: config=<CONFIG> checkpoint=<CHECKPOINT_PATH> GPU=<GPU_ID> started=$(date '+%Y-%m-%d %H:%M')" > "$CLAUDE_PROJECT_DIR/.claude/.session_state"
```

Set the GPU device and launch:

```bash
export CUDA_VISIBLE_DEVICES=<selected GPU id>
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<CONFIG> --policy.dir=<CHECKPOINT_PATH>
```

Report the full command you are running before executing it.

## After Serving

Once the server is running, remind the user they can evaluate in a separate terminal — pick the client that matches `--policy.config`:

```bash
# MetaWorld (root venv, from repo root)
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py              # defaults --split subset

# LIBERO (libero_env venv)
cd examples/libero_env
MUJOCO_GL=egl uv run python eval_all.py                          # defaults --task_suite_name libero_10

# RoboCasa (robocasa_env venv)
cd examples/robocasa_env
MUJOCO_GL=egl uv run python eval_all.py                          # defaults --task_set subset
```
