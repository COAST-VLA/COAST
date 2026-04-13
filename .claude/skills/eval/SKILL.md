---
name: eval
description: Run evaluation against a policy server for MetaWorld, LIBERO, or RoboCasa
disable-model-invocation: true
argument-hint: --client metaworld|libero|robocasa [--split SPLIT] [--env_name TASK]
allowed-tools: Bash(uv run:*) Bash(export:*) Bash(nvidia-smi:*) Bash(cd:*) Bash(curl:*) Bash(pgrep:*) Read Glob
---

# Run Evaluation

<command-name>eval</command-name>

You are running an evaluation against a policy server. Follow these steps in order.

## Step 1: GPU Status

```!
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
```

## Step 2: Parse Arguments

Arguments: `$ARGUMENTS`

Required:
- `--client`: one of `metaworld`, `libero`, `robocasa`

If `--client` is not specified, ask the user which client to evaluate.

## Step 3: Check Policy Server

The evaluation clients connect to a WebSocket policy server. Check if one is running:

```bash
pgrep -f serve_policy || curl -s --max-time 2 http://localhost:8000 2>/dev/null
```

If no server is running, **warn the user** and suggest they start one with `/serve` first. Do not proceed without confirming.

## Step 4: Run Evaluation

Each client has different commands and venv requirements:

### MetaWorld (root venv, from repo root)
```bash
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --split train
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3  # single task
```
- Use `--split train|test` for full ML45 eval, `--env_name` for single task
- Parallelism: `--num_envs N` (default 15 for eval_all, 10 for main)

### LIBERO (libero_env venv, from examples/libero_env/)
```bash
cd examples/libero_env
MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_spatial
MUJOCO_GL=egl uv run python main.py --task_suite_name libero_spatial --task_id 0  # single task
```
- Use `--task_suite_name` to pick suite (libero_spatial, libero_object, libero_goal, libero_10, libero_90)
- Parallelism: `--num_workers N` (default 5)

### RoboCasa (robocasa_env venv, from examples/robocasa_env/)
```bash
cd examples/robocasa_env
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen
MUJOCO_GL=egl uv run python main.py --env_name CloseBlenderLid  # single task
```
- Use `--task_set` to pick set (atomic_seen, composite_seen, composite_unseen, pretrain50)
- Parallelism: `--num_workers N` (default 5)

## Step 5: Report Results

After evaluation completes, read and display the `results.json` from the output directory.

Report the full command you are running before executing it.
