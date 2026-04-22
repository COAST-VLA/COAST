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
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py            # defaults --split subset (26 tasks)
MUJOCO_GL=egl uv run examples/metaworld/main.py --env_name reach-v3  # single task
```
- Use `--split subset|train|test` to pick the sweep. `subset` (default) is a curated 26-task subset; `train` = 45 ML45-train; `test` = 5 held-out.
- `--tasks reach-v3 push-v3 ...` overrides `--split` with an explicit list.
- Parallelism: `--num_envs N` (eval_all default 15, main default 10).

### LIBERO (libero_env venv, from examples/libero_env/)
```bash
cd examples/libero_env
MUJOCO_GL=egl uv run python eval_all.py                        # defaults --task_suite_name libero_10
MUJOCO_GL=egl uv run python main.py --task_suite_name libero_10 --task_id 0  # single task
```
- Use `--task_suite_name` to pick suite: `libero_10` (default), `libero_spatial`, `libero_object`, `libero_goal`.
- `--num_episodes` default: eval_all = 15, main = 1.
- Parallelism: `--num_workers N` (default 10).

### RoboCasa (robocasa_env venv, from examples/robocasa_env/)
```bash
cd examples/robocasa_env
MUJOCO_GL=egl uv run python eval_all.py                        # defaults --task_set subset (7 curated tasks)
MUJOCO_GL=egl uv run python main.py --env_name CloseBlenderLid  # single task
```
- Use `--task_set` to pick set: `subset` (default, 7 curated), `atomic_seen`, `composite_seen`, `composite_unseen`, `target50`, `pretrain50`.
- `--tasks OpenDrawer CloseFridge ...` overrides `--task_set` with an explicit list.
- `--num_episodes` default: eval_all = 15, main = 1.
- Parallelism: `--num_workers N` (default 10).

## Step 5: Report Results

After evaluation completes, read and display the `results.json` from the output directory.

Report the full command you are running before executing it.
