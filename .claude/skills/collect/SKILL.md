---
name: collect
description: Collect intermediate activations from VLA models for mechanistic interpretability
disable-model-invocation: true
argument-hint: --client metaworld|libero|robocasa --checkpoint PATH [--tasks TASK...]
allowed-tools: Bash(uv run:*) Bash(export:*) Bash(nvidia-smi:*) Bash(cd:*) Bash(pgrep:*) Read Glob
---

# Collect Activations

<command-name>collect</command-name>

You are collecting intermediate activations from a VLA model during evaluation rollouts. The workflow differs significantly by client.

## Step 1: GPU Status

```!
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
```

## Step 2: Parse Arguments

Arguments: `$ARGUMENTS`

Required:
- `--client`: one of `metaworld`, `libero`, `robocasa`
- `--checkpoint`: path to checkpoint directory

Optional:
- `--tasks`: explicit task list — MetaWorld (`reach-v3 push-v3 ...`) or RoboCasa (`OpenDrawer CloseFridge ...`)
- `--split`: MetaWorld `subset` (default, 26 curated tasks) / `train` / `test`
- `--num_envs`: MetaWorld parallel envs (eval_all default 15, main default 10)
- `--gpus`: GPU IDs for multi-GPU collection (MetaWorld eval_all only)
- `--output-dir`: activation output directory (server-side for LIBERO/RoboCasa)
- `--collect_output_dir`: activation root for MetaWorld in-process collection
- `--task_suite_name`: LIBERO suite — `libero_10` (default) / `libero_spatial` / `libero_object` / `libero_goal`
- `--task_set`: RoboCasa task set — `subset` (default, 7 curated tasks) / `atomic_seen` / `composite_seen` / `composite_unseen` / `target50` / `pretrain50`
- `--num_episodes`: episodes per task (eval_all default 15 for LIBERO/RoboCasa; main default 1)
- `--num_workers`: LIBERO/RoboCasa subprocess concurrency (default 10)

If `--client` is not specified, ask the user.
If `--checkpoint` is not specified, list available checkpoints and ask:
```bash
ls -dt checkpoints/*/ outputs/*/checkpoints/*/ 2>/dev/null | head -10
```

## Step 3: Validate Checkpoint

```bash
ls <checkpoint-path>/
```

If the path doesn't exist, show available checkpoints and ask the user.

## Step 4: Run Collection

### MetaWorld — In-Process (single command, NO server)

MetaWorld loads the policy directly via `main.py --collect` (single task) or `eval_all.py --collect` (all tasks). Do **not** run `serve_policy.py`. Start with `--num_envs 16` and halve if you OOM.

**All tasks in a split:**
```bash
export CUDA_VISIBLE_DEVICES=<GPU>
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --collect --split <SPLIT> --num_envs 16 \
    --policy.config=pi05_metaworld \
    --policy.dir=<CHECKPOINT> \
    --collect_output_dir ./activations
```

**Single task:**
```bash
export CUDA_VISIBLE_DEVICES=<GPU>
MUJOCO_GL=egl uv run examples/metaworld/main.py \
    --collect --env_name <TASK> --num_envs 16 \
    --policy.config=pi05_metaworld \
    --policy.dir=<CHECKPOINT> \
    --collect_output_dir ./activations
```

For a task subset, use `--tasks reach-v3 push-v3 ...` on `eval_all.py` (skips `--split`).
For multi-GPU, use `--gpus 0 1` on `eval_all.py` instead of `CUDA_VISIBLE_DEVICES`.

### LIBERO — Server + Client (two commands, two venvs)

Collection requires **two terminals**: a collection-mode server (root venv) and a client (libero_env venv).

**Important:** The collection server requires `--pytorch` and `--collect_activations`. It is collection-only and rejects plain inference requests.

**Terminal 1 — Server (root venv):**
```bash
export CUDA_VISIBLE_DEVICES=<GPU>
uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi05_libero \
    --policy.dir=<CHECKPOINT>
```

**Terminal 2 — Client (libero_env venv):**
```bash
cd examples/libero_env
MUJOCO_GL=egl uv run python eval_all.py \
    --task_suite_name <SUITE> --collect --num_workers <N>
```

For a single task: `uv run python main.py --task_suite_name <SUITE> --task_id <ID> --collect`

### RoboCasa — Server + Client (two commands, two venvs)

Same architecture as LIBERO. Collection server in root venv, client in robocasa_env venv.

**Terminal 1 — Server (root venv):**
```bash
export CUDA_VISIBLE_DEVICES=<GPU>
uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi05_robocasa \
    --policy.dir=<CHECKPOINT>
```

**Terminal 2 — Client (robocasa_env venv):**
```bash
cd examples/robocasa_env
MUJOCO_GL=egl uv run python eval_all.py \
    --task_set <TASK_SET> --collect --num_workers <N>
```

For a single task: `uv run python main.py --env_name <ENV> --collect`

## Step 5: Write Session State

Before launching, write session state for compaction recovery:
```bash
echo "Collecting: client=<CLIENT> checkpoint=<CHECKPOINT> GPU=<GPU_ID> started=$(date '+%Y-%m-%d %H:%M')" > "$CLAUDE_PROJECT_DIR/.claude/.session_state"
```

## Step 6: After Collection

- Report the output directory path
- For LIBERO/RoboCasa: activations are on the **server's** filesystem under `--output-dir`
- Suggest validation (pick the validator that matches the `collection_mode` of the run):
  - pi0 / pi0.5 (`v1`): `ACTIVATIONS_DIR=<dir>/<task> uv run pytest tests/test_activations.py -v`
  - pi0-FAST (`fast_v1`): `ACTIVATIONS_FAST_DIR=<dir>/<task> uv run pytest tests/test_activations_fast.py -v`
  - GR00T N1.5 (`groot_v1`): `cd groot_env && ACTIVATIONS_DIR=<dir>/<task> uv run pytest tests/test_groot_activations.py -v`

Report the full commands you are running before executing them.
