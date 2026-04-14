# Robocasa Client Example

Since [Robocasa](https://robocasa.ai/docs/build/html/index.html) has weird dependencies that we cannot resolve with `openpi`, we need to create a separate virtual env in `examples/robocasa_env`. Thus, the simulation (client) and the model (server) uses websocket to communicate.

## Installation

This is a modified version of the [original setup guide](https://robocasa.ai/docs/build/html/introduction/installation.html).

```bash
cd examples/robocasa_env
uv sync
```

Install the package and download assets:

```bash
cd examples/robocasa_env
uv run python -m robocasa.scripts.setup_macros              # Set up system variables.
uv run python -m robocasa.scripts.download_kitchen_assets   # Caution: Assets to be downloaded are around 10GB.
```

## Serving the Robocasa Policy

### Prepare the Checkpoint

```bash
hf download robocasa/robocasa365_checkpoints --include "pi05_pretrain_human300/multitask_learning/75000/*"  --local-dir checkpoints

# Do some path surgery to match the expected structure of the policy server
mkdir -p checkpoints/pi05_pretrain_human300/multitask_learning/75000/assets/robocasa
mv checkpoints/pi05_pretrain_human300/multitask_learning/75000/assets/norm_stats.json checkpoints/pi05_pretrain_human300/multitask_learning/75000/assets/robocasa
```

### Start the Server

```bash
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_robocasa \
    --policy.dir=checkpoints/pi05_pretrain_human300/multitask_learning/75000
```

To use the PyTorch backend instead of JAX, add `--pytorch`. The first run converts the JAX checkpoint to `model.safetensors` (cached, so later runs are fast):

```bash
uv run scripts/serve_policy.py --pytorch policy:checkpoint \
    --policy.config=pi05_robocasa \
    --policy.dir=checkpoints/pi05_pretrain_human300/multitask_learning/75000
```

The client (`main.py` / `eval_all.py`) is unchanged — the WebSocket protocol is the same for both backends.

### Run Evaluation

There are two evaluation entry points:

1. **`main.py`** — evaluate a single task (one `env_name`) in the current process.
2. **`eval_all.py`** — evaluate every task in a task set (e.g. `atomic_seen`, `composite_seen`, `composite_unseen`, `pretrain50`) in parallel by launching one `main.py` subprocess per env.

Both default to `--split pretrain` (in-distribution object instances). Each episode's video is built by tiling the env's three cameras (`agentview_left`, `agentview_right`, `eye_in_hand`) into one grid frame.

RoboCasa does **not** support parallel envs inside one process (EGL/OpenGL contexts are not shareable across threads in a single process, and `gym.vector.AsyncVectorEnv`'s fork model trips over MuJoCo's GL init), so the per-env grid tiles multiple cameras of a single env rather than multiple parallel envs as in the metaworld example. `eval_all.py` sidesteps this by giving each task its own `main.py` subprocess: every subprocess has its own EGL context, which **is** safe.

#### Single environment

```bash
cd examples/robocasa_env
MUJOCO_GL=egl uv run python main.py --env_name CloseBlenderLid
```

#### All tasks in a task set (parallel subprocesses)

```bash
cd examples/robocasa_env
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen
MUJOCO_GL=egl uv run python eval_all.py --task_set composite_seen --num_episodes 3 --num_workers 5
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen --output_dir /tmp/mech_interp_run1
```

`eval_all.py` submits one `main.py` subprocess per env in the task set to a `ThreadPoolExecutor` with `--num_workers` max concurrency. Because RoboCasa env stepping is roughly 10x slower than libero (~400 ms per step), parallelism buys a substantial wall-clock win; 5–10 workers is the sweet spot. Pass `--num_workers 1` to fall back to fully sequential execution with inline stack traces on crash.

Everything one run produces lives under a single top-level directory (by default `examples/robocasa_env/output/<task_set>-<split>/`, or whatever `--output_dir` points at). No split between results and videos; no `single-` prefix leaking out:

```
<output_dir>/
├── results.json                         # per-task + mean success rate summary
├── parallel_logs/
│   ├── task_00_CloseBlenderLid.log      # per-subprocess stdout+stderr
│   ├── task_01_OpenCabinet.log
│   └── ...
├── CloseBlenderLid/
│   ├── episode_000.mp4
│   └── ...
├── OpenCabinet/
│   └── episode_000.mp4
└── ...
```

`results.json` is updated incrementally after each task finishes so progress is preserved on early exit. The final summary is sorted by success rate, descending.

## Activation Collection

For mech-interp work you can have the policy server save per-step intermediate
activations to disk while a robocasa rollout runs. This uses the same
"collection-mode" policy server as the libero example: it wraps the policy in
`CollectingPolicy` and writes the same on-disk format as
`examples/metaworld/collect_activations.py`. Activations live entirely on the
**server's** filesystem — the robocasa client never touches them, so the
client and server can be on different machines.

Start the collection-mode server from the repo root in one terminal:

```bash
# Terminal 1 (main openpi venv) — server pinned to GPU 0
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi05_robocasa \
    --policy.dir=checkpoints/pi05_pretrain_human300/multitask_learning/75000
```

Then run a robocasa rollout with `--collect` from this directory:

```bash
# Terminal 2 (robocasa_env venv)
cd examples/robocasa_env
MUJOCO_GL=egl uv run python main.py --env_name CloseBlenderLid --collect

# or for a whole task set (parallel subprocesses, each with its own CollectionSession):
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen --collect --num_workers 5
```

Each `eval_all.py` subprocess creates its own `CollectionSession` keyed on its distinct `env_name`, so the shared collection-mode server writes activations to disjoint output directories with no cross-subprocess coordination. The server's single-threaded asyncio dispatch serializes the underlying hook-based `infer_with_intermediates` call automatically, and `CollectingPolicy`'s explicit lock documents the invariant for future executor-based optimizations.

Notes:
- Collection mode requires `--pytorch` on the server. `infer_with_intermediates`
  is implemented for the PyTorch backend only.
- A collection-mode server **rejects** plain inference requests. If you want to
  also run regular eval, start a separate non-collection server on a different
  port.
- The server's `--output-dir` is on the **server's** filesystem. With
  `--output-dir ./activations`, files land at
  `./activations/<checkpoint_step>/<env_name>/episode_NNN_env_000/step_NNNN/`
  relative to wherever the server was launched from.
- The robocasa client uses `env_name` (e.g. `CloseBlenderLid`) as the
  `task_name` in the collection metadata. The `task_id` field is fixed at 0
  since each robocasa env is its own standalone task. The `episode_id` cycles
  through `0..num_episodes-1` per env.
- See `examples/libero_env/README.md` (the **Protocol** section under
  Activation Collection) for the full wire-level spec of the `__collect__`
  and `__finalize_episode__` payloads. The same `openpi_client.collection_session.CollectionSession`
  helper handles the bookkeeping for libero, robocasa, and any future client.

## Running with Steering

Same flag surface as libero — the end-user entry point is `--steer`. The NPZ
key is the RoboCasa env name directly (e.g., `CloseFridge`).

### Prereqs

```bash
hf download brandonyang/robocasa-conceptors robocasa_conceptors.npz \
    --repo-type dataset --local-dir conceptors/

# Server (from repo root)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_robocasa \
    --policy.dir checkpoints/pi05_pretrain_human300/multitask_learning/75000 \
    --env ROBOCASA --pytorch --steer
```

### Single env, default steering

```bash
cd examples/robocasa_env
MUJOCO_GL=egl uv run python main.py --env_name CloseFridge --steer
```

### Single env, explicit params

```bash
MUJOCO_GL=egl uv run python main.py --env_name CloseFridge --steer \
    --steering_layer 11 --steering_alpha 0.5 --steering_beta 0.1 \
    --steering_strategy per_step_0
```

### Full task_set with per-env tuned configs

```bash
MUJOCO_GL=egl uv run python eval_all.py \
    --task_set atomic_seen --num_episodes 10 \
    --steer --steering_config ../../experiments/robocasa/best_configs.json
```

Flag names match libero; see `examples/libero_env/README.md#running-with-steering`
for the full table. The only difference is the NPZ task-key source: robocasa
uses `args.env_name` directly, so `--steering_task` is rarely needed.

To produce new tuned configs, see `experiments/robocasa/README.md`.

