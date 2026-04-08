# Robocasa Client Example

Since [Robocasa](https://robocasa.ai/docs/build/html/index.html) has weird dependencies that we cannot resolve with `openpi`, we need to create a separate virtual env in `examples/robocasa_env`. Thus, the simulation (client) and the model (server) uses websocket to communicate.

## Installation

This is a modified version of the [original setup guide](https://robocasa.ai/docs/build/html/introduction/installation.html).

```bash
cd examples/robocasa_env  # All commands below should be run in this directory.
uv sync
```

Install the package and download assets:

```bash
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

1. **`main.py`** — evaluate a single task (one `env_name`).
2. **`eval_all.py`** — evaluate every task in a task set (e.g. `atomic_seen`, `composite_seen`, `composite_unseen`, `pretrain50`).

Both default to `--split pretrain` (in-distribution object instances). Each episode's video is built by tiling the env's three cameras (`agentview_left`, `agentview_right`, `eye_in_hand`) into one grid frame. RoboCasa **does not support parallel envs** (EGL contexts are not multiprocess-safe), so the grid is across cameras of one env rather than across N parallel envs.

#### Single environment

```bash
cd examples/robocasa_env
MUJOCO_GL=egl uv run python main.py --env_name CloseBlenderLid
```

Common flags:

- `--env_name` — RoboCasa task name (e.g. `CloseBlenderLid`, `OpenCabinet`, `TurnOnMicrowave`).
- `--split` — `pretrain` (default) or `target`.
- `--num_episodes` — Episodes to run (default 1).
- `--max_steps` — Override max steps per episode (default `1.5 * task_horizon`).
- `--replan_steps` — Steps to execute from each action chunk before re-querying (default 5).
- `--render_cameras` — Cameras to tile in the video (default all three).

Videos are written to `examples/robocasa_env/output/single-<split>/<env_name>/episode_<idx>.mp4`.

#### All tasks in a task set

```bash
cd examples/robocasa_env
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen
```

Common flags (in addition to the policy/server/replan flags from `main.py`):

- `--task_set` — name of a task set in `robocasa.utils.dataset_registry.TASK_SET_REGISTRY`. Common choices:
  - `atomic_seen` (18 atomic target tasks)
  - `composite_seen` (16 seen composite target tasks)
  - `composite_unseen` (16 unseen composite target tasks)
  - `target50` (atomic_seen + composite_seen + composite_unseen)
  - `pretrain50` / `pretrain100` / `pretrain200` / `pretrain300`
- `--split` — `pretrain` (default) or `target`. Independent of the task set name; controls which object instances are used.

Per-task videos are written to `examples/robocasa_env/output/<task_set>-<split>/<env_name>/episode_<idx>.mp4`, and an aggregated `results.json` (with per-task and mean success rates) is written to `examples/robocasa_env/output/<task_set>-<split>/results.json`. The summary file is updated incrementally after each task so progress is preserved on early exit.

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
# or for a whole task set:
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen --collect
```

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

