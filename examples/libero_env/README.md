# LIBERO Client Example

LIBERO needs its own Python 3.8 environment, so this example follows the same separate-client pattern as `examples/robocasa_env`: the simulation runs in `examples/libero_env`, the policy server stays in the main repo environment, and the two talk over WebSocket.

Unlike the old `examples/libero` setup, this version is organized like the newer env examples:
- `main.py` evaluates one LIBERO task.
- `eval_all.py` evaluates every task in one LIBERO suite.

## Installation

Initialize submodules first if you have not already:

```bash
git submodule update --init --recursive
```

Then sync the dedicated environment and write LIBERO's default config for this checkout:

```bash
cd examples/libero_env
uv sync
uv run python setup_libero_config.py
```

`~/.libero/config.yaml` is LIBERO's default config file. It tells LIBERO where to find the benchmark, assets, init states, and datasets. Rerun `setup_libero_config.py` if this checkout moves.

If EGL gives you MuJoCo rendering issues, rerun the client commands below with `MUJOCO_GL=glx` instead of `MUJOCO_GL=egl`.

## Serving the LIBERO Policy

Start the policy server from the repo root in a separate terminal:

```bash
uv run scripts/serve_policy.py --env LIBERO
```

To serve a specific checkpoint instead of the default one:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero"
```

## Run Evaluation

### Single task

```bash
cd examples/libero_env
MUJOCO_GL=egl uv run python main.py --task_suite_name libero_spatial --task_id 0
```

Common flags:
- `--task_suite_name` — one of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`.
- `--task_id` — task index within the suite.
- `--num_episodes` — number of fixed initial states to evaluate.
- `--max_steps` — optional override for the suite default horizon.
- `--num_steps_wait` — number of no-op settling steps before the policy starts acting.
- `--replan_steps` — number of planned actions to execute before re-querying the server.
- `--render_cameras` — cameras tiled into the saved video. Defaults to `agentview` and `eye_in_hand`.

Videos are written to `examples/libero_env/output/single-<task_suite_name>/<task_id>-<task_name>/episode_<idx>.mp4`.

### Full suite

```bash
cd examples/libero_env
MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_spatial
```

Common flags are the same as `main.py`, except `eval_all.py` runs every task in the selected suite and writes:
- Per-task videos to `examples/libero_env/output/<task_suite_name>/<task_id>-<task_name>/episode_<idx>.mp4`
- Aggregate results to `examples/libero_env/output/<task_suite_name>/results.json`

The results file is updated after each task so partial progress is preserved if a long run stops early.

## Activation Collection

For mech-interp work you can have the policy server save per-step intermediate
activations to disk while a libero rollout runs. This uses a separate
"collection-mode" policy server that wraps the policy in `CollectingPolicy`
and writes the same on-disk format as `scripts/collect_activations.py`
(metaworld's collector). Activations live entirely on the **server's**
filesystem — the libero client never touches them, so the client and server
can be on different machines.

Start the collection-mode server from the repo root in one terminal:

```bash
# Terminal 1 (main openpi venv) — server pinned to GPU 0
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi05_libero \
    --policy.dir="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero"
```

Then run a libero rollout with `--collect` from this directory:

```bash
# Terminal 2 (libero_env venv)
cd examples/libero_env
MUJOCO_GL=egl uv run python eval_all.py --task_suite_name libero_spatial --collect
# or for a single task:
MUJOCO_GL=egl uv run python main.py --task_suite_name libero_spatial --task_id 0 --collect
```

Notes:
- Collection mode requires `--pytorch` on the server. `infer_with_intermediates`
  is implemented for the PyTorch backend only.
- Use the local cache path `$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero`,
  not the `gs://` URL — `--pytorch` + `gs://` has a known bug in
  `ensure_pytorch_checkpoint`. To pre-populate the cache, run plain
  `eval_all.py` once against a non-collection server, or run:
  `uv run python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"`.
- A collection-mode server **rejects** plain inference requests. If you want to
  also run regular eval, start a separate non-collection server on a different
  port.
- The server's `--output-dir` is on the **server's** filesystem. With
  `--output-dir ./activations`, files land at
  `./activations/pi05_libero/<task_name>/episode_NNN_env_000/step_NNNN/`
  relative to wherever the server was launched from.
- `eval_all.py --collect` defaults to 2 episodes per task. Override with
  `--num_episodes N`. Each episode uses the next deterministic initial state
  from LIBERO's task suite, so reproducibility is built-in.
