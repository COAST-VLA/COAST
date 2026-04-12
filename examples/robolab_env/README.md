# RoboLab Client Example

[RoboLab](https://research.nvidia.com/labs/srl/projects/robolab) is NVIDIA's 120-task evaluation benchmark for robot manipulation policies, built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab). Because it depends on Isaac Sim / Isaac Lab and a CUDA-specific PyTorch build, it cannot share the main `openpi` virtual environment — we install it in a dedicated venv under `examples/robolab_env` and communicate with the policy server over WebSocket, the same pattern used by `examples/robocasa_env` and `examples/libero_env`.

Unlike robocasa/libero, RoboLab natively vectorizes episodes (`num_envs` parallel rollouts inside one Isaac Sim process), so `main.py` drives `num_envs` episodes per "run" and loops `num_runs` times. `num_envs=1` matches the robocasa/libero flow most closely.

- `main.py` evaluates one RoboLab task.
- `eval_all.py` evaluates all (or a filtered subset of) tasks, launching one `main.py` subprocess per task.

## Requirements

| Dependency | Version |
|---|---|
| Isaac Sim | 5.0 |
| Isaac Lab | 2.2.0 |
| Python | 3.11 |
| Linux | Ubuntu 22.04+ |

## Installation

Initialize submodules first if you have not already:

```bash
git submodule update --init --recursive third_party/robolab
```

Then sync the dedicated environment:

```bash
cd examples/robolab_env
uv venv --python 3.11
uv pip install "setuptools<81"
uv sync
```

Verify the install by listing RoboLab's registered tasks. Isaac Sim prompts interactively for EULA acceptance on first boot — set `OMNI_KIT_ACCEPT_EULA=YES` to accept non-interactively (required on every invocation):

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run python ../../third_party/robolab/scripts/check_registered_envs.py
```

A successful run prints the full 120-task benchmark registry. Warnings about `Replicator:Annotators` and `omni.physx.plugin` during shutdown are harmless.

## Serving the RoboLab Policy

RoboLab's [`docs/inference.md`](../../third_party/robolab/docs/inference.md) documents four DROID joint-position configs, all served from a public PI bucket. We've ported `pi05_droid_jointpos` into this repo's `src/openpi/training/config.py` (and companions from `xuningy/openpi`).

### Start the Server

```bash
# Terminal 1 (repo root, main openpi venv) — server pinned to a free GPU
export CUDA_VISIBLE_DEVICES=0
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

The checkpoint (~12 GB) downloads to `~/.cache/openpi/` on the first run and is reused thereafter. Norm stats ship inside the checkpoint bundle.

### Run Evaluation

#### Single task

```bash
# Terminal 2 (robolab_env venv)
cd examples/robolab_env
OMNI_KIT_ACCEPT_EULA=YES uv run python main.py --headless --task-name BananaInBowlTask
```

See [`tasks.py`](tasks.py) for a pure-Python snapshot of all 120 registered task names and their tags.

Output layout for a single run:

```
examples/robolab_env/output/single-<instruction_type>/<task_name>/
└── run_<run_idx:02d>_env<env_id:02d>.mp4
```

Each video tiles the external and wrist cameras side-by-side. Final `success_rate=A/B` is printed to stdout after the last run completes.

#### All tasks (parallel subprocesses)

`eval_all.py` runs every task (or a filtered subset) by launching one `main.py` subprocess per task. Each subprocess boots its own Isaac Sim process. Isaac Sim is heavy (~35 s boot, 10+ GB VRAM), so unlike robocasa/libero, **the default is ``--num_workers 1``** (sequential). Increase only if you have multiple GPUs.

```bash
cd examples/robolab_env

# All 120 tasks, sequential (recommended for single GPU):
OMNI_KIT_ACCEPT_EULA=YES uv run python eval_all.py

# Filter by tag:
OMNI_KIT_ACCEPT_EULA=YES uv run python eval_all.py --tag simple

# Specific tasks:
OMNI_KIT_ACCEPT_EULA=YES uv run python eval_all.py \
    --tasks BananaInBowlTask RubiksCubeTask

# Parallel envs per task (RoboLab's native vectorization):
OMNI_KIT_ACCEPT_EULA=YES uv run python eval_all.py \
    --num_envs 4 --num_runs 2   # 8 episodes per task
```

Output layout:

```
examples/robolab_env/output/<filter>-<instruction_type>/
├── results.json
├── parallel_logs/
│   ├── task_000_BananaInBowlTask.log
│   ├── task_001_RubiksCubeTask.log
│   └── ...
├── BananaInBowlTask/
│   └── run_00_env00.mp4
└── RubiksCubeTask/
    └── run_00_env00.mp4
```

`results.json` is saved incrementally after each task completes. The final version is sorted by success rate, descending.

## Known Quirks

- **Isaac Sim hijacks root logging**: `logging.basicConfig` is a no-op once `isaaclab.app.AppLauncher` has run. `main.py` uses `print(..., flush=True)` for its own status lines instead of `logger.info`.
- **`simulation_app.close()` can hard-exit**: it sometimes calls `os._exit` from `finally`, silently dropping any stdout that hasn't been flushed. `main.py` prints the final `success_rate` line *inside* the `try` block before reaching the cleanup `finally`.
- **Video writer**: `main.py` uses RoboLab's own `robolab.core.utils.video_utils.VideoWriter` (cv2-based H.264) rather than `imageio[pyav]`. The current `av==17.0.0` has a `write_frame` regression that leaves `stream.codec_context.time_base` unset. `av<16` is pinned anyway.

## Tests

All simulator tests live under `tests/` and are marked `manual` because they require a working Isaac Sim 5.0 install and an NVIDIA GPU (~40 s just to boot Kit). CI runs with `-m "not manual"` and skips them entirely — it does not `uv sync` this venv at all.

Run the full suite locally from this directory:

```bash
CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES uv run pytest tests/ -m manual -v
```

Single test:

```bash
CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES uv run pytest \
    tests/test_robolab_env.py::TestParallelRendering::test_parallel_envs_are_not_identical_at_reset \
    -m manual -v
```

The suite finishes in ~85 s on an L40 and covers:

- **Single-env smoke** — `make_env`, `reset`, `step` with the right obs-dict keys and tensor shapes; full `eval_task` loop with a stub policy (guards against the pyav / `observation/state` / hard-exit regressions).
- **Parallel rendering (`num_envs=4`)** — batch dim propagates on every camera/proprio tensor, per-env scene randomization produces distinct camera views, actions route per-env (only env 0 moves when only env 0 is actioned), and `reset_eval_state()` correctly unfreezes all envs.

The `test_robolab_env.py` module is importable **without** Isaac Sim installed — `import main` only runs inside the session-scoped `main_module` fixture — so CI collection works cleanly. If you ever wire this example's venv into CI, you can keep the `manual` skip and nothing changes.

## Porting Notes: `pi05_droid_jointpos`

The config in this repo was ported from [`xuningy/openpi`](https://github.com/xuningy/openpi), which RoboLab's own `docs/inference.md` points users to. Two small changes were required in our fork:

1. **`src/openpi/training/config.py`** — added the `pi05_droid_jointpos` `TrainConfig`. All dependencies (`_transforms.AbsoluteActions`, `make_bool_mask`, `droid_policy.DroidInputs`/`DroidOutputs`, `SimpleDataConfig`, `AssetsConfig`) already existed; no new files. The `pi0_droid_jointpos` and `pi0_fast_droid_jointpos` siblings from `xuningy/openpi` have **not** been ported — add them the same way if you need those backends.

2. **`src/openpi/policies/policy.py`** — the pre-transform batching check read `obs["observation/state"]` directly, which DROID payloads don't contain (they send `observation/joint_position` + `observation/gripper_position` and let `DroidInputs` build `state` internally). Changed to `obs.get("observation/state")` with a `None` guard so batched callers still work but DROID unbatched inference is unblocked.
