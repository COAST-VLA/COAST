"""Tests for Metaworld environment creation and basic functionality.

These tests require a MuJoCo-compatible rendering backend (EGL for headless GPU).
They are marked as ``manual`` so CI (which runs with ``-m "not manual"``) skips them.

Run all MetaWorld tests locally:
    MUJOCO_GL=egl uv run pytest tests/metaworld/test_metaworld_envs.py -v

Run only the pure-logic tests (no rendering / no GPU required):
    uv run pytest tests/metaworld/test_metaworld_envs.py -v -m "not manual"
"""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

# Import from examples/metaworld/ scripts — sys.path must be set before these imports.
# Both this file and tests/robocasa/test_robocasa_main.py do `from main import ...`
# at module load time. Pytest collects them in the same process, so whichever loads
# first caches sys.modules['main'] and the second one's `from main import` returns
# the wrong (cached) module. Pop the cache before our own imports so the next
# example dir on sys.path[0] takes effect.
sys.modules.pop("main", None)
sys.modules.pop("eval_all", None)
_examples_dir = str(Path(__file__).parents[2] / "examples" / "metaworld")
sys.path.insert(0, _examples_dir)

from eval_all import make_env as make_eval_env  # noqa: E402
from main import CAMERA_IDS  # noqa: E402
from main import TASK_TO_PROMPT  # noqa: E402
from main import make_env as make_single_task_env  # noqa: E402
from main import tile_frames  # noqa: E402

SEED = 42
CAMERA_NAMES = ["corner4", "gripperPOV", "corner"]
WIDTH = 224
HEIGHT = 224

# A few representative tasks for parametrized tests.
SAMPLE_TASKS = ["reach-v3", "pick-place-v3", "door-open-v3"]


# ── eval_all.make_env ─────────────────────────────────────────────────────────


@pytest.mark.manual
@pytest.mark.parametrize("env_name", SAMPLE_TASKS)
def test_eval_env_creates_and_runs(env_name):
    """Env can be created, reset, and stepped for each task."""
    env = make_eval_env(
        env_name=env_name,
        num_envs=2,
        width=WIDTH,
        height=HEIGHT,
        seed=SEED,
        camera_names=CAMERA_NAMES,
    )
    try:
        assert env.num_envs == 2

        obs, info = env.reset(seed=SEED)
        assert obs.shape[0] == env.num_envs
        assert "cameras" in info
        # AsyncVectorEnv stacks info into {cam_name: stacked_array}
        for cam_name in CAMERA_NAMES:
            assert cam_name in info["cameras"], f"camera '{cam_name}' missing from info"
            assert info["cameras"][cam_name].shape[0] == env.num_envs

        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)

        assert obs2.shape[0] == env.num_envs
        assert reward.shape == (env.num_envs,)
        assert terminated.shape == (env.num_envs,)
        assert truncated.shape == (env.num_envs,)
        assert "cameras" in info2
        for cam_name in CAMERA_NAMES:
            assert cam_name in info2["cameras"]
            assert info2["cameras"][cam_name].shape[0] == env.num_envs
    finally:
        env.close()


@pytest.mark.manual
@pytest.mark.parametrize("env_name", SAMPLE_TASKS)
def test_camera_images_shape_and_dtype(env_name):
    """Camera images are (num_envs, H, W, 3) uint8 arrays after reset and step."""
    num_envs = 2
    env = make_eval_env(
        env_name=env_name,
        num_envs=num_envs,
        width=WIDTH,
        height=HEIGHT,
        seed=SEED,
        camera_names=CAMERA_NAMES,
    )
    try:
        # AsyncVectorEnv stacks info into {cam_name: (num_envs, H, W, 3)}
        for info in [env.reset(seed=SEED)[1], env.step(env.action_space.sample())[4]]:
            cameras = info["cameras"]
            for cam_name in CAMERA_NAMES:
                assert cam_name in cameras, f"camera '{cam_name}' missing"
                img = cameras[cam_name]
                assert img.shape == (
                    num_envs,
                    HEIGHT,
                    WIDTH,
                    3,
                ), f"'{cam_name}': expected ({num_envs}, {HEIGHT}, {WIDTH}, 3), got {img.shape}"
                assert img.dtype == np.uint8, f"'{cam_name}': expected uint8, got {img.dtype}"
    finally:
        env.close()


@pytest.mark.manual
@pytest.mark.parametrize("env_name", SAMPLE_TASKS)
def test_obs_state_has_at_least_four_dims(env_name):
    """Observation state has at least 4 dims (first 4 are passed to the policy)."""
    env = make_eval_env(
        env_name=env_name,
        num_envs=2,
        width=WIDTH,
        height=HEIGHT,
        seed=SEED,
        camera_names=CAMERA_NAMES,
    )
    try:
        obs, _ = env.reset(seed=SEED)
        assert obs.ndim == 2, f"Expected 2D obs (num_envs, obs_dim), got shape {obs.shape}"
        assert obs.shape[1] >= 4, f"Expected obs_dim >= 4 for policy state input, got {obs.shape[1]}"
    finally:
        env.close()


# ── main.TASK_TO_PROMPT ───────────────────────────────────────────────────────


def test_task_to_prompt_is_nonempty():
    assert len(TASK_TO_PROMPT) > 0


def test_task_to_prompt_all_values_are_nonempty_strings():
    for task, prompt in TASK_TO_PROMPT.items():
        assert isinstance(prompt, str) and prompt.strip(), f"'{task}' maps to an empty or non-string prompt: {prompt!r}"  # noqa: PT018


def test_task_to_prompt_contains_known_tasks():
    """A representative sample of tasks are present in TASK_TO_PROMPT."""
    expected = [
        "pick-place-v3",
        "door-open-v3",
        "drawer-open-v3",
        "button-press-v3",
        "reach-v3",
        "push-v3",
        "assembly-v3",
        "bin-picking-v3",
    ]
    for task in expected:
        assert task in TASK_TO_PROMPT, f"'{task}' missing from TASK_TO_PROMPT"


def test_task_to_prompt_all_keys_end_with_v3():
    """All task names follow the -v3 naming convention."""
    for task in TASK_TO_PROMPT:
        assert task.endswith("-v3"), f"Task '{task}' does not end with '-v3'"


# ── main.CAMERA_IDS ───────────────────────────────────────────────────────────


def test_camera_ids_contains_policy_cameras():
    """CAMERA_IDS includes both cameras used during policy inference."""
    for cam in ["corner4", "gripperPOV"]:
        assert cam in CAMERA_IDS, f"'{cam}' missing from CAMERA_IDS"


def test_camera_ids_values_are_nonneg_ints():
    for cam, idx in CAMERA_IDS.items():
        assert isinstance(idx, int) and idx >= 0, f"CAMERA_IDS['{cam}'] = {idx!r} is not a non-negative integer"  # noqa: PT018


def test_camera_ids_values_are_unique():
    ids = list(CAMERA_IDS.values())
    assert len(ids) == len(set(ids)), f"Duplicate camera IDs found: {ids}"


# ── main.tile_frames ──────────────────────────────────────────────────────────


def test_tile_frames_single():
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    result = tile_frames([frame])
    assert result.shape == (224, 224, 3)


def test_tile_frames_four_frames_2x2():
    frames = [np.ones((224, 224, 3), dtype=np.uint8) * i for i in range(4)]
    result = tile_frames(frames)
    assert result.shape == (448, 448, 3)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 10])
def test_tile_frames_grid_shape(n):
    """Output shape matches expected grid dimensions for any number of frames."""
    h, w = 64, 64
    frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]
    result = tile_frames(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    assert result.shape == (rows * h, cols * w, 3), f"n={n}: expected ({rows * h}, {cols * w}, 3), got {result.shape}"


def test_tile_frames_preserves_dtype():
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    assert tile_frames(frames).dtype == np.uint8


def test_tile_frames_black_padding_for_incomplete_grid():
    """Empty slots in the grid tile are filled with zeros."""
    frames = [np.ones((64, 64, 3), dtype=np.uint8) * 255 for _ in range(3)]
    result = tile_frames(frames)
    # 3 frames → 2 cols, 2 rows; bottom-right slot should be black
    assert result[64:, 64:].max() == 0, "Empty grid slot is not zero-filled"


# ── main.make_env (single-task vectorised env) ────────────────────────────────


@pytest.mark.manual
def test_single_task_env_num_envs():
    """main.make_env produces an env with the requested number of sub-envs."""
    env = make_single_task_env(
        env_name="pick-place-v3",
        num_envs=2,
        width=WIDTH,
        height=HEIGHT,
        seed=SEED,
        camera_names=["corner4", "gripperPOV"],
    )
    try:
        assert env.num_envs == 2
    finally:
        env.close()


@pytest.mark.manual
def test_single_task_env_reset_and_step():
    """main.make_env env resets and steps without error; obs/reward shapes are correct."""
    env = make_single_task_env(
        env_name="pick-place-v3",
        num_envs=2,
        width=WIDTH,
        height=HEIGHT,
        seed=SEED,
        camera_names=["corner4", "gripperPOV"],
    )
    try:
        obs, info = env.reset()
        assert obs.shape[0] == 2
        assert "cameras" in info

        action = env.action_space.sample()
        obs2, reward, _, _, info2 = env.step(action)
        assert obs2.shape[0] == 2
        assert reward.shape == (2,)
        assert "cameras" in info2
    finally:
        env.close()


@pytest.mark.manual
def test_seed_controls_initial_state():
    """Different ``--seed`` values yield different initial env observations.

    Regression test for the claim that MetaWorld's ``env.reset(seed=...)``
    actually randomizes object positions / joint initial angles per seed.
    ``main.py`` uses ``env.reset(seed=args.seed + episode)``; different base
    seeds must produce different per-episode seeds and thus different starts.

    Note: only the *different-seeds → different-state* direction is asserted.
    Same-seed determinism is brittle because successive ``reset(seed=X)`` calls
    on a single env advance internal RNG beyond the seed specification
    (gymnasium quirk — not something pi0.5 eval relies on).
    """
    env_a = make_single_task_env(
        env_name="pick-place-v3",
        num_envs=1,
        width=WIDTH,
        height=HEIGHT,
        seed=100,
        camera_names=["corner4"],
    )
    env_b = make_single_task_env(
        env_name="pick-place-v3",
        num_envs=1,
        width=WIDTH,
        height=HEIGHT,
        seed=200,
        camera_names=["corner4"],
    )
    try:
        obs_a, _ = env_a.reset(seed=100)
        obs_b, _ = env_b.reset(seed=200)

        # Different seeds → different starts (joint angles / object pos differ).
        delta_abs = np.abs(obs_a - obs_b)
        n_differing = int(np.sum(delta_abs > 1e-6))
        max_delta = float(delta_abs.max())
        assert n_differing >= 1 and max_delta > 1e-3, (
            f"expected seed=100 vs seed=200 to differ, got n_differing={n_differing}, max|Δ|={max_delta:.3e}"
        )
    finally:
        env_a.close()
        env_b.close()
