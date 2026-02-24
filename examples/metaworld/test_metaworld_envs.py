"""Tests for Metaworld environment creation and basic functionality.

Run with:
    MUJOCO_GL=egl uv run pytest examples/metaworld/test_metaworld_envs.py -v

Skip heavy benchmarks (MT50, ML45-*) with:
    MUJOCO_GL=egl uv run pytest examples/metaworld/test_metaworld_envs.py -v -m "not slow"
"""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from eval_all import make_env as make_benchmark_env
from main import CAMERA_IDS
from main import TASK_TO_PROMPT
from main import make_env as make_single_task_env
from main import tile_frames

# Benchmarks that create 10 or fewer envs (fast to run)
FAST_BENCHMARKS = ["MT10", "ML10-test", "ML10-train"]
# Benchmarks that create many envs (slow)
SLOW_BENCHMARKS = ["MT50", "ML45-test", "ML45-train"]
UNIMPLEMENTED_BENCHMARKS = ["MT1", "ML1"]

SEED = 42
CAMERA_NAMES = ["corner4", "gripperPOV", "corner"]
WIDTH = 224
HEIGHT = 224


# ── eval_all.make_env ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("benchmark_name", FAST_BENCHMARKS)
def test_benchmark_creates_and_runs(benchmark_name):
    """Env can be created, reset, and stepped for each supported benchmark."""
    env = make_benchmark_env(
        benchmark_name,
        env_name=None,
        seed=SEED,
        width=WIDTH,
        height=HEIGHT,
        camera_names=CAMERA_NAMES,
    )
    try:
        assert env.num_envs > 0

        obs, info = env.reset(seed=SEED)
        assert obs.shape[0] == env.num_envs
        assert "cameras" in info
        assert len(info["cameras"]) == env.num_envs

        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)

        assert obs2.shape[0] == env.num_envs
        assert reward.shape == (env.num_envs,)
        assert terminated.shape == (env.num_envs,)
        assert truncated.shape == (env.num_envs,)
        assert "cameras" in info2
        assert len(info2["cameras"]) == env.num_envs
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.parametrize("benchmark_name", SLOW_BENCHMARKS)
def test_heavy_benchmark_creates_and_runs(benchmark_name):
    """Large benchmarks (MT50, ML45-*) can be created, reset, and stepped."""
    env = make_benchmark_env(
        benchmark_name,
        env_name=None,
        seed=SEED,
        width=WIDTH,
        height=HEIGHT,
        camera_names=CAMERA_NAMES,
    )
    try:
        assert env.num_envs > 0
        obs, info = env.reset(seed=SEED)
        assert obs.shape[0] == env.num_envs
        assert "cameras" in info
        assert len(info["cameras"]) == env.num_envs

        action = env.action_space.sample()
        _, reward, _, _, info2 = env.step(action)
        assert reward.shape == (env.num_envs,)
        assert "cameras" in info2
    finally:
        env.close()


@pytest.mark.parametrize("benchmark_name", UNIMPLEMENTED_BENCHMARKS)
def test_unimplemented_benchmarks_raise(benchmark_name):
    """MT1 and ML1 raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        make_benchmark_env(benchmark_name, env_name=None, seed=SEED, camera_names=CAMERA_NAMES)


@pytest.mark.parametrize("benchmark_name", FAST_BENCHMARKS)
def test_camera_images_shape_and_dtype(benchmark_name):
    """Camera images are (H, W, 3) uint8 arrays for every env after reset and step."""
    env = make_benchmark_env(
        benchmark_name,
        env_name=None,
        seed=SEED,
        width=WIDTH,
        height=HEIGHT,
        camera_names=CAMERA_NAMES,
    )
    try:
        for info in [env.reset(seed=SEED)[1], env.step(env.action_space.sample())[4]]:
            for i, cam_dict in enumerate(info["cameras"]):
                for cam_name in CAMERA_NAMES:
                    assert cam_name in cam_dict, f"env[{i}]: camera '{cam_name}' missing"
                    img = cam_dict[cam_name]
                    assert img.ndim == 3, f"env[{i}] '{cam_name}': expected 3D, got {img.ndim}D"
                    assert img.shape == (HEIGHT, WIDTH, 3), (
                        f"env[{i}] '{cam_name}': expected ({HEIGHT}, {WIDTH}, 3), got {img.shape}"
                    )
                    assert img.dtype == np.uint8, f"env[{i}] '{cam_name}': expected uint8, got {img.dtype}"
    finally:
        env.close()


@pytest.mark.parametrize("benchmark_name", FAST_BENCHMARKS)
def test_obs_state_has_at_least_four_dims(benchmark_name):
    """Observation state has at least 4 dims (first 4 are passed to the policy)."""
    env = make_benchmark_env(
        benchmark_name,
        env_name=None,
        seed=SEED,
        width=WIDTH,
        height=HEIGHT,
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
