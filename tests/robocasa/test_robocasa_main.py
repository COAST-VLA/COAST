"""Tests for examples/robocasa_env/main.py.

The robocasa env example lives in its own venv (`examples/robocasa_env/.venv`)
because robocasa pulls in dependencies that conflict with the main openpi venv.
The main project venv used by the test runner does not have ``robocasa`` or
``robosuite`` installed, so we must mock those modules out at import time before
loading ``main`` from the example directory.

These tests focus on the pure-Python pieces of main.py:
- ``tile_frames`` (grid layout helper, copied from metaworld)
- ``build_state`` (concatenation of proprioceptive obs into the 16-dim policy state)
- ``Args`` (CLI dataclass defaults)
- ``CAMERA_KEYS`` (constant mapping)

Env-dependent tests (env creation, reset, step) cannot run from the main venv
because robocasa is not installed there. They should be run inside the
robocasa_env venv:

    cd examples/robocasa_env && uv run pytest test_main.py -v

Run the unit tests in this file with:

    uv run pytest tests/robocasa/test_robocasa_main.py -v
"""

import dataclasses
import math
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest

# robocasa, robosuite, and the openpi_client/robocasa_env example imports must
# be mocked before we load main.py / eval_all.py from examples/robocasa_env/. The
# mock targets match what those modules import at module load time.
_MOCKED_MODULES = [
    "robocasa",
    "robocasa.utils",
    "robocasa.utils.dataset_registry",
    "robocasa.utils.dataset_registry_utils",
    "robocasa.utils.env_utils",
]
for _mod in _MOCKED_MODULES:
    sys.modules.setdefault(_mod, mock.MagicMock())

# Provide concrete implementations for the symbols main.py / eval_all.py import
# by name so we can still call ``build_state``/``tile_frames`` without surprises.
sys.modules["robocasa.utils.dataset_registry_utils"].get_task_horizon = lambda task: 200  # type: ignore[attr-defined]
sys.modules["robocasa.utils.env_utils"].convert_action = lambda a: {"action": a}  # type: ignore[attr-defined]
# Pretend the registry has the task sets the user cares about so eval_all imports cleanly.
sys.modules["robocasa.utils.dataset_registry"].TASK_SET_REGISTRY = {  # type: ignore[attr-defined]
    "atomic_seen": ["CloseBlenderLid", "OpenCabinet"],
    "composite_seen": ["KettleBoiling", "PrepareCoffee"],
    "composite_unseen": ["ArrangeBreadBasket", "BreadSelection"],
    "pretrain50": ["CloseBlenderLid"],
}

# Pop any cached `main`/`eval_all` modules from a sibling test file (e.g.
# tests/metaworld/test_metaworld_envs.py) before our own imports. Both files do
# `from main import ...` at module load time, and pytest collects them in the
# same process — so whichever loaded first wins sys.modules['main'] and the
# second one's `from main import` returns the wrong cached module. Popping
# forces a fresh load from sys.path[0], which we just set to robocasa_env.
sys.modules.pop("main", None)
sys.modules.pop("eval_all", None)
_examples_dir = str(Path(__file__).parents[2] / "examples" / "robocasa_env")
sys.path.insert(0, _examples_dir)

import eval_all  # noqa: E402
from main import CAMERA_KEYS  # noqa: E402
from main import Args  # noqa: E402
from main import build_state  # noqa: E402
from main import eval_task  # noqa: E402
from main import tile_frames  # noqa: E402

# ── Args ──────────────────────────────────────────────────────────────────────


def test_args_defaults():
    args = Args()
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.env_name == "CloseBlenderLid"
    assert args.split == "pretrain"
    assert args.num_episodes == 1
    assert args.max_steps is None
    assert args.replan_steps == 5
    assert args.resize_size == 224
    assert args.fps == 24
    assert args.seed == 7


def test_args_default_render_cameras_are_independent_instances():
    """Each Args() instantiation must get its own list to avoid shared mutable state."""
    a, b = Args(), Args()
    assert a.render_cameras == b.render_cameras
    assert a.render_cameras is not b.render_cameras
    a.render_cameras.append("extra")
    assert "extra" not in b.render_cameras


def test_args_render_cameras_are_valid_camera_keys():
    """Default render cameras must all exist in CAMERA_KEYS."""
    for cam in Args().render_cameras:
        assert cam in CAMERA_KEYS, f"Default render camera '{cam}' not in CAMERA_KEYS"


def test_args_is_dataclass():
    assert dataclasses.is_dataclass(Args)


# ── CAMERA_KEYS ───────────────────────────────────────────────────────────────


def test_camera_keys_contains_expected_cameras():
    expected = {"agentview_left", "agentview_right", "eye_in_hand"}
    assert set(CAMERA_KEYS.keys()) == expected


def test_camera_keys_values_match_obs_format():
    """Values must match the obs dict keys produced by robocasa's gym wrapper."""
    assert CAMERA_KEYS["agentview_left"] == "video.robot0_agentview_left"
    assert CAMERA_KEYS["agentview_right"] == "video.robot0_agentview_right"
    assert CAMERA_KEYS["eye_in_hand"] == "video.robot0_eye_in_hand"


def test_camera_keys_values_use_video_prefix():
    for value in CAMERA_KEYS.values():
        assert value.startswith("video.robot0_"), f"Bad camera key value: {value}"


# ── build_state ───────────────────────────────────────────────────────────────


def test_build_state_concatenates_in_order():
    obs = {
        "state.end_effector_position_relative": np.array([1.0, 2.0, 3.0]),
        "state.end_effector_rotation_relative": np.array([4.0, 5.0, 6.0, 7.0]),
        "state.base_position": np.array([8.0, 9.0, 10.0]),
        "state.base_rotation": np.array([11.0, 12.0, 13.0, 14.0]),
        "state.gripper_qpos": np.array([15.0, 16.0]),
    }
    state = build_state(obs)
    assert state.shape == (16,)
    expected = np.arange(1.0, 17.0)
    np.testing.assert_array_equal(state, expected)


def test_build_state_returns_16_dims():
    """The robocasa policy expects exactly 16 proprioceptive dims."""
    obs = {
        "state.end_effector_position_relative": np.zeros(3),
        "state.end_effector_rotation_relative": np.zeros(4),
        "state.base_position": np.zeros(3),
        "state.base_rotation": np.zeros(4),
        "state.gripper_qpos": np.zeros(2),
    }
    state = build_state(obs)
    assert state.shape == (16,)


def test_build_state_preserves_dtype():
    obs = {
        "state.end_effector_position_relative": np.zeros(3, dtype=np.float32),
        "state.end_effector_rotation_relative": np.zeros(4, dtype=np.float32),
        "state.base_position": np.zeros(3, dtype=np.float32),
        "state.base_rotation": np.zeros(4, dtype=np.float32),
        "state.gripper_qpos": np.zeros(2, dtype=np.float32),
    }
    assert build_state(obs).dtype == np.float32


def test_build_state_field_layout():
    """Verify each slice of the output corresponds to the right field."""
    obs = {
        "state.end_effector_position_relative": np.array([0.1, 0.2, 0.3]),
        "state.end_effector_rotation_relative": np.array([0.4, 0.5, 0.6, 0.7]),
        "state.base_position": np.array([0.8, 0.9, 1.0]),
        "state.base_rotation": np.array([1.1, 1.2, 1.3, 1.4]),
        "state.gripper_qpos": np.array([1.5, 1.6]),
    }
    state = build_state(obs)
    np.testing.assert_array_equal(state[0:3], obs["state.end_effector_position_relative"])
    np.testing.assert_array_equal(state[3:7], obs["state.end_effector_rotation_relative"])
    np.testing.assert_array_equal(state[7:10], obs["state.base_position"])
    np.testing.assert_array_equal(state[10:14], obs["state.base_rotation"])
    np.testing.assert_array_equal(state[14:16], obs["state.gripper_qpos"])


# ── tile_frames ───────────────────────────────────────────────────────────────


def test_tile_frames_single():
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    result = tile_frames([frame])
    assert result.shape == (224, 224, 3)


def test_tile_frames_three_frames_2x2_grid():
    """3 frames → 2x2 grid (one black slot)."""
    frames = [np.ones((64, 64, 3), dtype=np.uint8) * 255 for _ in range(3)]
    result = tile_frames(frames)
    assert result.shape == (128, 128, 3)
    # Bottom-right slot must be black-padded.
    assert result[64:, 64:].max() == 0


def test_tile_frames_four_frames_2x2_grid():
    frames = [np.ones((224, 224, 3), dtype=np.uint8) * i for i in range(4)]
    result = tile_frames(frames)
    assert result.shape == (448, 448, 3)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 10])
def test_tile_frames_grid_shape(n):
    """Output shape matches metaworld's grid layout for any N."""
    h, w = 32, 32
    frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]
    result = tile_frames(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    assert result.shape == (rows * h, cols * w, 3)


def test_tile_frames_preserves_dtype():
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    assert tile_frames(frames).dtype == np.uint8


def test_tile_frames_places_frames_in_row_major_order():
    """Frame i should land at grid position (i // cols, i % cols)."""
    n = 3
    h, w = 16, 16
    cols = math.ceil(math.sqrt(n))  # = 2
    # Mark each frame with a unique value so we can find it.
    frames = [np.full((h, w, 3), i + 1, dtype=np.uint8) for i in range(n)]
    grid = tile_frames(frames)
    for i in range(n):
        r, c = divmod(i, cols)
        cell = grid[r * h : (r + 1) * h, c * w : (c + 1) * w]
        assert cell[0, 0, 0] == i + 1, f"Frame {i} not at position (row={r}, col={c})"


def test_tile_frames_matches_metaworld_implementation():
    """tile_frames should produce identical output to metaworld's implementation."""
    sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "metaworld"))
    from main import tile_frames as metaworld_tile_frames

    frames = [np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(5)]
    np.testing.assert_array_equal(tile_frames(frames), metaworld_tile_frames(frames))


# ── eval_task signature ───────────────────────────────────────────────────────


def test_eval_task_is_callable_and_documented():
    """eval_task must be a top-level function (so eval_all.py can import it)."""
    assert callable(eval_task)
    assert eval_task.__doc__ is not None
    assert eval_task.__doc__.strip() != ""


def test_eval_task_signature():
    """eval_task takes (env_name, policy, args, output_dir)."""
    import inspect

    sig = inspect.signature(eval_task)
    params = list(sig.parameters)
    assert params == [
        "env_name",
        "policy",
        "args",
        "output_dir",
    ], f"Expected (env_name, policy, args, output_dir), got {params}"


def test_eval_task_args_field_compatibility():
    """eval_task's args parameter is duck-typed; main.Args must provide all the
    fields it accesses (split, num_episodes, max_steps, replan_steps, resize_size,
    render_cameras, fps, seed)."""
    required = {"split", "num_episodes", "max_steps", "replan_steps", "resize_size", "render_cameras", "fps", "seed"}
    field_names = {f.name for f in dataclasses.fields(Args)}
    missing = required - field_names
    assert not missing, f"main.Args missing fields needed by eval_task: {missing}"


# ── eval_all.Args ─────────────────────────────────────────────────────────────


def test_eval_all_args_defaults():
    args = eval_all.Args()
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.task_set == "atomic_seen"
    assert args.split == "pretrain"
    assert args.num_episodes == 1
    assert args.max_steps is None
    assert args.replan_steps == 5
    assert args.resize_size == 224
    assert args.fps == 24
    assert args.seed == 7


def test_eval_all_args_default_render_cameras_match_main():
    """eval_all.Args.render_cameras default should match main.Args.render_cameras default."""
    assert eval_all.Args().render_cameras == Args().render_cameras


def test_eval_all_args_default_render_cameras_are_independent_instances():
    a, b = eval_all.Args(), eval_all.Args()
    assert a.render_cameras is not b.render_cameras


def test_eval_all_args_provides_eval_task_required_fields():
    """eval_all.Args must satisfy the same duck-typed contract as main.Args
    so it can be passed to eval_task."""
    required = {"split", "num_episodes", "max_steps", "replan_steps", "resize_size", "render_cameras", "fps", "seed"}
    field_names = {f.name for f in dataclasses.fields(eval_all.Args)}
    missing = required - field_names
    assert not missing, f"eval_all.Args missing fields needed by eval_task: {missing}"


def test_eval_all_imports_eval_task_from_main():
    """eval_all.py should reuse main.eval_task rather than duplicating it."""
    assert eval_all.eval_task is eval_task


def test_eval_all_main_rejects_unknown_task_set():
    """Passing an unknown task_set must raise a clear ValueError before any
    server connection happens."""
    args = eval_all.Args(task_set="this_task_set_does_not_exist")
    with pytest.raises(ValueError, match="Unknown task_set"):
        eval_all.main(args)


def test_eval_all_supported_task_sets_present_in_registry_mock():
    """Sanity check that the user-facing task sets exist as registry keys after
    our test-time mock setup."""
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    for name in ["atomic_seen", "composite_seen", "composite_unseen"]:
        assert name in TASK_SET_REGISTRY
