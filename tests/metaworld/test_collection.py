"""Tests for metaworld activation collection via ``main.py --collect`` /
``eval_all.py --collect``.

Unit tests (no GPU) cover CLI argument handling, the per-env bookkeeping class,
metadata shapes, and the lazy-import contract (normal eval must not pull in
torch / openpi). Integration tests (@pytest.mark.manual) exercise the full
pipeline end to end and validate the on-disk schema.

Run unit tests only:
    uv run pytest tests/metaworld/test_collection.py -v -m "not manual"

Run everything (needs a GPU, a PyTorch checkpoint, and EGL):
    MUJOCO_GL=egl uv run pytest tests/metaworld/test_collection.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

# See tests/metaworld/test_metaworld_envs.py for the rationale: pytest may collect
# examples/{metaworld,libero,robocasa}/main.py in the same process, and the first
# one caches sys.modules['main']. Pop before our imports so examples/metaworld/
# lands on sys.path[0] first.
sys.modules.pop("main", None)
sys.modules.pop("eval_all", None)
_examples_dir = str(Path(__file__).parents[2] / "examples" / "metaworld")
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)

import eval_all as mw_eval_all  # noqa: E402
import main as mw_main  # noqa: E402

# Where integration tests look for a PyTorch checkpoint. Override via env var.
CHECKPOINT_DIR = os.environ.get(
    "METAWORLD_COLLECT_TEST_CKPT",
    "checkpoints/pi05_metaworld/pi05_metaworld_test/5000/",
)


# ── MetaworldCollectState ────────────────────────────────────────────────────


def test_collect_state_initial_zeros():
    s = mw_main.MetaworldCollectState.new(num_envs=3)
    assert s.num_envs == 3
    assert s.inference_step == 0
    assert s.cumulative_reward.shape == (3,)
    assert np.all(s.cumulative_reward == 0)
    assert np.all(s.reward_at_last_inference == 0)
    assert np.all(s.steps_to_success == -1)
    assert s.per_step_rewards == [[], [], []]
    assert s.per_step_success == [[], [], []]


def test_collect_state_record_step_tracks_per_env():
    s = mw_main.MetaworldCollectState.new(num_envs=2)
    s.record_step(np.array([1.0, 2.5]), np.array([False, False]), step=0)
    s.record_step(np.array([0.5, 0.0]), np.array([True, False]), step=1)

    assert s.per_step_rewards[0] == [1.0, 0.5]
    assert s.per_step_rewards[1] == [2.5, 0.0]
    assert s.per_step_success[0] == [False, True]
    assert s.per_step_success[1] == [False, False]
    np.testing.assert_allclose(s.cumulative_reward, [1.5, 2.5])
    # env 0 hit success on step 1, env 1 never did
    assert s.steps_to_success[0] == 1
    assert s.steps_to_success[1] == -1


def test_collect_state_advance_inference_step_snapshots_reward():
    s = mw_main.MetaworldCollectState.new(num_envs=2)
    s.record_step(np.array([1.0, 0.0]), np.array([False, False]), step=0)
    s.advance_inference_step()
    assert s.inference_step == 1
    np.testing.assert_allclose(s.reward_at_last_inference, [1.0, 0.0])
    s.record_step(np.array([0.5, 2.0]), np.array([False, False]), step=1)
    # reward_since_last should now be cumulative - reward_at_last_inference
    meta = s.snapshot_step_metadata(env_id=1, step=1, task_name="reach-v3", episode_id=0, prompt="x", success=False)
    assert meta["reward_since_last_inference"] == pytest.approx(2.0)


def test_collect_state_step_metadata_has_required_keys():
    s = mw_main.MetaworldCollectState.new(num_envs=1)
    meta = s.snapshot_step_metadata(env_id=0, step=5, task_name="reach-v3", episode_id=2, prompt="reach", success=False)
    expected = {
        "task_name",
        "episode_id",
        "env_id",
        "step",
        "inference_step",
        "prompt",
        "cumulative_reward",
        "success_so_far",
        "reward_since_last_inference",
    }
    assert set(meta.keys()) == expected
    assert meta["task_name"] == "reach-v3"
    assert meta["episode_id"] == 2
    assert meta["step"] == 5


def test_collect_state_episode_metadata_has_required_keys():
    s = mw_main.MetaworldCollectState.new(num_envs=1)
    s.record_step(np.array([2.5]), np.array([True]), step=0)
    meta = s.episode_metadata(
        env_id=0,
        task_name="reach-v3",
        episode_id=0,
        prompt="reach",
        success=True,
        policy_dir="/tmp/ckpt/5000/",
        config_name="pi05_metaworld",
    )
    expected = {
        "task_name",
        "episode_id",
        "env_id",
        "episode_success",
        "total_reward",
        "steps_to_success",
        "total_env_steps",
        "total_inference_steps",
        "prompt",
        "checkpoint_dir",
        "config_name",
    }
    assert set(meta.keys()) == expected
    assert meta["episode_success"] is True
    assert meta["total_reward"] == pytest.approx(2.5)
    assert meta["total_env_steps"] == 1
    assert meta["steps_to_success"] == 0


# ── CLI / Args ───────────────────────────────────────────────────────────────


def test_main_args_defaults_normal_eval():
    args = mw_main.Args()
    assert args.collect is False
    assert args.collect_output_dir == "./activations"
    assert isinstance(args.policy, mw_main.PolicyArgs)
    assert args.policy.config == "pi05_metaworld"


def test_eval_all_args_defaults():
    args = mw_eval_all.Args()
    assert args.collect is False
    assert args.gpus == []
    assert args.tasks == []
    assert args.split == "train"
    assert args.collect_output_dir == "./activations"


def test_eval_all_gpus_without_collect_raises():
    args = mw_eval_all.Args(gpus=[0, 1], collect=False)
    with pytest.raises(ValueError, match="--gpus"):
        mw_eval_all.main(args)


def test_checkpoint_step_derivation():
    # Trailing slash, middle path, simple name — all should give the same basename.
    assert Path("checkpoints/pi05_metaworld/pi05_metaworld_test/5000/").name == "5000"
    assert Path("/x/y/5000").name == "5000"
    assert Path("5000").name == "5000"


# ── Lazy import contract ──────────────────────────────────────────────────────


def test_load_policy_normal_eval_does_not_import_openpi(monkeypatch):
    """Normal eval path must not trigger torch / openpi submodule imports."""
    # Remove cached imports (if any) so we can detect fresh ones. raising=False
    # lets us delete keys that may already be absent.
    probe_modules = [
        "openpi.models_pytorch.convert",
        "openpi.policies.policy_config",
        "openpi.training.config",
    ]
    for name in probe_modules:
        monkeypatch.delitem(sys.modules, name, raising=False)

    class _FakeWSPolicy:
        def __init__(self, host, port):
            self._host = host
            self._port = port

        def get_server_metadata(self):
            return {"mode": "fake"}

    monkeypatch.setattr(mw_main._websocket_client_policy, "WebsocketClientPolicy", _FakeWSPolicy)  # noqa: SLF001

    policy, extras = mw_main.load_policy(mw_main.Args(collect=False))
    assert isinstance(policy, _FakeWSPolicy)
    assert extras == {}
    for name in probe_modules:
        assert name not in sys.modules, f"{name} was imported for normal eval"


# ── Integration: end-to-end collection ────────────────────────────────────────


def _skip_if_no_checkpoint():
    if not Path(CHECKPOINT_DIR).exists():
        pytest.skip(
            f"PyTorch checkpoint not found at {CHECKPOINT_DIR}. "
            "Set METAWORLD_COLLECT_TEST_CKPT env var to point at a 5000/ or similar dir."
        )


@pytest.mark.manual
def test_main_collect_end_to_end(tmp_path):
    """``main.py --collect`` produces a step-0 activation tree with the expected layout."""
    _skip_if_no_checkpoint()

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    result = subprocess.run(
        [
            "uv",
            "run",
            "examples/metaworld/main.py",
            "--collect",
            "--env_name=reach-v3",
            "--num_envs=1",
            "--num_episodes=1",
            "--max_steps=15",  # at least 2 replan windows (replan_steps=10)
            "--replan_steps=10",
            f"--policy.dir={CHECKPOINT_DIR}",
            f"--collect_output_dir={tmp_path}",
            f"--output_dir={tmp_path / 'eval'}",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"main.py --collect failed\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}")

    ckpt_step = Path(CHECKPOINT_DIR).name
    ep_dir = tmp_path / ckpt_step / "reach-v3" / "episode_000_env_000"
    assert ep_dir.exists(), f"Episode dir missing: {ep_dir}"

    # Episode-level files
    assert (ep_dir / "metadata.json").exists()
    assert (ep_dir / "rewards.npz").exists()
    with open(ep_dir / "metadata.json") as f:
        ep_meta = json.load(f)
    assert ep_meta["task_name"] == "reach-v3"
    assert ep_meta["episode_id"] == 0
    assert ep_meta["env_id"] == 0
    assert ep_meta["config_name"] == "pi05_metaworld"

    rewards = np.load(ep_dir / "rewards.npz")
    for key in ("per_step_reward", "cumulative_reward", "success_at_step"):
        assert key in rewards.files

    # Step 0 directory
    step_dir = ep_dir / "step_0000"
    assert step_dir.exists(), f"Step dir missing: {step_dir}"
    for fname in ("denoising.npz", "adarms_cond.npz", "suffix_residual.npz", "suffix_mlp_hidden.npz", "metadata.json"):
        assert (step_dir / fname).exists(), f"Missing: {step_dir / fname}"

    denoising = np.load(step_dir / "denoising.npz")
    assert denoising["all_x_t"].shape == (10, 32, 32)
    assert denoising["all_v_t"].shape == (10, 32, 32)

    adarms = np.load(step_dir / "adarms_cond.npz")
    assert adarms["all_adarms_cond"].shape == (10, 1024)

    residual = np.load(step_dir / "suffix_residual.npz")
    assert residual["all_suffix_residual"].shape == (10, 4, 32, 1024)

    mlp = np.load(step_dir / "suffix_mlp_hidden.npz")
    assert mlp["all_suffix_mlp_hidden"].shape == (10, 4, 32, 4096)

    # Eval artifact (video) must also be written under --output_dir even in collect mode.
    assert (tmp_path / "eval" / "episode_000.mp4").exists()


@pytest.mark.manual
def test_eval_all_collect_two_tasks(tmp_path):
    """``eval_all.py --collect --tasks t1 t2`` populates both task trees."""
    _skip_if_no_checkpoint()

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    result = subprocess.run(
        [
            "uv",
            "run",
            "examples/metaworld/eval_all.py",
            "--collect",
            "--tasks",
            "reach-v3",
            "push-v3",
            "--num_envs=1",
            "--num_episodes=1",
            "--max_steps=15",
            "--replan_steps=10",
            f"--policy.dir={CHECKPOINT_DIR}",
            f"--collect_output_dir={tmp_path}",
            f"--output_dir={tmp_path / 'eval'}",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"eval_all.py --collect failed\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}")

    ckpt_step = Path(CHECKPOINT_DIR).name
    for task in ("reach-v3", "push-v3"):
        ep_dir = tmp_path / ckpt_step / task / "episode_000_env_000"
        assert ep_dir.exists(), f"Missing {ep_dir}"
        assert (ep_dir / "step_0000" / "denoising.npz").exists()

    # Shared results.json for both tasks.
    results_path = tmp_path / "eval" / "results.json"
    assert results_path.exists()
    with open(results_path) as f:
        summary = json.load(f)
    assert "per_task" in summary
    assert set(summary["per_task"].keys()) == {"reach-v3", "push-v3"}
