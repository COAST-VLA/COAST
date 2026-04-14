"""Local-only (GPU + Isaac Sim) smoke tests for ``examples/robolab_env``.

Every test in this file is marked ``manual`` — they require a working Isaac
Sim 5.0 install, an NVIDIA GPU, and ~40 seconds just to boot the simulator.
CI runs with ``-m "not manual"`` and skips them entirely.

The test module itself is importable **without** Isaac Sim installed (``import
main`` only runs inside the ``main_module`` fixture, which is session-scoped
so the simulator boots at most once per pytest session). That keeps CI
collection cheap and errors-free even though CI does not install the
``robolab_env`` venv.

Run them locally from this directory:

    cd examples/robolab_env
    OMNI_KIT_ACCEPT_EULA=YES uv run pytest tests/test_robolab_env.py -m manual -v

Selected parallel-rendering tests build a ``num_envs=4`` Isaac Sim env in one
process. RoboLab's own ``RobolabEnv`` explicitly supports multi-env stepping
(see ``third_party/robolab/robolab/core/environments/env.py``), but freezing +
stage recreation can be fragile — these tests serve as a regression guard.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
import torch

# The whole module is GPU + Isaac Sim territory.
pytestmark = pytest.mark.manual


# ---------------------------------------------------------- session fixtures


@pytest.fixture(scope="session")
def main_module():
    """Import ``main``, booting Isaac Sim exactly once per pytest session.

    ``main.py`` constructs ``AppLauncher`` at module load time, so importing
    the module is what actually boots Omniverse Kit. We defer the import to
    the first test that needs it (rather than doing it at this test file's
    top-level) so pytest collection under CI — where ``isaacsim`` is not
    installed — does not crash.

    Before the import, we replace ``sys.argv`` with
    ``[argv[0], "--headless"]``. Omniverse Kit's bootstrap re-scans
    ``sys.argv`` internally (separately from ``AppLauncher``'s argparser)
    and segfaults on any argument it does not recognise — including the
    pytest ``-m manual`` flag that selected this test in the first place.
    """
    import sys

    sys.argv = [sys.argv[0], "--headless"]
    import main  # noqa: PLC0415 — deliberately lazy, sys.argv must be sanitized first

    return main


@pytest.fixture()
def make_env(main_module):
    """Factory that creates a RoboLab env and guarantees teardown.

    Yields a callable ``make(num_envs=1, task_name=...) -> (env, env_cfg)``.
    Any env it creates is closed automatically when the test finishes, even
    on failure. We go through a factory (rather than a plain fixture) so a
    single test can compare ``num_envs=1`` vs ``num_envs=4`` without fighting
    pytest scoping.
    """
    created = []

    def _make(
        num_envs: int = 1,
        task_name: str = "BananaInBowlTask",
        seed: int = 7,
    ):
        args = main_module.Args(
            task_name=task_name,
            num_envs=num_envs,
            seed=seed,
            max_steps=30,
            replan_steps=4,
        )
        env, env_cfg, _ = main_module.make_env(args)
        created.append(env)
        return env, env_cfg, args

    yield _make

    for env in created:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------- single-env smoke


class TestSingleEnv:
    def test_make_env_reset_and_step(self, make_env) -> None:
        """``num_envs=1`` env can be created, reset, and stepped once."""
        env, env_cfg, _ = make_env(num_envs=1)

        assert env.num_envs == 1
        assert isinstance(env_cfg.instruction, str) and env_cfg.instruction

        obs, _ = env.reset()

        # Obs dict contract — this is what ``build_policy_request`` reads.
        assert "image_obs" in obs and "proprio_obs" in obs
        assert "external_cam" in obs["image_obs"]
        assert "wrist_cam" in obs["image_obs"]
        assert "arm_joint_pos" in obs["proprio_obs"]
        assert "gripper_pos" in obs["proprio_obs"]

        ext = obs["image_obs"]["external_cam"]
        wrist = obs["image_obs"]["wrist_cam"]
        arm = obs["proprio_obs"]["arm_joint_pos"]
        grip = obs["proprio_obs"]["gripper_pos"]

        # Batch dim = num_envs, trailing dim = 3 (RGB) for images.
        assert ext.shape[0] == 1
        assert ext.ndim == 4 and ext.shape[-1] == 3
        assert wrist.shape[0] == 1 and wrist.ndim == 4 and wrist.shape[-1] == 3
        assert arm.shape == (1, 7)
        assert grip.shape == (1, 1)

        # Zero action → one clean step → no exception.
        actions = torch.zeros(1, 8, device=env.device)
        obs2, reward, term, trunc, info = env.step(actions)

        assert obs2["image_obs"]["external_cam"].shape[0] == 1
        assert reward.shape[0] == 1
        assert term.shape[0] == 1
        assert trunc.shape[0] == 1

        # Active env tracking — nothing has terminated yet.
        assert env.active_env_ids == [0]
        assert env.all_terminated is False

    def test_eval_task_end_to_end_with_stub_policy(
        self, make_env, tmp_path: pathlib.Path
    ) -> None:
        """``eval_task`` runs start-to-finish with a stub policy and writes a video.

        This is the integration test that would have caught the imageio/pyav
        and ``observation/state`` server-side bugs discovered during the first
        end-to-end run. We short-circuit the real policy with a stub returning
        zero actions, so the server never needs to be up.
        """

        class StubPolicy:
            def get_server_metadata(self) -> dict:
                return {"stub": True}

            def infer(self, element: dict) -> dict:
                # Payload shape contract — mirrors what the real
                # pi05_droid_jointpos server would receive.
                assert element["observation/exterior_image_1_left"].shape == (
                    224,
                    224,
                    3,
                )
                assert element["observation/wrist_image_left"].shape == (224, 224, 3)
                assert element["observation/joint_position"].shape == (7,)
                assert element["observation/gripper_position"].shape == (1,)
                assert isinstance(element["prompt"], str) and element["prompt"]
                # (action_horizon=10, action_dim=8) — must exceed args.replan_steps.
                return {"actions": np.zeros((10, 8), dtype=np.float32)}

        import main  # noqa: PLC0415 — same instance the fixture already booted

        env, env_cfg, args = make_env(num_envs=1)
        args.max_steps = 10  # keep the smoke test short

        result = main.eval_task(
            env=env,
            env_cfg=env_cfg,
            policy=StubPolicy(),
            args=args,
            output_dir=str(tmp_path),
            collect_session=None,
        )

        assert set(result.keys()) >= {"success_rate", "num_episodes"}
        assert result["num_episodes"] == 1.0
        assert 0.0 <= result["success_rate"] <= 1.0

        videos = sorted(tmp_path.rglob("*.mp4"))
        assert len(videos) == 1, f"expected 1 video, got {len(videos)}: {videos}"
        assert videos[0].stat().st_size > 0, "video file is empty"


# ---------------------------------------------------------- parallel rendering


class TestParallelRendering:
    """Tests that exercise ``num_envs > 1`` — RoboLab's key differentiator.

    RoboLab natively vectorizes rollouts inside one Isaac Sim process. The
    tests below verify three distinct properties that can silently break:

    1. **Batch dim propagates** — every camera/proprio tensor has leading
       shape ``num_envs``.
    2. **Per-env state is independent** — scenes are randomized differently
       per env, so cameras diverge at reset and proprio states drift
       independently.
    3. **Actions are routed per env** — sending a large action to env 0 and
       zero to the rest moves env 0 strictly more than the passive envs.
    """

    NUM_ENVS = 4

    def test_obs_tensors_carry_num_envs_batch_dim(self, make_env) -> None:
        env, _, _ = make_env(num_envs=self.NUM_ENVS)

        obs, _ = env.reset()

        ext = obs["image_obs"]["external_cam"]
        wrist = obs["image_obs"]["wrist_cam"]
        arm = obs["proprio_obs"]["arm_joint_pos"]
        grip = obs["proprio_obs"]["gripper_pos"]

        assert ext.shape[0] == self.NUM_ENVS, (
            f"external_cam batch dim {ext.shape[0]} != {self.NUM_ENVS}"
        )
        assert wrist.shape[0] == self.NUM_ENVS
        assert arm.shape == (self.NUM_ENVS, 7)
        assert grip.shape == (self.NUM_ENVS, 1)

        # Full list of still-running env ids after reset.
        assert sorted(env.active_env_ids) == list(range(self.NUM_ENVS))
        assert env.all_terminated is False

    def test_parallel_envs_are_not_identical_at_reset(self, make_env) -> None:
        """Per-env scene randomization must produce distinct camera views.

        If this test fails, every parallel rollout is observing the *same*
        scene and parallelism buys us nothing — a silent regression that
        would kill eval throughput without raising an error.
        """
        env, _, _ = make_env(num_envs=self.NUM_ENVS)
        obs, _ = env.reset()

        ext = obs["image_obs"]["external_cam"].detach().cpu().numpy()

        # At least one pair of parallel envs must render differently.
        any_different = False
        for i in range(self.NUM_ENVS):
            for j in range(i + 1, self.NUM_ENVS):
                if not np.array_equal(ext[i], ext[j]):
                    any_different = True
                    break
            if any_different:
                break

        assert any_different, (
            f"All {self.NUM_ENVS} parallel envs rendered an identical "
            "external_cam view at reset — scene randomization appears "
            "broken. Check the task's reset/event configuration."
        )

    def test_only_actioned_env_moves(self, make_env) -> None:
        """A large joint delta on env 0 must move env 0 more than the passive envs.

        This verifies that ``env.step(actions)`` actually routes actions
        per-env rather than broadcasting a single action across the batch.
        Passive envs still drift slightly from gravity / contact resolution,
        so we compare magnitudes rather than requiring them to be frozen.
        """
        env, _, _ = make_env(num_envs=self.NUM_ENVS)
        obs, _ = env.reset()

        arm_before = obs["proprio_obs"]["arm_joint_pos"].detach().cpu().numpy()

        actions = torch.zeros(self.NUM_ENVS, 8, device=env.device)
        # Large (but safe) joint delta on env 0; envs 1..3 hold.
        actions[0, :7] = 0.1

        # Accumulate over a few steps so gravity/inertia don't swamp the signal.
        for _ in range(5):
            obs, _, _, _, _ = env.step(actions)

        arm_after = obs["proprio_obs"]["arm_joint_pos"].detach().cpu().numpy()

        env0_delta = float(np.linalg.norm(arm_after[0] - arm_before[0]))
        other_deltas = [
            float(np.linalg.norm(arm_after[i] - arm_before[i]))
            for i in range(1, self.NUM_ENVS)
        ]

        assert env0_delta > max(other_deltas) + 1e-3, (
            f"Expected env 0 (actioned) to move more than the passive envs. "
            f"env0_delta={env0_delta:.4f}, other_deltas={other_deltas}"
        )
        # Envs 1..3 should stay close to their reset pose — drift under a
        # passive controller should be small compared to env 0's commanded move.
        assert max(other_deltas) < env0_delta, (
            "Passive envs drifted at least as much as the actioned env — "
            "something is broadcasting actions across the batch."
        )

    def test_reset_eval_state_unfreezes_all_envs(self, make_env) -> None:
        """After ``reset_eval_state``, no env is marked frozen / terminated."""
        env, _, _ = make_env(num_envs=self.NUM_ENVS)
        env.reset()
        env.reset_eval_state()

        assert env.all_terminated is False
        assert sorted(env.active_env_ids) == list(range(self.NUM_ENVS))
