"""Tests that lock in MetaWorld v3's post-success env behavior.

The activation-collection rollout (`examples/metaworld/main.py`) loops to
``success.all()`` or ``max_steps``, *not* per-env termination. So the post-
success activations a metaworld collection records are only well-defined if
MetaWorld v3 keeps the env in the same trajectory after success — i.e.
``terminated`` stays ``False`` so gymnasium 1.0's default
``AutoresetMode.NEXT_STEP`` doesn't auto-reset that env.

Empirically (verified by ``test_metaworld_does_not_terminate_on_success``)
MetaWorld v3 sets ``info["success"]=1`` but leaves ``terminated=truncated=
False`` — only the 500-step ``max_path_length`` triggers ``truncated=True``.
So post-success activations stay aligned to the same episode, no reset, no
data corruption.

These tests guard against either of those invariants flipping: a future
MetaWorld release that terminates on success, or a gymnasium upgrade that
changes the default autoreset mode.

Manual (GPU/render-required, ``MUJOCO_GL=egl``).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

# See test_metaworld_envs.py for the rationale behind this sys.path / cache dance.
sys.modules.pop("main", None)
sys.modules.pop("eval_all", None)
_examples_dir = str(Path(__file__).parents[2] / "examples" / "metaworld")
sys.path.insert(0, _examples_dir)

from main import make_env  # noqa: E402

SEED = 1234
NUM_ENVS = 2
WIDTH = 64  # tiny render — we only care about step semantics, not visuals.
HEIGHT = 64
CAMERAS = ["corner4", "gripperPOV", "corner"]
ENV_NAME = "reach-v3"
# 500 == metaworld v3 max_path_length (truncation point). Tests need to be
# able to observe both success (mid-rollout) and truncation (at the cap).
MAX_STEPS = 500


def _oracle_action(env_index: int, raw_obs: np.ndarray) -> np.ndarray:
    """Use MetaWorld's bundled scripted policy to drive reach-v3 to success.

    raw_obs here is the 39-D MetaWorld observation (the AsyncVectorEnv batches
    the per-env raw obs into ``(num_envs, 39)``).
    """
    from metaworld.policies import sawyer_reach_v3_policy

    policy = sawyer_reach_v3_policy.SawyerReachV3Policy()
    return policy.get_action(raw_obs)


@pytest.mark.manual
def test_metaworld_does_not_terminate_on_success() -> None:
    """At ``info['success']=1``, MetaWorld v3 keeps ``terminated=truncated=False``.

    This is the load-bearing invariant for the metaworld activation-collection
    rollout: the loop runs to ``success.all()`` or ``max_steps`` without
    consulting per-env termination. If MetaWorld ever started terminating on
    success, gymnasium 1.0's default ``AutoresetMode.NEXT_STEP`` would reset
    that env in place and subsequent activations would be from a fresh episode
    saved under the old ``episode_id``.
    """
    env = make_env(ENV_NAME, NUM_ENVS, WIDTH, HEIGHT, SEED, CAMERAS)
    try:
        # AsyncVectorEnv.reset() returns a *batched* obs; the metaworld step
        # loop in main.py uses the proprioceptive part directly, but the
        # scripted policy needs the full 39-D raw obs from each sub-env.
        obs, _ = env.reset(seed=SEED)

        first_success_terminated: list[bool] = []
        first_success_truncated: list[bool] = []
        already_succeeded = [False] * NUM_ENVS
        for _ in range(MAX_STEPS):
            actions = np.stack([_oracle_action(i, obs[i]) for i in range(NUM_ENVS)]).astype(np.float32)
            obs, _reward, terminated, truncated, info = env.step(actions)
            step_success = np.asarray(info.get("success", np.zeros(NUM_ENVS)), dtype=bool)
            for env_id in range(NUM_ENVS):
                if step_success[env_id] and not already_succeeded[env_id]:
                    already_succeeded[env_id] = True
                    first_success_terminated.append(bool(terminated[env_id]))
                    first_success_truncated.append(bool(truncated[env_id]))
            if all(already_succeeded):
                break

        assert any(already_succeeded), (
            f"oracle policy failed to drive any env to success within {MAX_STEPS} steps — test setup broken."
        )
        # Core invariant.
        assert all(t is False for t in first_success_terminated), (
            f"MetaWorld terminated=True at success ({first_success_terminated}); "
            "post-success activations risk corruption from autoreset."
        )
        assert all(t is False for t in first_success_truncated), (
            f"MetaWorld truncated=True at success ({first_success_truncated}); should only truncate at max_path_length."
        )
    finally:
        env.close()


@pytest.mark.manual
def test_metaworld_truncates_at_max_path_length() -> None:
    """The env *does* eventually truncate at ``max_path_length=500``.

    Sanity check that the episode cap is the only termination trigger. If
    metaworld lowered max_path_length silently, the eval defaults
    (``max_steps=300``) would start hitting truncation instead of running
    full-length rollouts.
    """
    env = make_env(ENV_NAME, NUM_ENVS, WIDTH, HEIGHT, SEED, CAMERAS)
    try:
        env.reset(seed=SEED)
        action = np.zeros((NUM_ENVS, env.single_action_space.shape[0]), dtype=np.float32)
        truncated_step = None
        # +1 because step counter is 0-indexed; we expect truncation at exactly
        # max_path_length steps after reset.
        for step in range(600):
            _obs, _r, terminated, truncated, _info = env.step(action)
            if np.asarray(truncated).any():
                truncated_step = step + 1  # 1-indexed step count
                assert not np.asarray(terminated).any(), (
                    "MetaWorld returned terminated=True at truncation. Expected truncation alone."
                )
                break
        assert truncated_step is not None, "env did not truncate within 600 steps"
        assert truncated_step == 500, f"expected truncation at step 500 (max_path_length), got {truncated_step}"
    finally:
        env.close()


@pytest.mark.manual
def test_async_vector_env_autoreset_on_terminated() -> None:
    """Exercise the autoreset directly: force terminated=True via a custom
    wrapper-free path is awkward, so instead we (a) read AsyncVectorEnv's
    advertised autoreset_mode, (b) construct one and verify the default.

    This test guards against a future gymnasium upgrade silently changing the
    default autoreset_mode (e.g. to SAME_STEP), which would change what the
    obs/reward at the terminated-step actually represents.
    """
    import gymnasium as gym

    env = make_env(ENV_NAME, NUM_ENVS, WIDTH, HEIGHT, SEED, CAMERAS)
    try:
        # gymnasium 1.0+ exposes `autoreset_mode`. Older versions might not.
        mode = getattr(env, "autoreset_mode", None)
        assert mode is not None, "gymnasium version too old: no autoreset_mode attribute"
        # NEXT_STEP is the default; SAME_STEP / DISABLED would change semantics.
        assert mode == gym.vector.AutoresetMode.NEXT_STEP, (
            f"AsyncVectorEnv autoreset_mode is {mode!r}, not NEXT_STEP. "
            "Activation collection assumes NEXT_STEP semantics — verify the "
            "metaworld step loop is still correct under the new mode."
        )
    finally:
        env.close()


@pytest.mark.manual
def test_post_success_obs_continues_same_trajectory() -> None:
    """After success, env.step continues stepping the same trajectory, not a
    reset one. Activations captured after success belong to the same episode.

    This is the empirical observation that justifies the metaworld step loop
    not having a per-env early-stop.
    """
    env = make_env(ENV_NAME, NUM_ENVS, WIDTH, HEIGHT, SEED, CAMERAS)
    try:
        obs, _ = env.reset(seed=SEED)
        success_obs = None
        post_success_obs = None
        success_env_id = None
        for _ in range(MAX_STEPS):
            actions = np.stack([_oracle_action(i, obs[i]) for i in range(NUM_ENVS)]).astype(np.float32)
            obs, _r, _t, _tr, info = env.step(actions)
            step_success = np.asarray(info.get("success", np.zeros(NUM_ENVS)), dtype=bool)
            if step_success.any():
                success_env_id = int(np.argmax(step_success))
                success_obs = obs[success_env_id].copy()
                # Step once more; if autoreset fired, the obs would jump to a
                # new initial state (norm distance to obs would spike).
                actions = np.stack([_oracle_action(i, obs[i]) for i in range(NUM_ENVS)]).astype(np.float32)
                obs, _r, _t, _tr, _info = env.step(actions)
                post_success_obs = obs[success_env_id].copy()
                break

        assert success_obs is not None, "oracle policy failed to drive env to success"
        # Continuity: positions in the next obs should be very close to the
        # success obs (a tiny step's worth of motion). A reset would jump >0.1.
        diff = float(np.linalg.norm(post_success_obs[:7] - success_obs[:7]))
        assert diff < 0.05, (
            f"post-success obs jumped by {diff:.4f} (>0.05) — env may have "
            "auto-reset on success. metaworld activation collection assumes "
            "continuous post-success rollout."
        )
    finally:
        env.close()
