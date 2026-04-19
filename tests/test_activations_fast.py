"""Schema validator for pi0-fast activation datasets (``fast_v1`` schema).

Parallels ``tests/test_activations.py`` (which validates the diffusion ``v1``
schema with ``denoising.npz`` / ``adarms_cond.npz`` / ``suffix_residual.npz`` /
``suffix_mlp_hidden.npz``). The pi0-fast autoregressive collector writes a
different per-step file set — per-token hidden states rather than per-
denoising-step tensors — so the shapes and invariants are different and
need their own validator.

Run after collecting activations, pointing at one task directory:

    # metaworld pi0-fast
    ACTIVATIONS_FAST_DIR=pi0fast-metaworld-activations-v1-15env/2500/reach-v3 \\
        uv run pytest tests/test_activations_fast.py -v

    # libero pi0-fast
    ACTIVATIONS_FAST_DIR=pi0fast-libero-activations-v1-2000-15env/2000/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket \\
        uv run pytest tests/test_activations_fast.py -v

The env-var gate is the same as ``tests/test_activations.py`` (default path is
a common metaworld location; tests skip cleanly when the directory is missing).
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

ACTIVATIONS_FAST_DIR = os.environ.get(
    "ACTIVATIONS_FAST_DIR",
    "pi0fast-metaworld-activations-v1-15env/2500/reach-v3",
)


@pytest.fixture
def act_dir():
    d = pathlib.Path(ACTIVATIONS_FAST_DIR)
    if not d.exists():
        pytest.skip(f"Activations directory not found: {d}")
    return d


@pytest.fixture
def episode_dirs(act_dir):
    dirs = sorted(act_dir.glob("episode_*"))
    assert len(dirs) > 0, "No episode directories found"
    return dirs


@pytest.fixture
def first_episode(episode_dirs):
    return episode_dirs[0]


@pytest.fixture
def step_dirs(first_episode):
    dirs = sorted(first_episode.glob("step_*"))
    assert len(dirs) > 0, "No step directories found"
    return dirs


@pytest.fixture
def first_step(step_dirs):
    return step_dirs[0]


# --- Structure ---


class TestDirectoryStructure:
    def test_episode_dirs_exist(self, episode_dirs):
        assert len(episode_dirs) >= 1

    def test_episode_metadata_exists(self, first_episode):
        assert (first_episode / "metadata.json").exists()

    def test_episode_rewards_exists(self, first_episode):
        assert (first_episode / "rewards.npz").exists()

    def test_step_dirs_exist(self, step_dirs):
        assert len(step_dirs) >= 1

    def test_step_has_required_files(self, first_step):
        # tokens / token_logprobs / metadata are always present; hidden_states
        # is conditional on num_tokens > 1 (the EOS-trigger iteration has no
        # pre_logits to write — see save_step_activations_fast).
        for fname in ["tokens.npz", "token_logprobs.npz", "metadata.json"]:
            assert (first_step / fname).exists(), f"Missing {fname} in {first_step}"

    def test_hidden_states_present_when_num_tokens_gt_1(self, first_step):
        with open(first_step / "metadata.json") as f:
            num_tokens = int(json.load(f)["num_tokens"])
        hs_path = first_step / "hidden_states.npz"
        if num_tokens > 1:
            assert hs_path.exists(), f"hidden_states.npz missing for num_tokens={num_tokens}"
        # num_tokens == 1 ⇒ 0 hidden states ⇒ file omitted. Both orderings
        # are legal here (the writer may or may not have cleaned up on retry);
        # just assert the file, if present, is load-able and empty.
        elif hs_path.exists():
            data = np.load(hs_path)
            assert data["token_pre_logits"].shape[0] == 0


# --- Episode-level metadata ---


class TestEpisodeMetadata:
    def test_required_fields(self, first_episode):
        with open(first_episode / "metadata.json") as f:
            meta = json.load(f)
        required = [
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
        ]
        for field in required:
            assert field in meta, f"Missing episode field: {field}"

    def test_config_name_points_to_pi0_fast(self, first_episode):
        """fast_v1 is produced by pi0-fast configs. If a diffusion config name
        sneaked in, something about the writer-dispatch path is wrong."""
        with open(first_episode / "metadata.json") as f:
            meta = json.load(f)
        assert "pi0_fast" in meta["config_name"], (
            f"fast_v1 dataset should be produced by a pi0-fast config; got {meta['config_name']!r}"
        )

    def test_success_implies_steps_to_success(self, episode_dirs):
        for ep_dir in episode_dirs:
            with open(ep_dir / "metadata.json") as f:
                meta = json.load(f)
            if meta["episode_success"]:
                assert meta["steps_to_success"] >= 0
            else:
                assert meta["steps_to_success"] == -1

    def test_rewards_npz_length_matches(self, episode_dirs):
        for ep_dir in episode_dirs:
            with open(ep_dir / "metadata.json") as f:
                meta = json.load(f)
            data = np.load(ep_dir / "rewards.npz")
            assert len(data["per_step_reward"]) == meta["total_env_steps"]
            assert len(data["cumulative_reward"]) == meta["total_env_steps"]
            assert len(data["success_at_step"]) == meta["total_env_steps"]

    def test_rewards_cumulative_matches_total(self, episode_dirs):
        for ep_dir in episode_dirs:
            with open(ep_dir / "metadata.json") as f:
                meta = json.load(f)
            data = np.load(ep_dir / "rewards.npz")
            if len(data["cumulative_reward"]) == 0:
                assert meta["total_reward"] == 0
                continue
            np.testing.assert_allclose(data["cumulative_reward"][-1], meta["total_reward"], rtol=1e-4)


# --- Step-level metadata ---


class TestStepMetadata:
    def test_required_fields(self, first_step):
        with open(first_step / "metadata.json") as f:
            meta = json.load(f)
        required = [
            "task_name",
            "episode_id",
            "env_id",
            "step",
            "inference_step",
            "prompt",
            "cumulative_reward",
            "success_so_far",
            "reward_since_last_inference",
            # fast_v1 extras added by save_step_activations_fast:
            "num_tokens",
            "collection_version",
        ]
        for field in required:
            assert field in meta, f"Missing step field: {field}"

    def test_collection_version_is_fast_v1(self, step_dirs):
        """The writer unconditionally stamps every step metadata with
        collection_version="fast_v1". If any step has a different value, that's
        a schema-drift bug (e.g., a v1-writer call polluted a fast_v1 dataset)."""
        for step_dir in step_dirs:
            with open(step_dir / "metadata.json") as f:
                meta = json.load(f)
            assert meta["collection_version"] == "fast_v1", (
                f"{step_dir}: collection_version={meta['collection_version']!r}, expected 'fast_v1'"
            )

    def test_num_tokens_positive(self, step_dirs):
        for step_dir in step_dirs:
            with open(step_dir / "metadata.json") as f:
                meta = json.load(f)
            assert int(meta["num_tokens"]) >= 1, f"{step_dir}: num_tokens={meta['num_tokens']}"

    def test_cumulative_reward_non_decreasing(self, step_dirs):
        rewards = []
        for step_dir in step_dirs:
            with open(step_dir / "metadata.json") as f:
                meta = json.load(f)
            rewards.append(meta["cumulative_reward"])
        for i in range(1, len(rewards)):
            assert rewards[i] >= rewards[i - 1] - 1e-6, (
                f"Cumulative reward decreased: step {i - 1}={rewards[i - 1]}, step {i}={rewards[i]}"
            )


# --- Activation shape + dtype + consistency ---


class TestActivationShapes:
    def _num_tokens(self, step_dir):
        with open(step_dir / "metadata.json") as f:
            return int(json.load(f)["num_tokens"])

    def test_tokens_shape_and_dtype(self, first_step):
        n = self._num_tokens(first_step)
        data = np.load(first_step / "tokens.npz")
        assert data["generated_tokens"].shape == (n,), f"tokens shape {data['generated_tokens'].shape}, expected ({n},)"
        assert data["generated_tokens"].dtype == np.int32

    def test_token_logprobs_shape_and_dtype(self, first_step):
        n = self._num_tokens(first_step)
        data = np.load(first_step / "token_logprobs.npz")
        assert data["token_logprobs"].shape == (n,), f"logprobs shape {data['token_logprobs'].shape}, expected ({n},)"
        assert data["token_logprobs"].dtype == np.float32

    def test_hidden_states_shape_and_dtype(self, first_step):
        n = self._num_tokens(first_step)
        hs_path = first_step / "hidden_states.npz"
        if n <= 1:
            pytest.skip(f"num_tokens={n}, hidden_states file is not required")
        data = np.load(hs_path)
        arr = data["token_pre_logits"]
        assert arr.ndim == 2, f"token_pre_logits ndim={arr.ndim}, expected 2"
        # Leading axis is num_tokens - 1 (pre_logits for the EOS-trigger
        # iteration is intentionally dropped at the slicing boundary).
        assert arr.shape[0] == n - 1, f"leading axis {arr.shape[0]}, expected {n - 1}"
        # Width for gemma_2b is 2048. If a future model variant changes
        # this, loosen to `arr.shape[1] > 0` but log the observed width.
        assert arr.shape[1] == 2048, f"width {arr.shape[1]}, expected 2048 (gemma_2b)"
        assert arr.dtype == np.float16

    def test_no_nan_inf(self, first_step):
        for fname in ["tokens.npz", "token_logprobs.npz"]:
            data = np.load(first_step / fname)
            for key in data:
                assert np.all(np.isfinite(data[key])), f"{fname}/{key} has NaN/Inf"
        hs_path = first_step / "hidden_states.npz"
        if hs_path.exists():
            data = np.load(hs_path)
            assert np.all(np.isfinite(data["token_pre_logits"])), "hidden_states has NaN/Inf"


# --- Sanity ---


class TestSanityChecks:
    def test_tokens_nonnegative(self, first_step):
        """FAST tokens come out of paligemma's vocab; IDs are always >= 0.
        Negative entries would mean the writer dropped a dtype somewhere (e.g.
        cast from int32 to int8 wrapped a >127 paligemma token)."""
        data = np.load(first_step / "tokens.npz")
        assert np.all(data["generated_tokens"] >= 0)

    def test_tokens_within_paligemma_vocab(self, first_step):
        """paligemma's vocab is ~257k tokens. If a generated token exceeds
        that, something downstream will blow up in decode."""
        data = np.load(first_step / "tokens.npz")
        assert int(data["generated_tokens"].max()) < 257_152, (
            f"max token id {data['generated_tokens'].max()} >= paligemma vocab size"
        )

    def test_logprobs_nonpositive(self, first_step):
        """log(p) for any probability p in [0, 1] is <= 0. Small positive
        drift from fp32 rounding near log(1) is tolerated."""
        data = np.load(first_step / "token_logprobs.npz")
        assert np.all(data["token_logprobs"] <= 1e-4), (
            f"max logprob {data['token_logprobs'].max()} > 0; expected all <= 0"
        )

    def test_hidden_states_nonzero(self, first_step):
        """All-zero pre_logits would mean the model emitted the EOS-only
        sequence or the writer zeroed the buffer — both would be bugs in
        almost any real rollout."""
        hs_path = first_step / "hidden_states.npz"
        if not hs_path.exists():
            pytest.skip("hidden_states file absent (num_tokens == 1)")
        arr = np.load(hs_path)["token_pre_logits"]
        assert np.any(arr != 0), "token_pre_logits is all zero across the entire sequence"

    def test_cross_episode_tokens_eventually_differ(self, act_dir, request):
        """Different env/episode seeds should *usually* diverge in the rollout.

        This is a behavioral sanity signal, not a schema invariant — so it
        emits a pytest warning rather than failing. Legitimately identical
        trajectories across envs happen (e.g. metaworld's
        ``plate-slide-back-side-v3`` seeds env state such that the visible
        observation doesn't vary across the ``seed + i`` range even though
        the reward function does). That's a property of the env, not a
        dataset corruption, so it shouldn't trip CI.

        If we see *every* step across two envs produce bit-identical tokens,
        that's still a signal worth surfacing — warn so a human can sanity-
        check seeding / batch-slicing / policy determinism upstream.
        """
        eps = sorted(act_dir.glob("episode_*"))
        if len(eps) < 2:
            pytest.skip("Need at least 2 episodes/envs to compare")
        s0 = sorted(eps[0].glob("step_*"))
        s1 = sorted(eps[1].glob("step_*"))
        common = min(len(s0), len(s1))
        if common == 0:
            pytest.skip("No step dirs in one of the episodes")
        for i in range(common):
            t0 = np.load(s0[i] / "tokens.npz")["generated_tokens"]
            t1 = np.load(s1[i] / "tokens.npz")["generated_tokens"]
            n = min(len(t0), len(t1))
            if not np.array_equal(t0[:n], t1[:n]):
                return  # divergence found — healthy
        import warnings

        warnings.warn(
            f"{act_dir.name}: envs {eps[0].name}/{eps[1].name} produced bit-identical tokens "
            f"across all {common} common steps. This can be legitimate (deterministic policy on "
            "an env whose observations don't vary with seed) — sanity-check seeding or batch slicing "
            "if unexpected.",
            stacklevel=2,
        )
