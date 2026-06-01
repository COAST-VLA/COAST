"""Unit tests for src/openpi/serving/conceptors.py.

Pure numpy — no torch, no GPU, no real checkpoints. Covers:
- correlation_matrix basic math properties
- conceptor identity/limit behavior
- boolean_and / boolean_not algebraic identities
- contrastive_conceptor combination
- flatten_{global,per_step} output shapes
- compute_task_conceptors key layout
- Directory walk / success-label classification
"""

# ruff: noqa: N802, N806, RUF001, RUF002, RUF003
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from openpi.serving.conceptors import DEFAULT_COLLECT_LAYERS
from openpi.serving.conceptors import boolean_and
from openpi.serving.conceptors import boolean_not
from openpi.serving.conceptors import compute_all_conceptors
from openpi.serving.conceptors import compute_linear_direction
from openpi.serving.conceptors import compute_task_conceptors
from openpi.serving.conceptors import conceptor
from openpi.serving.conceptors import contrastive_conceptor
from openpi.serving.conceptors import correlation_matrix
from openpi.serving.conceptors import episode_is_success
from openpi.serving.conceptors import flatten_global
from openpi.serving.conceptors import flatten_per_step
from openpi.serving.conceptors import iter_episode_dirs
from openpi.serving.conceptors import load_episode_hiddens
from openpi.serving.conceptors import random_matched_conceptor

# ═══════════════════════════════════════════════════════════════════════════════
# correlation_matrix
# ═══════════════════════════════════════════════════════════════════════════════


def test_correlation_symmetric():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 16))
    R = correlation_matrix(X)
    np.testing.assert_allclose(R, R.T, atol=1e-12)


def test_correlation_identity_for_orthonormal_rows():
    """If rows of X are orthonormal scaled so X^T X / N = I, we expect identity."""
    d = 8
    N = 8
    X = np.eye(d) * np.sqrt(N)  # X^T X = N*I, divided by N → I
    R = correlation_matrix(X)
    np.testing.assert_allclose(R, np.eye(d), atol=1e-12)


def test_correlation_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        correlation_matrix(np.zeros((3, 4, 5)))


def test_correlation_rejects_empty():
    with pytest.raises(ValueError, match="N=0"):
        correlation_matrix(np.zeros((0, 4)))


# ═══════════════════════════════════════════════════════════════════════════════
# conceptor
# ═══════════════════════════════════════════════════════════════════════════════


def test_conceptor_identity_alpha1_gives_half():
    """R=I, α=1 → C = I @ inv(I + I) = 0.5 I"""
    d = 6
    C = conceptor(np.eye(d), alpha=1.0)
    np.testing.assert_allclose(C, 0.5 * np.eye(d), atol=1e-12)


def test_conceptor_large_alpha_approaches_identity():
    """α → ∞ makes C(R, α) → I for any nonzero R."""
    rng = np.random.default_rng(1)
    d = 12
    X = rng.standard_normal((500, d))
    R = correlation_matrix(X)
    C = conceptor(R, alpha=1e6)
    # Allow generous tolerance — not a tight limit
    assert np.allclose(C, np.eye(d), atol=1e-4)


def test_conceptor_small_alpha_approaches_zero():
    rng = np.random.default_rng(2)
    d = 12
    X = rng.standard_normal((500, d))
    R = correlation_matrix(X)
    C = conceptor(R, alpha=1e-6)
    assert np.allclose(C, np.zeros((d, d)), atol=1e-4)


def test_conceptor_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        conceptor(np.zeros((3, 4)), alpha=1.0)


def test_conceptor_rejects_non_positive_alpha():
    with pytest.raises(ValueError, match="positive"):
        conceptor(np.eye(4), alpha=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# boolean_and / boolean_not / contrastive
# ═══════════════════════════════════════════════════════════════════════════════


def test_boolean_and_with_identity_returns_other_operand():
    """AND(C, I) = C @ inv(C + I - C) @ I = C @ inv(I) = C. Cleanest algebraic identity."""
    rng = np.random.default_rng(3)
    d = 8
    X = rng.standard_normal((100, d))
    C = conceptor(correlation_matrix(X), alpha=1.0)
    np.testing.assert_allclose(boolean_and(C, np.eye(d)), C, atol=1e-8)
    np.testing.assert_allclose(boolean_and(np.eye(d), C), C, atol=1e-8)


def test_boolean_and_idempotent_on_projection():
    """For a projection P (eigenvalues 0 or 1), AND(P, P) = P.

    Conceptors with α=∞ degenerate to projections onto the column space of R.
    Non-projection conceptors do NOT satisfy idempotence — that's an easy math
    trap (AND(C,C) has eigenvalue λ/(2-λ), not λ, for λ∈(0,1)).
    """
    d = 8
    # Explicit projection onto first 4 coords
    P = np.zeros((d, d))
    P[:4, :4] = np.eye(4)
    np.testing.assert_allclose(boolean_and(P, P), P, atol=1e-8)


def test_boolean_and_zero_absorbs():
    """AND(0, anything) = 0 — zero conceptor represents the empty subspace."""
    d = 6
    # solve(0 + C - 0@C) = solve(C, C) = I, so C1 @ I @ C2 = 0 @ C2 = 0
    # But with zero conceptor the matrix C1+C2-C1@C2 = C2 is potentially singular.
    # Add a tiny ridge to C2 to avoid failure; check symbolically.
    Z = np.zeros((d, d))
    rng = np.random.default_rng(4)
    X = rng.standard_normal((100, d))
    C = conceptor(correlation_matrix(X), alpha=1.0)
    out = boolean_and(Z, C)
    np.testing.assert_allclose(out, np.zeros((d, d)), atol=1e-8)


def test_boolean_not_flips_identity_and_zero():
    d = 5
    np.testing.assert_allclose(boolean_not(np.eye(d)), np.zeros((d, d)), atol=1e-12)
    np.testing.assert_allclose(boolean_not(np.zeros((d, d))), np.eye(d), atol=1e-12)


def test_boolean_not_involutive():
    rng = np.random.default_rng(5)
    d = 8
    X = rng.standard_normal((100, d))
    C = conceptor(correlation_matrix(X), alpha=1.0)
    np.testing.assert_allclose(boolean_not(boolean_not(C)), C, atol=1e-12)


def test_contrastive_conceptor_shape_and_finite():
    rng = np.random.default_rng(6)
    d = 8
    X_s = rng.standard_normal((150, d))
    X_f = rng.standard_normal((120, d))
    C_s = conceptor(correlation_matrix(X_s), alpha=0.5)
    C_f = conceptor(correlation_matrix(X_f), alpha=0.5)
    C_c = contrastive_conceptor(C_s, C_f)
    assert C_c.shape == (d, d)
    assert np.all(np.isfinite(C_c))


def test_contrastive_on_projection_is_zero():
    """For a projection P, AND(P, NOT(P)) = 0 exactly.

    Non-projection conceptors give nonzero AND(C, NOT(C)) (eigenvalue
    λ(1-λ)/(1-λ+λ²) ≠ 0 at interior λ). We test the projection case where
    the identity is clean.
    """
    d = 8
    P = np.zeros((d, d))
    P[:4, :4] = np.eye(4)
    out = contrastive_conceptor(P, P)
    np.testing.assert_allclose(out, np.zeros((d, d)), atol=1e-8)


def test_contrastive_disjoint_subspaces():
    """If success and failure span disjoint subspaces (no overlap), C_contrastive = C_success.

    This is the intuitive case: a conceptor pointing only at success directions,
    zero failure energy there → the AND keeps it intact.
    """
    d = 8
    # P_s projects onto first 4, P_f projects onto last 4 — disjoint
    P_s = np.zeros((d, d))
    P_s[:4, :4] = np.eye(4)
    P_f = np.zeros((d, d))
    P_f[4:, 4:] = np.eye(4)
    out = contrastive_conceptor(P_s, P_f)
    np.testing.assert_allclose(out, P_s, atol=1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# Linear direction (for the `linear` steering strategy)
# ═══════════════════════════════════════════════════════════════════════════════


def test_linear_direction_is_unit():
    rng = np.random.default_rng(20)
    X_s = rng.standard_normal((100, 16)) + 3.0  # shifted positive
    X_f = rng.standard_normal((100, 16))
    v = compute_linear_direction(X_s, X_f)
    assert v.shape == (16,)
    assert v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_linear_direction_points_from_failure_to_success():
    """If success cluster is shifted in the +e0 direction, v should align with e0."""
    d = 8
    rng = np.random.default_rng(21)
    X_s = rng.standard_normal((200, d)) * 0.1 + np.eye(d)[0] * 5.0  # shifted on e0
    X_f = rng.standard_normal((200, d)) * 0.1
    v = compute_linear_direction(X_s, X_f)
    # v[0] should be ≈ 1, others ≈ 0
    assert v[0] > 0.95
    assert np.max(np.abs(v[1:])) < 0.1


def test_linear_direction_zero_when_means_coincide():
    """Identical distributions → zero vector (no NaN)."""
    X = np.random.default_rng(22).standard_normal((50, 8))
    v = compute_linear_direction(X, X)  # same data both classes
    assert v.shape == (8,)
    np.testing.assert_allclose(v, np.zeros(8), atol=1e-6)


def test_linear_direction_rejects_mismatched_dim():
    with pytest.raises(ValueError, match="hidden dim mismatch"):
        compute_linear_direction(np.zeros((10, 8)), np.zeros((10, 16)))


def test_linear_direction_rejects_empty():
    with pytest.raises(ValueError, match="empty success or failure"):
        compute_linear_direction(np.zeros((0, 8)), np.zeros((10, 8)))


# ═══════════════════════════════════════════════════════════════════════════════
# Random-matched conceptor (for the `random_matched` control strategy)
# ═══════════════════════════════════════════════════════════════════════════════


def test_random_matched_preserves_spectrum():
    """Eigenvalues of C_random match those of C_reference (within numerical tolerance)."""
    rng = np.random.default_rng(30)
    d = 12
    X = rng.standard_normal((500, d))
    C_ref = conceptor(correlation_matrix(X), alpha=0.5)
    C_rand = random_matched_conceptor(C_ref, seed=7)
    eig_ref = np.sort(np.linalg.eigvalsh(0.5 * (C_ref + C_ref.T)))
    eig_rand = np.sort(np.linalg.eigvalsh(0.5 * (C_rand + C_rand.T)))
    np.testing.assert_allclose(eig_rand, eig_ref, atol=1e-4, rtol=1e-4)


def test_random_matched_is_deterministic_for_same_seed():
    d = 8
    C_ref = np.eye(d) * 0.5
    a = random_matched_conceptor(C_ref, seed=42)
    b = random_matched_conceptor(C_ref, seed=42)
    np.testing.assert_array_equal(a, b)


def test_random_matched_differs_for_different_seed():
    d = 16
    rng = np.random.default_rng(31)
    C_ref = conceptor(correlation_matrix(rng.standard_normal((200, d))), alpha=1.0)
    a = random_matched_conceptor(C_ref, seed=1)
    b = random_matched_conceptor(C_ref, seed=2)
    # Different random basis → matrices should differ substantially
    assert np.linalg.norm(a - b) > 0.1


def test_random_matched_output_is_symmetric_and_float32():
    d = 10
    rng = np.random.default_rng(32)
    C_ref = conceptor(correlation_matrix(rng.standard_normal((300, d))), alpha=1.0)
    C_rand = random_matched_conceptor(C_ref, seed=5)
    assert C_rand.shape == (d, d)
    assert C_rand.dtype == np.float32
    np.testing.assert_allclose(C_rand, C_rand.T, atol=1e-5)


def test_random_matched_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        random_matched_conceptor(np.zeros((3, 4)), seed=0)


def test_random_matched_rotational_invariant_eigenvalues():
    """Using a scaled identity C_ref, the random result must also have constant eigenvalues."""
    d = 6
    C_ref = np.eye(d) * 0.3
    C_rand = random_matched_conceptor(C_ref, seed=99)
    eig = np.sort(np.linalg.eigvalsh(0.5 * (C_rand + C_rand.T)))
    np.testing.assert_allclose(eig, np.full(d, 0.3), atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# Flatten helpers
# ═══════════════════════════════════════════════════════════════════════════════


def test_flatten_global_concats_all_axes_except_hidden():
    T, num_denoise, num_layers, num_tokens, d = 3, 10, 4, 32, 16
    h1 = np.random.default_rng(10).standard_normal((T, num_denoise, num_layers, num_tokens, d))
    h2 = np.random.default_rng(11).standard_normal((T + 1, num_denoise, num_layers, num_tokens, d))
    X = flatten_global([h1, h2], layer_axis=2)
    expected_n = (T + (T + 1)) * num_denoise * num_tokens
    assert X.shape == (expected_n, d)


def test_flatten_per_step_picks_single_denoise_step():
    T, num_denoise, num_layers, num_tokens, d = 2, 10, 4, 32, 16
    h = np.random.default_rng(12).standard_normal((T, num_denoise, num_layers, num_tokens, d))
    X = flatten_per_step([h], layer_axis=1, denoise_step=3)
    assert X.shape == (T * num_tokens, d)
    # Verify the selected data matches h[:, 3, 1, :, :]
    expected = h[:, 3, 1, :, :].reshape(-1, d)
    np.testing.assert_allclose(X, expected.astype(np.float64), atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_task_conceptors
# ═══════════════════════════════════════════════════════════════════════════════


def _fake_episode_hiddens(seed: int, num_rollout_steps: int = 2, d: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # shape: (T, 10, 4, 32, d) — 4 collected layers
    return rng.standard_normal((num_rollout_steps, 10, 4, 32, d)).astype(np.float32)


def test_compute_task_conceptors_key_layout():
    success = [_fake_episode_hiddens(s) for s in range(3)]
    failure = [_fake_episode_hiddens(s + 100) for s in range(3)]
    out = compute_task_conceptors(
        success,
        failure,
        layers=(0, 11),
        alphas=(0.1, 1.0),
        per_step_indices=(0, 9),
        collect_layers=DEFAULT_COLLECT_LAYERS,
    )

    # Per layer: 2 alphas × 3 kinds + 2 per_step × 3 kinds + 1 linear = 13 keys. × 2 layers = 26
    assert len(out) == 26

    for layer in (0, 11):
        for alpha in (0.1, 1.0):
            for kind in ("C_success", "C_failure", "C_contrastive"):
                assert f"L{layer}__{alpha}__{kind}" in out
        for t in (0, 9):
            for kind in ("C_success", "C_failure", "C_contrastive"):
                assert f"L{layer}__per_step_{t}__{kind}" in out
        assert f"L{layer}__linear_direction" in out


def test_compute_task_conceptors_matrix_shape_and_dtype():
    d = 16
    success = [_fake_episode_hiddens(s, d=d) for s in range(2)]
    failure = [_fake_episode_hiddens(s + 50, d=d) for s in range(2)]
    out = compute_task_conceptors(
        success,
        failure,
        layers=(11,),
        alphas=(1.0,),
        per_step_indices=(0,),
        collect_layers=DEFAULT_COLLECT_LAYERS,
    )
    for k, M in out.items():
        if k.endswith("linear_direction"):
            assert M.shape == (d,), f"{k}: shape {M.shape}"
        else:
            assert M.shape == (d, d), f"{k}: shape {M.shape}"
        assert M.dtype == np.float32
        assert np.all(np.isfinite(M))


def test_layer_not_in_collect_layers_raises():
    success = [_fake_episode_hiddens(0)]
    failure = [_fake_episode_hiddens(1)]
    with pytest.raises(ValueError, match="not in collect_layers"):
        compute_task_conceptors(
            success,
            failure,
            layers=(3,),  # not in DEFAULT_COLLECT_LAYERS = (0,5,11,17)
            alphas=(1.0,),
            per_step_indices=(),
            collect_layers=DEFAULT_COLLECT_LAYERS,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Directory walk
# ═══════════════════════════════════════════════════════════════════════════════


def _make_fake_tree(tmp_path: pathlib.Path, d: int = 16) -> pathlib.Path:
    """Build a minimal activation tree with 2 tasks × 4 episodes × 2 rollout steps."""
    root = tmp_path / "acts"
    ckpt = root / "step_1000"
    rng = np.random.default_rng(42)

    for task in ("taskA", "taskB"):
        for ep_idx in range(4):
            ep_dir = ckpt / task / f"episode_{ep_idx:03d}_env_000"
            ep_dir.mkdir(parents=True)
            # Mark ep 0,1 success, ep 2,3 failure for each task
            is_success = ep_idx < 2
            success_at_step = np.array([False, is_success], dtype=bool)
            rewards = np.zeros(2, dtype=np.float32)
            np.savez(
                ep_dir / "rewards.npz",
                per_step_reward=rewards,
                cumulative_reward=rewards,
                success_at_step=success_at_step,
            )
            (ep_dir / "metadata.json").write_text(json.dumps({"task_name": task}))

            # 2 rollout steps per episode
            for s in range(2):
                step_dir = ep_dir / f"step_{s:04d}"
                step_dir.mkdir()
                # shape (10, 4, 32, d)
                resid = rng.standard_normal((10, 4, 32, d)).astype(np.float32)
                np.savez(step_dir / "suffix_residual.npz", all_suffix_residual=resid)
                (step_dir / "metadata.json").write_text(json.dumps({"step": s}))

    return root


def test_iter_episode_dirs_finds_all(tmp_path):
    root = _make_fake_tree(tmp_path)
    eps = list(iter_episode_dirs(root))
    assert len(eps) == 8  # 2 tasks × 4 episodes
    tasks = {e.parent.name for e in eps}
    assert tasks == {"taskA", "taskB"}


def test_episode_is_success_reads_success_at_step(tmp_path):
    root = _make_fake_tree(tmp_path)
    for ep in iter_episode_dirs(root):
        idx = int(ep.name.split("_")[1])  # "episode_002_env_000" → 2
        expected = idx < 2
        assert episode_is_success(ep) == expected


def test_load_episode_hiddens_stacks_step_dirs(tmp_path):
    root = _make_fake_tree(tmp_path)
    ep = next(iter_episode_dirs(root))
    H = load_episode_hiddens(ep)
    assert H.shape == (2, 10, 4, 32, 16)  # (T=2, denoise=10, layers=4, tokens=32, d=16)


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end pipeline on synthetic tree
# ═══════════════════════════════════════════════════════════════════════════════


def test_compute_all_conceptors_end_to_end(tmp_path):
    root = _make_fake_tree(tmp_path)
    out_path = tmp_path / "output.npz"
    summary = compute_all_conceptors(
        root,
        out_path,
        layers=(11,),
        alphas=(0.1, 1.0),
        per_step_indices=(0, 9),
    )
    assert summary["num_tasks"] == 2
    # 2 tasks × (1 layer × (2 alphas × 3 kinds + 2 per_step × 3 kinds + 1 linear)) = 2 × 13 = 26
    assert summary["num_keys"] == 26
    assert summary["skipped_tasks"] == []
    assert set(summary["included_tasks"]) == {"taskA", "taskB"}

    # Load and verify keys
    npz = np.load(out_path)
    for task in ("taskA", "taskB"):
        assert f"{task}__L11__0.1__C_contrastive" in npz.files
        assert f"{task}__L11__1.0__C_success" in npz.files
        assert f"{task}__L11__per_step_0__C_failure" in npz.files
        assert f"{task}__L11__per_step_9__C_contrastive" in npz.files
        assert f"{task}__L11__linear_direction" in npz.files
        v = npz[f"{task}__L11__linear_direction"]
        assert v.shape == (16,)  # d=16 in the fake tree
        assert v.dtype == np.float32

    # Matrix shape should be (d, d) where d=16 in the fake tree
    M = npz["taskA__L11__1.0__C_contrastive"]
    assert M.shape == (16, 16)
    assert M.dtype == np.float32

    # Sidecar metadata exists
    meta = json.loads((out_path.with_suffix(".meta.json")).read_text())
    assert meta["num_tasks_included"] == 2
    assert meta["layers"] == [11]


def test_compute_all_conceptors_skips_tasks_without_both_classes(tmp_path):
    """A task with only successes (or only failures) is skipped with a warning."""
    root = tmp_path / "acts"
    ckpt = root / "step_1000"
    rng = np.random.default_rng(99)

    # taskA: 4 successes, 0 failures → should be skipped
    # taskB: 2 successes, 2 failures → included
    for task, success_flags in [("taskA", [True] * 4), ("taskB", [True, True, False, False])]:
        for ep_idx, is_success in enumerate(success_flags):
            ep_dir = ckpt / task / f"episode_{ep_idx:03d}_env_000"
            ep_dir.mkdir(parents=True)
            np.savez(
                ep_dir / "rewards.npz",
                per_step_reward=np.zeros(2, dtype=np.float32),
                cumulative_reward=np.zeros(2, dtype=np.float32),
                success_at_step=np.array([False, is_success], dtype=bool),
            )
            step_dir = ep_dir / "step_0000"
            step_dir.mkdir()
            np.savez(
                step_dir / "suffix_residual.npz",
                all_suffix_residual=rng.standard_normal((10, 4, 32, 16)).astype(np.float32),
            )

    summary = compute_all_conceptors(
        root,
        tmp_path / "out.npz",
        layers=(11,),
        alphas=(1.0,),
        per_step_indices=(),
        min_episodes_per_class=2,
    )
    assert summary["num_tasks"] == 1
    assert summary["skipped_tasks"] == ["taskA"]
    assert summary["included_tasks"] == ["taskB"]


def test_compute_all_conceptors_output_is_steering_compatible(tmp_path):
    """Produced NPZ must load cleanly via steering.load_conceptor_npz / get_conceptor_matrix."""
    from openpi.serving.steering import available_tasks
    from openpi.serving.steering import get_conceptor_matrix
    from openpi.serving.steering import load_conceptor_npz

    root = _make_fake_tree(tmp_path)
    out_path = tmp_path / "out.npz"
    compute_all_conceptors(
        root,
        out_path,
        layers=(11,),
        alphas=(0.1,),
        per_step_indices=(0,),
    )

    npz = load_conceptor_npz(out_path)
    assert available_tasks(npz) == {"taskA", "taskB"}
    C = get_conceptor_matrix(npz, "taskA", 11, 0.1, "global")
    assert C.shape == (16, 16)
    # per_step keys exist in the NPZ but are not looked up via get_conceptor_matrix —
    # the per_step strategy returns a LIST via get_per_step_conceptor_matrices. Just
    # assert the NPZ key is present.
    assert "taskB__L11__per_step_0__C_contrastive" in npz.files


def test_compute_all_conceptors_supports_all_five_strategies(tmp_path):
    """Full-stack integration: synthetic activations → NPZ → dispatch all 5 strategies via wrapper."""
    from openpi.serving.steering import ConceptorSteeringHook
    from openpi.serving.steering import LinearSteeringHook
    from openpi.serving.steering import SteeredPolicyWrapper

    root = _make_fake_tree(tmp_path)
    out_path = tmp_path / "out.npz"
    compute_all_conceptors(
        root,
        out_path,
        layers=(11,),
        alphas=(0.1,),
        per_step_indices=tuple(range(10)),
    )

    # Stub policy that records calls; wrapper dispatches hooks through us.
    class _Stub:
        def __init__(self):
            self.last_steering_hooks = None
            self._metadata = {}

        def infer(self, obs):
            return {"actions": np.zeros((1, 4))}

        def infer_with_steering(self, obs, *, steering_hooks):
            self.last_steering_hooks = steering_hooks
            return {"actions": np.ones((1, 4))}, {}

        @property
        def metadata(self):
            return self._metadata

    w = SteeredPolicyWrapper(_Stub(), conceptor_npz_path=out_path, device="cpu")
    expected_hook_types = {
        "global": ConceptorSteeringHook,
        "per_step": ConceptorSteeringHook,
        "positive_only": ConceptorSteeringHook,
        "random_matched": ConceptorSteeringHook,
        "linear": LinearSteeringHook,
    }
    for strategy, expected_cls in expected_hook_types.items():
        payload = {
            "task": "taskA",
            "layer": 11,
            "alpha": 0.1,
            "beta": 0.3,
            "strategy": strategy,
        }
        w.infer({"__steering__": payload})
        layer_idx, hook = w._policy.last_steering_hooks[0]  # noqa: SLF001
        assert layer_idx == 11, f"{strategy}: wrong layer"
        assert isinstance(hook, expected_cls), f"{strategy}: got {type(hook).__name__}"

    # Cache should hold exactly 5 entries (one per strategy).
    assert len(w._hook_cache) == 5  # noqa: SLF001
