"""Unit tests for ``openpi.policies.policy.collate_transformed_singles``.

The helper batches a list of per-example transform outputs back into a dict
of leading-batch-dim arrays. Two callers share it: ``Policy.infer_batched``
(pi0/pi0.5/pi0-fast) and ``Policy.infer_with_intermediates`` (pi0.5 PyTorch
and pi0-fast JAX).

pi0-fast's model transform adds ``token_ar_mask`` / ``token_loss_mask``
keys that pi0/pi0.5 don't produce. The helper must stack those extras
when present and skip them when absent, without either path needing to
know which model it's running.
"""

from __future__ import annotations

import numpy as np

from openpi.policies.policy import collate_transformed_singles


def _diffusion_single(i: int) -> dict:
    """Shape of what DataConfig.model_transforms produces for pi0/pi0.5."""
    return {
        "state": np.full(8, i, dtype=np.float32),
        "tokenized_prompt": np.full(48, i, dtype=np.int32),
        "tokenized_prompt_mask": np.ones(48, dtype=bool),
        "image": {
            "base_0_rgb": np.full((224, 224, 3), i, dtype=np.uint8),
            "left_wrist_0_rgb": np.full((224, 224, 3), i, dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": np.ones((), dtype=bool),
            "left_wrist_0_rgb": np.ones((), dtype=bool),
        },
    }


def _fast_single(i: int) -> dict:
    """pi0-fast adds token_ar_mask + token_loss_mask to the per-example dict."""
    base = _diffusion_single(i)
    base["token_ar_mask"] = np.full(48, i % 2, dtype=bool)
    base["token_loss_mask"] = np.full(48, (i + 1) % 2, dtype=bool)
    return base


def test_collate_diffusion_case_stacks_core_fields():
    singles = [_diffusion_single(0), _diffusion_single(1), _diffusion_single(2)]
    out = collate_transformed_singles(singles)

    assert out["state"].shape == (3, 8)
    assert out["tokenized_prompt"].shape == (3, 48)
    assert out["tokenized_prompt_mask"].shape == (3, 48)
    assert out["image"]["base_0_rgb"].shape == (3, 224, 224, 3)
    assert out["image_mask"]["base_0_rgb"].shape == (3,)

    # No pi0-fast keys introduced when the inputs don't have them.
    assert "token_ar_mask" not in out
    assert "token_loss_mask" not in out

    # Stacking preserves per-example values.
    np.testing.assert_array_equal(out["state"][0], np.zeros(8, dtype=np.float32))
    np.testing.assert_array_equal(out["state"][2], np.full(8, 2, dtype=np.float32))


def test_collate_pi0_fast_case_stacks_token_masks():
    singles = [_fast_single(0), _fast_single(1)]
    out = collate_transformed_singles(singles)

    # The two extras must be stacked with the same leading batch dim as state.
    assert out["token_ar_mask"].shape == (2, 48)
    assert out["token_loss_mask"].shape == (2, 48)
    assert out["token_ar_mask"].dtype == bool
    assert out["token_loss_mask"].dtype == bool

    # Per-example values preserved — first row is i=0, second is i=1.
    np.testing.assert_array_equal(out["token_ar_mask"][0], np.zeros(48, dtype=bool))
    np.testing.assert_array_equal(out["token_ar_mask"][1], np.ones(48, dtype=bool))
    np.testing.assert_array_equal(out["token_loss_mask"][0], np.ones(48, dtype=bool))
    np.testing.assert_array_equal(out["token_loss_mask"][1], np.zeros(48, dtype=bool))


def test_collate_single_example_still_produces_batch_dim():
    out = collate_transformed_singles([_fast_single(7)])
    assert out["state"].shape == (1, 8)
    assert out["token_ar_mask"].shape == (1, 48)
    assert out["image"]["base_0_rgb"].shape == (1, 224, 224, 3)
