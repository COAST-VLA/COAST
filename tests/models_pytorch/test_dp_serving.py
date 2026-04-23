"""Unit tests for Diffusion Policy inference through openpi's generic Policy pipeline.

These tests verify that ``dp_metaworld`` and ``dp_libero`` configs route through
``openpi.policies.policy_config.create_trained_policy`` without any DP-specific server
code — the generic ``scripts/serve_policy.py --pytorch policy:checkpoint`` path works.

They depend on having a trained checkpoint on disk (norm_stats + model.safetensors).
For ``dp_metaworld`` this is produced by ``tests/metaworld/test_dp_e2e.py`` on first run;
for ``dp_libero`` by ``tests/libero/test_dp_e2e.py``. Both tests are ``manual`` since they
require a GPU. The released robocasa ``.ckpt`` is not eval'd via this repo — use
upstream's ``eval_robocasa.py`` at https://github.com/robocasa-benchmark/diffusion_policy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
METAWORLD_CKPT = REPO_ROOT / "checkpoints/dp_metaworld/e2e_dp_transformer/20"
LIBERO_CKPT = REPO_ROOT / "checkpoints/dp_libero/e2e_dp_transformer/20"


def _skip_without_cuda() -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.manual
def test_dp_metaworld_infer_through_generic_policy_pipeline():
    """dp_metaworld checkpoint loads via create_trained_policy and infers (16, 4) actions."""
    _skip_without_cuda()
    if not (METAWORLD_CKPT / "model.safetensors").exists():
        pytest.skip(f"Missing {METAWORLD_CKPT}; generate with `pytest tests/metaworld/test_dp_e2e.py -m manual`")

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config("dp_metaworld")
    policy = _policy_config.create_trained_policy(train_config, str(METAWORLD_CKPT), use_pytorch=True)

    rng = np.random.default_rng(0)
    obs = {
        "observation/image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal(4).astype(np.float32),
        "prompt": "reach the goal position",
    }
    result = policy.infer(obs)
    actions = result["actions"]
    assert actions.shape == (16, 4), f"expected (16, 4), got {actions.shape}"
    assert np.isfinite(actions).all(), "non-finite actions"


@pytest.mark.manual
def test_dp_libero_infer_through_generic_policy_pipeline():
    """dp_libero checkpoint loads via create_trained_policy and infers (16, 7) actions."""
    _skip_without_cuda()
    if not (LIBERO_CKPT / "model.safetensors").exists():
        pytest.skip(f"Missing {LIBERO_CKPT}; generate with `pytest tests/libero/test_dp_e2e.py -m manual`")

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config("dp_libero")
    policy = _policy_config.create_trained_policy(train_config, str(LIBERO_CKPT), use_pytorch=True)

    rng = np.random.default_rng(0)
    obs = {
        "observation/image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal(8).astype(np.float32),
        "prompt": "pick up the object",
    }
    result = policy.infer(obs)
    actions = result["actions"]
    assert actions.shape == (16, 7), f"expected (16, 7), got {actions.shape}"
    assert np.isfinite(actions).all(), "non-finite actions"
