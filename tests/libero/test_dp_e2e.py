"""End-to-end test for Diffusion Policy on LIBERO (root-venv-side).

Trains ``dp_libero`` for a handful of steps on ``physical-intelligence/libero``,
loads the resulting checkpoint, and runs a single synthetic-obs inference to
verify the train -> checkpoint -> policy-load -> infer path.

This test lives in the root venv and stops at ``policy.infer`` — the real
LIBERO env rollout requires the Python 3.8 ``examples/libero_env/`` venv and
is covered manually via ``scripts/serve_policy.py --pytorch`` + the client
``examples/libero_env/main.py``.

Marked ``manual`` because it needs:
  - a GPU (CUDA),
  - the LeRobot dataset cached locally (``physical-intelligence/libero``),
  - precomputed norm stats at
    ``assets/dp_libero/physical-intelligence/libero/norm_stats.json``
    (generate via ``uv run scripts/compute_norm_stats.py --config-name dp_libero``).

Run:
    CUDA_VISIBLE_DEVICES=0 uv run pytest tests/libero/test_dp_e2e.py -v -m manual
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NORM_STATS_PATH = REPO_ROOT / "assets" / "dp_libero" / "physical-intelligence" / "libero" / "norm_stats.json"


def _skip_if_missing_prereqs() -> None:
    if not NORM_STATS_PATH.exists():
        pytest.skip(
            f"Norm stats missing at {NORM_STATS_PATH}. "
            "Run `uv run scripts/compute_norm_stats.py --config-name dp_libero` first."
        )
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.manual
def test_dp_libero_train_then_infer(tmp_path):
    """Short dp_libero training produces a loadable checkpoint; policy infers (16, 7) on LIBERO-shaped obs."""
    _skip_if_missing_prereqs()

    exp_name = "pytest_e2e"
    num_train_steps = 20
    ckpt_base = tmp_path / "checkpoints"

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    train_result = subprocess.run(
        [
            "uv",
            "run",
            "scripts/train_pytorch.py",
            "dp_libero",
            f"--exp-name={exp_name}",
            f"--num-train-steps={num_train_steps}",
            f"--save-interval={num_train_steps}",
            "--batch-size=8",
            "--num-workers=2",
            "--no-wandb-enabled",
            "--overwrite",
            "--pytorch-training-precision=float32",
            f"--checkpoint-base-dir={ckpt_base}",
        ],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if train_result.returncode != 0:
        pytest.fail(
            "train_pytorch.py dp_libero failed\n"
            f"stdout (tail):\n{train_result.stdout[-4000:]}\n\n"
            f"stderr (tail):\n{train_result.stderr[-4000:]}"
        )

    ckpt_dir = ckpt_base / "dp_libero" / exp_name / str(num_train_steps)
    assert (ckpt_dir / "model.safetensors").exists(), f"missing model.safetensors under {ckpt_dir}"
    assert (ckpt_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json").exists()

    # Load policy in-process and run a single synthetic inference.
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config("dp_libero")
    policy = _policy_config.create_trained_policy(train_config, str(ckpt_dir), use_pytorch=True)

    # LIBERO obs: agentview + eye-in-hand (256x256x3 uint8), state is 8-dim (pos + axisangle + gripper).
    rng = np.random.default_rng(0)
    obs = {
        "observation/image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "observation/state": rng.standard_normal(8).astype(np.float32),
        "prompt": "pick up the object",
    }
    result = policy.infer(obs)
    actions = result["actions"]
    assert actions.shape == (16, 7), f"expected (16,7), got {actions.shape}"
    assert np.isfinite(actions).all(), "non-finite actions"
