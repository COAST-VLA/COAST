"""End-to-end test for Diffusion Policy on MetaWorld.

Trains ``dp_metaworld`` for a handful of steps on ``brandonyang/metaworld_ml45``,
loads the resulting checkpoint, and rolls out a real MetaWorld env for a few
steps to verify the full train -> checkpoint -> policy -> env path.

Marked ``manual`` because it needs:
  - a GPU (CUDA),
  - the LeRobot dataset cached locally (``brandonyang/metaworld_ml45``),
  - precomputed norm stats at
    ``assets/dp_metaworld/brandonyang/metaworld_ml45/norm_stats.json``
    (generate via ``uv run scripts/compute_norm_stats.py dp_metaworld``),
  - MuJoCo with EGL rendering available.

Run:
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \\
        uv run pytest tests/metaworld/test_dp_e2e.py -v -m manual
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NORM_STATS_PATH = REPO_ROOT / "assets" / "dp_metaworld" / "brandonyang" / "metaworld_ml45" / "norm_stats.json"


def _skip_if_missing_prereqs() -> None:
    if not NORM_STATS_PATH.exists():
        pytest.skip(
            f"Norm stats missing at {NORM_STATS_PATH}. Run `uv run scripts/compute_norm_stats.py dp_metaworld` first."
        )
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.manual
def test_dp_metaworld_train_then_infer(tmp_path):
    """Short dp_metaworld training produces a loadable checkpoint; policy rolls out in MetaWorld."""
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
            "dp_metaworld",
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
            "train_pytorch.py dp_metaworld failed\n"
            f"stdout (tail):\n{train_result.stdout[-4000:]}\n\n"
            f"stderr (tail):\n{train_result.stderr[-4000:]}"
        )

    ckpt_dir = ckpt_base / "dp_metaworld" / exp_name / str(num_train_steps)
    assert (ckpt_dir / "model.safetensors").exists(), f"missing model.safetensors under {ckpt_dir}"
    assert (ckpt_dir / "assets" / "brandonyang" / "metaworld_ml45" / "norm_stats.json").exists()

    # Load policy and step the env in-process.
    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium as gym
    import metaworld  # noqa: F401

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config("dp_metaworld")
    policy = _policy_config.create_trained_policy(train_config, str(ckpt_dir), use_pytorch=True)

    mw_env = gym.make("Meta-World/MT1", env_name="reach-v3", seed=0, width=224, height=224)
    obs, _ = mw_env.reset()

    def _render(cam_id: int) -> np.ndarray:
        renderer = mw_env.unwrapped.mujoco_renderer
        viewer = renderer._get_viewer(render_mode="rgb_array")  # noqa: SLF001
        if len(renderer._viewers.keys()) >= 1:  # noqa: SLF001
            viewer.make_context_current()
        return viewer.render(render_mode="rgb_array", camera_id=cam_id)[::-1].copy()

    try:
        for _ in range(3):
            obs_dict = {
                "observation/image": _render(4),  # corner4
                "observation/wrist_image": _render(6),  # gripperPOV
                "observation/state": obs.astype(np.float32)[:4],
                "prompt": "reach the goal position",
            }
            result = policy.infer(obs_dict)
            actions = result["actions"]
            assert actions.shape == (16, 4), f"expected (16,4), got {actions.shape}"
            assert np.isfinite(actions).all(), "non-finite actions"
            action = np.clip(actions[0], -1.0, 1.0).astype(np.float32)
            obs, _reward, terminated, truncated, _info = mw_env.step(action)
            assert not (terminated and truncated)
    finally:
        mw_env.close()
