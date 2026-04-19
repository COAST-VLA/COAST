"""In-process end-to-end smoke test: load DP checkpoint, run MetaWorld + LIBERO envs for a few steps.

Verifies that training → checkpointing → policy loading → environment stepping all wire up.
Because the checkpoints are 200-step sanity runs, we don't expect task success; we just expect
no exceptions, sensible action shapes, and env steps completing.
"""

from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    env: str = "metaworld"  # "metaworld" or "libero"
    config_name: str = "dp_metaworld"
    checkpoint_dir: str = "checkpoints/dp_metaworld/dp_sanity/200/"
    num_steps: int = 10
    task: str = "reach-v3"  # for metaworld
    libero_suite: str = "libero_spatial"
    libero_task_id: int = 0


def eval_metaworld(args: Args) -> None:
    import gymnasium as gym
    import metaworld  # noqa: F401

    cfg = _config.get_config(args.config_name)
    print(f"[metaworld] loading policy from {args.checkpoint_dir}")
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint_dir, use_pytorch=True)
    print("[metaworld] policy loaded")

    env = gym.make("Meta-World/MT1", env_name=args.task, seed=0, width=224, height=224)
    obs, info = env.reset()

    def render_cam(cam_id: int) -> np.ndarray:
        renderer = env.unwrapped.mujoco_renderer
        viewer = renderer._get_viewer(render_mode="rgb_array")  # noqa: SLF001
        if len(renderer._viewers.keys()) >= 1:  # noqa: SLF001
            viewer.make_context_current()
        return viewer.render(render_mode="rgb_array", camera_id=cam_id)[::-1].copy()

    corner_img = render_cam(4)  # corner4
    wrist_img = render_cam(6)  # gripperPOV

    for step in range(args.num_steps):
        obs_dict = {
            "observation/image": corner_img,
            "observation/wrist_image": wrist_img,
            "observation/state": obs.astype(np.float32)[:4],
            "prompt": "reach the goal position",
        }
        result = policy.infer(obs_dict)
        actions = result["actions"]
        assert actions.shape == (16, 4), f"unexpected action shape {actions.shape}"
        action = np.clip(actions[0], -1.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        corner_img = render_cam(4)
        wrist_img = render_cam(6)
        print(
            f"[metaworld] step={step} action={action.round(3)} reward={reward:.3f} infer_ms={result['policy_timing']['infer_ms']:.1f}"
        )
    env.close()
    print("[metaworld] OK")


def eval_libero(args: Args) -> None:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = _config.get_config(args.config_name)
    print(f"[libero] loading policy from {args.checkpoint_dir}")
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint_dir, use_pytorch=True)
    print("[libero] policy loaded")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_suite]()
    task = task_suite.get_task(args.libero_task_id)
    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    obs = env.reset()

    # Settle
    dummy = np.zeros(7)
    for _ in range(5):
        obs, _, _, _ = env.step(dummy)

    for step in range(args.num_steps):
        obs_dict = {
            "observation/image": np.ascontiguousarray(obs["agentview_image"][::-1]),
            "observation/wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
            "observation/state": np.concatenate(
                [obs["robot0_eef_pos"], _quat_to_axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"][:1]]
            ).astype(np.float32),
            "prompt": task.language,
        }
        result = policy.infer(obs_dict)
        actions = result["actions"]
        assert actions.shape == (16, 7), f"unexpected action shape {actions.shape}"
        action = np.clip(actions[0], -1.0, 1.0).astype(np.float32)
        obs, reward, done, info = env.step(action)
        print(
            f"[libero] step={step} action={action.round(3)} reward={reward:.3f} infer_ms={result['policy_timing']['infer_ms']:.1f}"
        )
    env.close()
    print("[libero] OK")


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Minimal quat (xyzw) -> axis-angle conversion, same as LIBERO example main.py expects."""
    # Adapted from robosuite utils; sufficient for a smoke test state.
    x, y, z, w = quat
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 1e-8))
    return np.array([x / s, y / s, z / s]) * angle


def main(args: Args) -> None:
    if args.env == "metaworld":
        eval_metaworld(args)
    elif args.env == "libero":
        eval_libero(args)
    else:
        raise ValueError(f"Unknown env: {args.env}")


if __name__ == "__main__":
    main(tyro.cli(Args))
