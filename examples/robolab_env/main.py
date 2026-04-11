"""
Evaluate a single RoboLab task using a policy server.

For evaluating one environment:
    OMNI_KIT_ACCEPT_EULA=YES uv run examples/robolab_env/main.py --headless --task-name BananaInBowlTask

RoboLab tasks ship with a Franka arm and the DROID joint-position action space
(7 joint deltas + 1 binary gripper). The WebSocket protocol used here matches
RoboLab's own ``Pi0DroidJointposClient`` — the server must be serving a
DROID-trained pi0/pi05 checkpoint (e.g. ``pi05_droid``), not a metaworld/libero one.

RoboLab's env is natively vectorized (``num_envs`` parallel episodes inside one
Isaac Sim process). Unlike RoboCasa / LIBERO which loop one episode at a time,
here we drive ``num_envs`` episodes in parallel per "run" and then
``env.reset_eval_state()`` between runs. Keep ``num_envs=1`` to match the
robocasa/libero main.py flow most closely.
"""

# isort: skip_file
# Must launch Isaac Sim BEFORE any robolab / torch-CUDA import. This mirrors
# the boot sequence used by robolab's own examples/policy/run_eval.py.

import argparse
import traceback

import cv2  # noqa: F401  # must be imported before isaaclab — do not remove

from isaaclab.app import AppLauncher

_launcher_parser = argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(_launcher_parser)
_launcher_args, _remaining_argv = _launcher_parser.parse_known_args()
_launcher_args.enable_cameras = True
_app_launcher = AppLauncher(_launcher_args)
simulation_app = _app_launcher.app

# Hand the remaining CLI tokens off to tyro so it only sees the app-specific
# args below.
import sys

sys.argv = [sys.argv[0]] + _remaining_argv

import collections
import dataclasses
import logging
import math
import os
from typing import Optional

import numpy as np
import torch
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.collection_session import CollectionSession
from robolab.core.utils.video_utils import VideoWriter
from tqdm import tqdm

from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env
from robolab.registrations.droid_jointpos.auto_env_registrations import (
    auto_register_droid_envs,
)

logger = logging.getLogger(__name__)

# Obs-dict paths for the DROID joint-position observation group that robolab's
# own pi05 client reads from. See third_party/robolab/robolab/inference/pi0_family.py.
CAMERA_KEYS = {
    "external_cam": ("image_obs", "external_cam"),
    "wrist_cam": ("image_obs", "wrist_cam"),
}
STATE_KEYS = {
    "joint_position": ("proprio_obs", "arm_joint_pos"),
    "gripper_position": ("proprio_obs", "gripper_pos"),
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    # RoboLab task name (e.g. "BananaInBowlTask", "RubiksCubeTask"). Must be
    # one of the tasks auto-registered from robolab/tasks — run
    # ``uv run python ../../third_party/robolab/scripts/check_registered_envs.py``
    # to list available tasks.
    task_name: str = "BananaInBowlTask"
    # Instruction variant when the task defines multiple ("default", "vague", ...).
    instruction_type: str = "default"
    # Number of parallel envs inside the single Isaac Sim process.
    num_envs: int = 1
    # Number of sequential runs; total episodes = num_runs * num_envs.
    num_runs: int = 1
    # Override the task's default max_episode_length.
    max_steps: Optional[int] = None
    # Number of steps to execute from the model's action plan before re-planning.
    replan_steps: int = 8

    # Image resize size for the policy input.
    resize_size: int = 224

    fps: Optional[int] = None
    seed: int = 7
    device: str = "cuda:0"

    # If True, attach activation-collection metadata to every infer call so the
    # server (started with --collect_activations) saves intermediates to its disk.
    collect: bool = False

    # Override the top-level output directory. If None, defaults to
    # ``output/single-{instruction_type}``.
    output_dir: Optional[str] = None


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """RoboLab returns camera tensors as [num_envs, H, W, 3] on GPU. Detach to host uint8."""
    return tensor.detach().cpu().numpy()


def _to_numpy_state(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def tile_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Arrange N frames into a grid image (same helper as robocasa_env/main.py)."""
    n = len(frames)
    h, w, c = frames[0].shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    grid = np.zeros((rows * h, cols * w, c), dtype=frames[0].dtype)
    for idx, frame in enumerate(frames):
        r, col = divmod(idx, cols)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = frame
    return grid


def resolve_env_name(task_name: str) -> str:
    """Run robolab's auto-registration and return the gym env id for ``task_name``."""
    auto_register_droid_envs(task=[task_name])
    matches = get_envs(task=[task_name])
    if not matches:
        raise ValueError(
            f"No registered robolab env found for task_name={task_name!r}. "
            "List available tasks with: uv run python "
            "../../third_party/robolab/scripts/check_registered_envs.py"
        )
    # A task may have multiple registered variants (e.g. different backgrounds);
    # pick the first one, matching what robolab's run_eval.py does when you pass
    # --task for a single name.
    return matches[0]


def make_env(args: Args):
    env_id = resolve_env_name(args.task_name)
    env, env_cfg = create_env(
        env_id,
        device=args.device,
        seed=args.seed,
        num_envs=args.num_envs,
        use_fabric=True,
        instruction_type=args.instruction_type,
        policy="pi05",
    )
    return env, env_cfg, env_id


def build_policy_request(
    obs: dict,
    instruction: str,
    resize_size: int,
    env_id: int,
) -> dict:
    """Assemble the DROID observation payload the pi0/pi05 server expects.

    This mirrors Pi0DroidJointposClient._extract_observation in
    third_party/robolab/robolab/inference/pi0_family.py:87-107 so the wire
    format is byte-identical to what robolab's own client would send.
    """
    external = _to_numpy_image(obs["image_obs"]["external_cam"][env_id])
    wrist = _to_numpy_image(obs["image_obs"]["wrist_cam"][env_id])
    joint = _to_numpy_state(obs["proprio_obs"]["arm_joint_pos"][env_id])
    gripper = _to_numpy_state(obs["proprio_obs"]["gripper_pos"][env_id])

    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            external, resize_size, resize_size
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(
            wrist, resize_size, resize_size
        ),
        "observation/joint_position": joint,
        "observation/gripper_position": gripper,
        "prompt": instruction,
    }


def binarize_gripper(action: np.ndarray) -> np.ndarray:
    """Threshold the 8th action channel at 0.5 — matches Pi0DroidJointposClient."""
    gripper = 1.0 if float(action[-1]) > 0.5 else 0.0
    return np.concatenate([action[:-1], np.asarray([gripper], dtype=action.dtype)])


def eval_task(
    env,
    env_cfg,
    policy: _websocket_client_policy.WebsocketClientPolicy,
    args: Args,
    output_dir: str,
    collect_session: CollectionSession | None = None,
) -> dict[str, float]:
    """Evaluate the env for ``args.num_runs`` runs of ``args.num_envs`` parallel episodes."""
    task_name = args.task_name
    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    instruction = env_cfg.instruction
    max_steps = args.max_steps if args.max_steps is not None else env.max_episode_length
    if args.fps is not None:
        fps = args.fps
    else:
        fps = max(1, int(round(1.0 / (env_cfg.sim.render_interval * env_cfg.sim.dt))))

    num_envs = env.num_envs
    successes: list[bool] = []

    for run_idx in range(args.num_runs):
        obs, _ = env.reset()
        # One action plan deque per parallel env.
        action_plans: list[collections.deque] = [
            collections.deque() for _ in range(num_envs)
        ]

        video_paths = [
            os.path.join(
                task_output_dir,
                f"run_{run_idx:02d}_env{env_id:02d}.mp4",
            )
            for env_id in range(num_envs)
        ]
        videos = [VideoWriter(path, fps=fps) for path in video_paths]

        if collect_session is not None:
            for env_id in range(num_envs):
                collect_session.start_episode(
                    task_name=task_name,
                    task_id=0,
                    episode_id=run_idx * num_envs + env_id,
                    prompt=str(instruction),
                )

        try:
            pbar = tqdm(
                range(max_steps),
                desc=f"[{task_name}] Run {run_idx + 1}/{args.num_runs}",
                leave=False,
            )
            for step in pbar:
                # Stack one action per env; frozen envs keep zeros and will be
                # ignored by robolab's internal termination bookkeeping.
                actions = torch.zeros(num_envs, 8, device=env.device)

                for env_id in env.active_env_ids:
                    # Record a tiled (external, wrist) frame for this env.
                    external = _to_numpy_image(obs["image_obs"]["external_cam"][env_id])
                    wrist = _to_numpy_image(obs["image_obs"]["wrist_cam"][env_id])
                    videos[env_id].write(
                        tile_frames(
                            [
                                image_tools.convert_to_uint8(external),
                                image_tools.convert_to_uint8(wrist),
                            ]
                        )
                    )

                    if not action_plans[env_id]:
                        element = build_policy_request(
                            obs, instruction, args.resize_size, env_id
                        )
                        if collect_session is not None:
                            element["__collect__"] = (
                                collect_session.make_collect_metadata(step)
                            )
                        result = policy.infer(element)
                        action_chunk = np.asarray(result["actions"], dtype=np.float32)
                        if action_chunk.ndim != 2:
                            raise ValueError(
                                "Model output must be (action_horizon, action_dim), "
                                f"got {action_chunk.shape}"
                            )
                        if action_chunk.shape[0] < args.replan_steps:
                            raise ValueError(
                                f"Model must output at least {args.replan_steps} "
                                f"actions, got {action_chunk.shape[0]}"
                            )
                        for t in range(args.replan_steps):
                            action_plans[env_id].append(action_chunk[t])

                    action = binarize_gripper(action_plans[env_id].popleft())
                    actions[env_id] = torch.as_tensor(action, device=env.device)

                obs, reward, term, trunc, info = env.step(actions)

                if collect_session is not None:
                    # RoboLab exposes per-env reward tensors; record each env
                    # independently so the activation dump matches the video.
                    reward_cpu = reward.detach().cpu().numpy()
                    term_cpu = term.detach().cpu().numpy()
                    for env_id in range(num_envs):
                        collect_session.record_step(
                            step, float(reward_cpu[env_id]), bool(term_cpu[env_id])
                        )

                if env.all_terminated:
                    break
        finally:
            for video in videos:
                video.release()

        if collect_session is not None:
            for _ in range(num_envs):
                collect_session.finalize_episode()

        results = env.get_env_results()  # list of {env_id, success, step}
        for r in results:
            successes.append(bool(r["success"]))
            logger.info(
                "[%s] run=%d env=%d success=%s step=%d video=%s",
                task_name,
                run_idx,
                r["env_id"],
                r["success"],
                r["step"],
                video_paths[r["env_id"]],
            )

        env.reset_eval_state()

    success_rate = float(np.mean(successes)) if successes else 0.0
    return {
        "success_rate": success_rate,
        "num_episodes": float(len(successes)),
    }


def main(args: Args) -> None:
    np.random.seed(args.seed)

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(__file__),
            "output",
            f"single-{args.instruction_type}",
        )
    os.makedirs(output_dir, exist_ok=True)

    env = None
    result = None
    try:
        env, env_cfg, env_id = make_env(args)

        policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        print(f"[main] Server metadata: {policy.get_server_metadata()}", flush=True)

        collect_session = CollectionSession(policy) if args.collect else None

        result = eval_task(
            env, env_cfg, policy, args, output_dir, collect_session=collect_session
        )
        print(
            f"[main] [{args.task_name}/{args.instruction_type}] "
            f"success_rate={result['success_rate']:.2f} "
            f"({int(result['success_rate'] * result['num_episodes'])}/"
            f"{int(result['num_episodes'])})",
            flush=True,
        )
    except Exception:
        print("[main] Exception during eval:", flush=True)
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
