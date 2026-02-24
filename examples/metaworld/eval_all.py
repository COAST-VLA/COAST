"""
NOTE: This creates a lot of parallel environments and may consume a lot of resources. Use main.py
      if you want to run inference on a single task.

      We might deprecate this script in the future in favor of single task eval.

Evaluate on all 45 ML45 train tasks (default):
- MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --benchmark_name ML45-train

Evaluate on 5 ML45 test tasks:
- MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --benchmark_name ML45-test
"""

import collections
import dataclasses
import logging
import math
import os
from typing import Literal

import gymnasium as gym
import imageio.v3 as iio
import metaworld  # noqa: F401
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from tqdm import tqdm
import tyro

logger = logging.getLogger(__name__)

# https://metaworld.farama.org/rendering/rendering/#render-from-a-specific-camera
CAMERA_IDS = {
    "topview": 0,
    "corner": 1,
    "corner2": 2,
    "corner3": 3,
    "corner4": 4,
    "behindGripper": 5,
    "gripperPOV": 6,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    benchmark_name: Literal["MT1", "MT10", "MT50", "ML1", "ML10-test", "ML10-train", "ML45-test", "ML45-train"] = (
        "ML45-test"
    )
    env_name: str | None = None

    width: int = 224
    height: int = 224

    # Cameras to use for policy input
    policy_cameras: list[str] = dataclasses.field(default_factory=lambda: ["corner", "corner4", "gripperPOV"])
    # The camera used for rendering the video output (must be one of the policy cameras)
    render_camera: str = "corner"

    num_episodes: int = 2
    max_steps: int = 200
    fps: int = 24

    seed: int = 42


class MultiCameraVectorWrapper(gym.vector.VectorWrapper):
    """
    Gym wrapper to render multiple camera views at each step and include them in the info dict.

    info["cameras"]: list[dict[str, np.ndarray]] # camera_names -> image
        - list length = env vector size
        - images are (H, W, 3) uint8 RGB
    """

    def __init__(self, env: gym.vector.VectorEnv, camera_names: list[str]):
        super().__init__(env)
        self.camera_names = camera_names

    def _render_cameras_one(self, e) -> dict[str, np.ndarray]:
        renderer = e.unwrapped.mujoco_renderer
        images = {}
        for cam_name in self.camera_names:
            # HACK (branyang02): Very Very Very Hacky
            # Take a look at gymnasium.envs.muojoco.mujoco_rendering.MujocoRenderer.render()
            # Implemented solutions from:
            # https://github.com/Farama-Foundation/Metaworld/issues/448
            # https://github.com/Farama-Foundation/Gymnasium/issues/736
            viewer = renderer._get_viewer(render_mode="rgb_array")  # noqa: SLF001
            if len(renderer._viewers.keys()) >= 1:  # noqa: SLF001
                viewer.make_context_current()
            img = viewer.render(render_mode="rgb_array", camera_id=CAMERA_IDS[cam_name])
            images[cam_name] = img[::-1].copy()  # flip vertically
        return images

    def _render_all(self) -> list[dict[str, np.ndarray]]:
        return [self._render_cameras_one(e) for e in self.env.envs]

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        infos["cameras"] = self._render_all()
        return obs, infos

    def step(self, actions):
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        infos["cameras"] = self._render_all()
        return obs, rewards, terms, truncs, infos


def tile_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Arrange N frames into a grid image.

    Grid layout: cols = ceil(sqrt(N)), rows = ceil(N / cols).
    Empty slots are filled with black.
    """
    n = len(frames)
    h, w, c = frames[0].shape
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    grid = np.zeros((rows * h, cols * w, c), dtype=frames[0].dtype)
    for idx, frame in enumerate(frames):
        r, col = divmod(idx, cols)
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = frame

    return grid


def make_env(
    benchmark_name: str,
    env_name: str | None,
    seed: int,
    width: int = 224,
    height: int = 224,
    camera_names: list[str] | None = None,
    vector_strategy: Literal["sync", "async"] = "sync",
) -> gym.vector.VectorEnv:
    """
    Environment creation notes:

    - MT1 and ML1 create a single environment.
    - MT10 and MT50 create 10 and 50 environments respectively, each with different tasks.
    - ML10-test/ML10-train and ML45-test/ML45-train create environments from the test or train
    split of ML10 and ML45 respectively.
    - We wrap the environments in a custom wrapper to render multiple camera views at each step

    Task breakdown:

    ML10 test tasks (5 tasks):
        0: SawyerDrawerOpenEnvV3
        1: SawyerDoorCloseEnvV3
        2: SawyerShelfPlaceEnvV3
        3: SawyerSweepIntoGoalEnvV3
        4: SawyerLeverPullEnvV3

    ML45 test tasks (5 tasks):
        0: SawyerBinPickingEnvV3
        1: SawyerBoxCloseEnvV3
        2: SawyerHandInsertEnvV3
        3: SawyerDoorLockEnvV3
        4: SawyerDoorUnlockEnvV3

    References:
    - Meta-World environments: https://meta-world.github.io/
    - Environment creation code adapted from:
    https://metaworld.farama.org/introduction/basic_usage/
    """

    # TODO(branyang02): should we support async vector envs?
    if vector_strategy == "async":
        raise NotImplementedError("Async vector environments are not supported yet!")

    # TODO(branyang02): should we support MT1 and ML1?
    if benchmark_name == "MT1":
        raise NotImplementedError("MT1 is not implemented yet")
        env = gym.make("Meta-World/MT1", env_name=env_name, seed=seed)
    if benchmark_name == "ML1":
        raise NotImplementedError("ML1 is not implemented yet")
        env = gym.make("Meta-World/ML1-test", env_name=env_name, seed=seed)

    env = gym.make_vec(
        f"Meta-World/{benchmark_name}",
        vector_strategy=vector_strategy,
        seed=seed,
        width=width,
        height=height,
    )

    return MultiCameraVectorWrapper(env, camera_names)


def main(args: Args) -> None:
    policy = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    output_dir = os.path.join(os.path.dirname(__file__), "output", args.benchmark_name)
    os.makedirs(output_dir, exist_ok=True)

    env = make_env(
        args.benchmark_name,
        args.env_name,
        args.seed,
        width=args.width,
        height=args.height,
        camera_names=args.policy_cameras,
    )
    num_envs = env.num_envs

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + episode)
        camera_views = info["cameras"]
        total_reward = np.zeros(num_envs)
        success = np.zeros(num_envs, dtype=bool)
        action_plan = collections.deque()

        video_path = os.path.join(output_dir, f"episode_{episode:03d}.mp4")
        with iio.imopen(video_path, "w", plugin="pyav") as video:
            video.init_video_stream("h264", fps=args.fps)

            pbar = tqdm(range(args.max_steps), desc=f"Episode {episode + 1}/{args.num_episodes}")
            for _step in pbar:
                frames = [cv[args.render_camera] for cv in camera_views]  # list of (H, W, 3)
                grid_frame = tile_frames(frames) if num_envs > 5 else np.concatenate(frames, axis=1)
                video.write_frame(grid_frame)

                if not action_plan:
                    result = policy.infer(
                        {
                            "observation/image": np.stack([cv["corner4"] for cv in camera_views], axis=0),
                            "observation/wrist_image": np.stack([cv["gripperPOV"] for cv in camera_views], axis=0),
                            "observation/state": obs.astype(np.float32)[
                                ..., :4
                            ],  # first 4 dims are the true observable state in Metaworld.
                            # TODO(branyang02): replace placeholder prompt with task-specific prompts
                            "prompt": ["Perform the task successfully and efficiently."] * num_envs,
                        }
                    )
                    action_chunk = np.clip(result["actions"], -1.0, 1.0).astype(
                        np.float32
                    )  # (b, action_horizon, action_dim)
                    for t in range(action_chunk.shape[1]):
                        action_plan.append(action_chunk[:, t, :])

                action = action_plan.popleft()  # (num_envs, action_dim=4)
                # action = env.action_space.sample()  # (5, 4)

                obs, reward, terminated, truncated, info = env.step(action)
                camera_views = info["cameras"]
                total_reward += reward
                success |= np.asarray(info.get("success", np.zeros(num_envs)), dtype=bool)
                pbar.set_postfix(reward=f"{total_reward.mean():.1f}", success=f"{success.mean():.0%}")

        logger.info(
            f"Episode {episode + 1}/{args.num_episodes}: "
            f"mean_reward={total_reward.mean():.2f}, success_rate={success.mean():.2f}, "
            f"video={video_path}"
        )

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
