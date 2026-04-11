"""
Record video of a single steering condition for visual inspection.

Usage:
    MUJOCO_GL=osmesa python experiments/record_steering_video.py \
        --policy.config=pi05_metaworld \
        --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
        --task assembly-v3 \
        --condition strategy3 \
        --alpha 0.5 --beta 0.1 \
        --num-envs 15 \
        --output-dir experiments/steering_results/videos
"""

import collections
import dataclasses
import gc
import json
import logging
import math
import pathlib

import gymnasium as gym
import imageio.v3 as iio
import metaworld  # noqa: F401
import numpy as np
from huggingface_hub import hf_hub_download
import torch
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger(__name__)

# Import shared pieces from the main experiment script
from conceptor_steering import (
    CAMERA_IDS,
    TASK_TO_PROMPT,
    LAYER_MAP,
    REPO_ID,
    HF_CACHE,
    MultiCameraWrapper,
    ConceptorSteeringHook,
    find_steerable_tasks,
    build_steering_conceptors,
)


def tile_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Arrange N frames into a grid image."""
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
    task: str, num_envs: int, seed: int,
    width: int = 224, height: int = 224,
    camera_names: list[str] | None = None,
) -> gym.Env:
    if camera_names is None:
        camera_names = ["corner", "corner4", "gripperPOV"]
    env_fns = [
        lambda i=i: MultiCameraWrapper(
            gym.make("Meta-World/MT1", env_name=task, seed=seed + i, width=width, height=height),
            camera_names,
        )
        for i in range(num_envs)
    ]
    return gym.vector.AsyncVectorEnv(env_fns)


def run_episode_with_video(
    policy, env, task_name: str, num_envs: int,
    max_steps: int = 300, replan_steps: int = 10,
    steering_hooks=None,
    render_camera: str = "corner",
    fps: int = 24,
    video_path: pathlib.Path | None = None,
) -> list[dict]:
    """Run episode and optionally record video."""
    prompt = TASK_TO_PROMPT[task_name]
    obs, info = env.reset()
    camera_views = info["cameras"]
    success = np.zeros(num_envs, dtype=bool)
    cumulative_reward = np.zeros(num_envs)
    steps_to_success = np.full(num_envs, -1, dtype=int)
    action_plan = collections.deque()

    video_ctx = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_ctx = iio.imopen(str(video_path), "w", plugin="pyav")
        video_ctx.__enter__()
        video_ctx.init_video_stream("h264", fps=fps)

    try:
        for step in range(max_steps):
            # Write frame
            if video_ctx is not None:
                grid_frame = tile_frames(list(camera_views[render_camera]))
                video_ctx.write_frame(grid_frame)

            if not action_plan:
                obs_dict = {
                    "observation/image": camera_views["corner4"],
                    "observation/wrist_image": camera_views["gripperPOV"],
                    "observation/state": obs.astype(np.float32)[..., :4],
                    "prompt": [prompt] * num_envs,
                }

                if steering_hooks is not None:
                    for _, hook_fn in steering_hooks:
                        hook_fn.reset_logs()
                    result, _diag = policy.infer_with_steering(
                        obs_dict, steering_hooks=steering_hooks
                    )
                else:
                    result = policy.infer(obs_dict)

                action_chunk = np.clip(result["actions"], -1.0, 1.0).astype(np.float32)
                for t in range(min(replan_steps, action_chunk.shape[1])):
                    action_plan.append(action_chunk[:, t, :])

            action = action_plan.popleft()
            obs, reward, terminated, truncated, info = env.step(action)
            camera_views = info["cameras"]
            cumulative_reward += reward
            step_success = np.asarray(info.get("success", np.zeros(num_envs)), dtype=bool)
            for env_id in range(num_envs):
                if step_success[env_id] and steps_to_success[env_id] == -1:
                    steps_to_success[env_id] = step
            success |= step_success
            if success.all():
                # Write a few more frames so the video doesn't cut abruptly
                if video_ctx is not None:
                    for _ in range(fps):  # ~1 second extra
                        grid_frame = tile_frames(list(camera_views[render_camera]))
                        video_ctx.write_frame(grid_frame)
                break
    finally:
        if video_ctx is not None:
            video_ctx.__exit__(None, None, None)

    results = []
    for env_id in range(num_envs):
        results.append({
            "env_id": env_id,
            "success": bool(success[env_id]),
            "total_reward": float(cumulative_reward[env_id]),
            "steps_to_success": int(steps_to_success[env_id]),
        })
    return results


@dataclasses.dataclass
class PolicyArgs:
    config: str = "pi05_metaworld"
    dir: str = "checkpoints/pi05_metaworld/pi05_metaworld_test/5000/"


@dataclasses.dataclass
class Args:
    policy: PolicyArgs = dataclasses.field(default_factory=PolicyArgs)

    task: str = "assembly-v3"
    condition: str = "strategy3"  # strategy3, strategy5, posonly_global, posonly_perstep, baseline
    alpha: float = 0.5
    beta: float = 0.1
    steering_layer: int = 11
    num_envs: int = 15
    max_steps: int = 300
    replan_steps: int = 10
    seed: int = 69_420
    width: int = 224
    height: int = 224
    render_camera: str = "corner"
    fps: int = 24

    output_dir: str = "experiments/steering_results/videos"


def main(args: Args) -> None:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    logger.info("Loading policy...")
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    train_config = _config.get_config(args.policy.config)
    ensure_pytorch_checkpoint(args.policy.dir, args.policy.config)
    policy = _policy_config.create_trained_policy(train_config, args.policy.dir)
    device = policy._pytorch_device  # noqa: SLF001
    logger.info(f"Policy loaded on {device}")

    # Build conceptor if not baseline
    steering_hooks = None
    condition_label = "baseline"

    if args.condition != "baseline":
        logger.info(f"Building conceptors for {args.task}...")
        steerable = find_steerable_tasks([args.task])
        if args.task not in steerable:
            raise ValueError(f"Task {args.task} has no steerable episodes")
        env_splits = steerable[args.task]
        layer_idx = LAYER_MAP[args.steering_layer]

        (global_C, step_Cs, pos_global_C, pos_step_Cs,
         diagnostics, _, _) = build_steering_conceptors(
            args.task, env_splits, alpha=args.alpha, layer_idx=layer_idx
        )

        if args.condition == "strategy3":
            hook = ConceptorSteeringHook(
                strategy="global", global_conceptor=global_C,
                beta=args.beta, device=device,
            )
            condition_label = f"strategy3_a{args.alpha}_b{args.beta}"
        elif args.condition == "strategy5":
            hook = ConceptorSteeringHook(
                strategy="per_step", step_conceptors=step_Cs,
                beta=args.beta, device=device,
            )
            condition_label = f"strategy5_a{args.alpha}_b{args.beta}"
        elif args.condition == "posonly_global":
            hook = ConceptorSteeringHook(
                strategy="global", global_conceptor=pos_global_C,
                beta=args.beta, device=device,
            )
            condition_label = f"posonly_global_a{args.alpha}_b{args.beta}"
        elif args.condition == "posonly_perstep":
            hook = ConceptorSteeringHook(
                strategy="per_step", step_conceptors=pos_step_Cs,
                beta=args.beta, device=device,
            )
            condition_label = f"posonly_perstep_a{args.alpha}_b{args.beta}"
        else:
            raise ValueError(f"Unknown condition: {args.condition}")

        steering_hooks = [(args.steering_layer, hook)]

    # Record baseline video
    logger.info(f"Recording baseline for {args.task}...")
    camera_names = ["corner", "corner4", "gripperPOV"]
    env = make_env(args.task, args.num_envs, args.seed, args.width, args.height, camera_names)
    baseline_video = output_dir / f"{args.task}_baseline.mp4"
    try:
        baseline_results = run_episode_with_video(
            policy, env, args.task, args.num_envs,
            args.max_steps, args.replan_steps,
            steering_hooks=None,
            render_camera=args.render_camera,
            fps=args.fps,
            video_path=baseline_video,
        )
    finally:
        env.close()
        gc.collect()
        torch.cuda.empty_cache()

    baseline_sr = np.mean([r["success"] for r in baseline_results])
    logger.info(f"  Baseline SR={baseline_sr:.3f}, video saved to {baseline_video}")

    # Record steered video (if not baseline-only)
    if steering_hooks is not None:
        logger.info(f"Recording {condition_label} for {args.task}...")
        env = make_env(args.task, args.num_envs, args.seed, args.width, args.height, camera_names)
        steered_video = output_dir / f"{args.task}_{condition_label}.mp4"
        try:
            steered_results = run_episode_with_video(
                policy, env, args.task, args.num_envs,
                args.max_steps, args.replan_steps,
                steering_hooks=steering_hooks,
                render_camera=args.render_camera,
                fps=args.fps,
                video_path=steered_video,
            )
        finally:
            env.close()
            gc.collect()
            torch.cuda.empty_cache()

        steered_sr = np.mean([r["success"] for r in steered_results])
        logger.info(f"  {condition_label} SR={steered_sr:.3f}, video saved to {steered_video}")
        logger.info(f"  Delta: {steered_sr - baseline_sr:+.3f}")

    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = tyro.cli(Args)
    main(args)
