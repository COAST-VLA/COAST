"""
Collect intermediate activations (v2) from pi0.5 during MetaWorld evaluation rollouts.

V2 improvements over v1:
- Selective denoising step collection (steps 0, 4, 9 only)
- Attention weight capture at specified layers
- adaRMS gate capture for all 18 expert layers
- Enhanced metadata (proprio_state, object_positions, predicted_actions)
- Global adaRMS conditioning saved once per checkpoint

Uses PyTorch inference with register_forward_hook for per-layer activation extraction.
Loads policy in-process (no WebSocket server).

Usage:
    export CUDA_VISIBLE_DEVICES=1
    MUJOCO_GL=egl uv run scripts/collect_activations_v2.py \
        --policy.config=pi05_metaworld \
        --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
        --tasks reach-v3 --num_envs 2
"""

import collections
import dataclasses
import json
import logging
import pathlib
import subprocess
import sys
from typing import Literal

import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
from tqdm import tqdm
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger(__name__)

CAMERA_IDS = {
    "topview": 0,
    "corner": 1,
    "corner2": 2,
    "corner3": 3,
    "corner4": 4,
    "behindGripper": 5,
    "gripperPOV": 6,
}

TASK_TO_PROMPT = {
    "assembly-v3": "pick up the nut and place it onto the peg",
    "disassemble-v3": "pick up the nut and remove it from the peg",
    "basketball-v3": "dunk the basketball into the hoop",
    "soccer-v3": "kick the soccer ball into the goal",
    "bin-picking-v3": "pick up the object and place it into the bin",
    "box-close-v3": "grasp the cover and close the box",
    "button-press-v3": "press the button",
    "button-press-topdown-v3": "press the button from the top",
    "button-press-topdown-wall-v3": "press the button on the wall from the top",
    "button-press-wall-v3": "press the button on the wall",
    "coffee-button-v3": "push the button on the coffee machine",
    "coffee-pull-v3": "pull the mug away from the coffee machine",
    "coffee-push-v3": "push the mug under the coffee machine",
    "dial-turn-v3": "rotate the dial",
    "lever-pull-v3": "pull the lever down",
    "door-close-v3": "close the door",
    "door-lock-v3": "lock the door by rotating the lock",
    "door-open-v3": "open the door",
    "door-unlock-v3": "unlock the door by rotating the lock",
    "drawer-close-v3": "push the drawer closed",
    "drawer-open-v3": "pull the drawer open",
    "faucet-close-v3": "rotate the faucet handle to close it",
    "faucet-open-v3": "rotate the faucet handle to open it",
    "hammer-v3": "hammer the nail into the board",
    "hand-insert-v3": "insert the gripper into the hole",
    "handle-press-v3": "press the handle down",
    "handle-press-side-v3": "press the handle down sideways",
    "handle-pull-v3": "pull the handle up",
    "handle-pull-side-v3": "pull the handle sideways",
    "peg-insert-side-v3": "insert the peg into the hole sideways",
    "peg-unplug-side-v3": "unplug the peg from the hole sideways",
    "pick-out-of-hole-v3": "pick the object out of the hole",
    "pick-place-v3": "pick up the object and place it at the goal",
    "pick-place-wall-v3": "pick up the object and place it at the goal behind the wall",
    "plate-slide-v3": "slide the plate to the goal",
    "plate-slide-back-v3": "slide the plate backwards to the goal",
    "plate-slide-back-side-v3": "slide the plate backwards and sideways to the goal",
    "plate-slide-side-v3": "slide the plate sideways to the goal",
    "push-v3": "push the object to the goal",
    "push-back-v3": "push the object backwards to the goal",
    "push-wall-v3": "push the object around the wall to the goal",
    "reach-v3": "reach the goal position",
    "reach-wall-v3": "reach the goal position behind the wall",
    "shelf-place-v3": "pick up the object and place it on the shelf",
    "stick-pull-v3": "use the stick to pull the object",
    "stick-push-v3": "use the stick to push the object",
    "sweep-v3": "sweep the object off the table",
    "sweep-into-v3": "sweep the object into the hole",
    "window-close-v3": "push the window closed",
    "window-open-v3": "push the window open",
}

ML45_TRAIN = [
    "assembly-v3",
    "basketball-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-open-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-close-v3",
    "faucet-open-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "pick-place-wall-v3",
    "plate-slide-back-side-v3",
    "plate-slide-back-v3",
    "plate-slide-side-v3",
    "plate-slide-v3",
    "push-back-v3",
    "push-v3",
    "push-wall-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "soccer-v3",
    "stick-pull-v3",
    "stick-push-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-close-v3",
    "window-open-v3",
]

ML45_TEST = [
    "bin-picking-v3",
    "box-close-v3",
    "door-lock-v3",
    "door-unlock-v3",
    "hand-insert-v3",
]

# V2 collection parameters (must match model method defaults)
COLLECT_DENOISE_STEPS = (0, 4, 9)
RESIDUAL_LAYERS = (5, 11)
MLP_LAYERS = (11,)
ATTENTION_LAYERS = (5, 11)


class MultiCameraWrapper(gym.Wrapper):
    """Wrapper that renders multiple cameras and includes images in info dict."""

    def __init__(self, env: gym.Env, camera_names: list[str]):
        super().__init__(env)
        self.camera_names = camera_names

    def _render_cameras(self) -> dict[str, np.ndarray]:
        renderer = self.unwrapped.mujoco_renderer
        images = {}
        for cam_name in self.camera_names:
            viewer = renderer._get_viewer(render_mode="rgb_array")  # noqa: SLF001
            if len(renderer._viewers.keys()) >= 1:  # noqa: SLF001
                viewer.make_context_current()
            img = viewer.render(render_mode="rgb_array", camera_id=CAMERA_IDS[cam_name])
            images[cam_name] = img[::-1].copy()
        return images

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["cameras"] = self._render_cameras()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["cameras"] = self._render_cameras()
        return obs, reward, terminated, truncated, info


@dataclasses.dataclass
class PolicyArgs:
    config: str = "pi05_metaworld"
    dir: str = "checkpoints/pi05_metaworld/pi05_metaworld_test/5000/"


@dataclasses.dataclass
class Args:
    policy: PolicyArgs = dataclasses.field(default_factory=PolicyArgs)

    # Tasks to collect activations for. If empty, uses --split to select.
    tasks: list[str] = dataclasses.field(default_factory=list)
    # ML45 split to use when --tasks is empty.
    split: Literal["train", "test"] = "train"
    # Number of parallel environments per task.
    num_envs: int = 2
    # Maximum steps per episode.
    max_steps: int = 300
    # Number of steps between re-planning.
    replan_steps: int = 10
    # Output directory for activations.
    output_dir: str = "activations_v2"

    width: int = 224
    height: int = 224
    policy_cameras: list[str] = dataclasses.field(default_factory=lambda: ["corner", "corner4", "gripperPOV"])
    seed: int = 69_420
    # GPU IDs to use for parallel collection. Each GPU loads its own model copy.
    # Example: --gpus 0 1 runs two models in parallel on cuda:0 and cuda:1.
    gpus: list[int] = dataclasses.field(default_factory=list)


def save_step_activations_v2(
    step_dir: pathlib.Path,
    intermediates: dict,
    env_id: int,
    step_metadata: dict,
):
    """Save per-env, per-step activation data (v2 format).

    Intermediates shapes (before env slicing):
    - all_x_t: (3, batch, action_horizon, action_dim)
    - all_v_t: (3, batch, action_horizon, action_dim)
    - all_suffix_residual: (3, 2, batch, action_horizon, 1024)
    - all_suffix_mlp_hidden: (3, 1, batch, action_horizon, 4096)
    - all_attention_weights: (3, 2, batch, 8, action_horizon, total_seq_len)
    - all_adarms_gates: (3, 18, 2, batch, action_horizon, 1024)
    """
    step_dir.mkdir(parents=True, exist_ok=True)

    # Slice out this env's data from the batch dimension
    # all_x_t: (3, batch, 32, 32) -> (3, 32, 32)
    all_x_t = intermediates["all_x_t"][:, env_id]
    all_v_t = intermediates["all_v_t"][:, env_id]

    # all_suffix_residual: (3, 2, batch, seq, 1024) -> (3, 2, 32, 1024)
    all_suffix_residual = intermediates["all_suffix_residual"][:, :, env_id]

    # all_suffix_mlp_hidden: (3, 1, batch, seq, 4096) -> (3, 1, 32, 4096)
    all_suffix_mlp_hidden = intermediates["all_suffix_mlp_hidden"][:, :, env_id]

    # all_attention_weights: (3, 2, batch, 8, seq, total_seq) -> (3, 2, 8, 32, total_seq)
    all_attention_weights = intermediates["all_attention_weights"][:, :, env_id]

    # all_adarms_gates: (3, 18, 2, batch, seq, 1024) -> (3, 18, 2, 32, 1024)
    all_adarms_gates = intermediates["all_adarms_gates"][:, :, :, env_id]

    np.savez(step_dir / "denoising.npz", all_x_t=all_x_t, all_v_t=all_v_t)
    np.savez(step_dir / "suffix_residual.npz", all_suffix_residual=all_suffix_residual)
    np.savez(step_dir / "suffix_mlp_hidden.npz", all_suffix_mlp_hidden=all_suffix_mlp_hidden)
    np.savez(step_dir / "attention_weights.npz", all_attention_weights=all_attention_weights)
    np.savez(step_dir / "adarms_gates.npz", all_adarms_gates=all_adarms_gates)

    with open(step_dir / "metadata.json", "w") as f:
        json.dump(step_metadata, f, indent=2)


def collect_task(policy, task_name: str, args: Args, base_output_dir: pathlib.Path, adarms_cond_saved: set):
    """Collect activations for a single task."""
    logger.info(f"Collecting activations for {task_name}")
    prompt = TASK_TO_PROMPT[task_name]
    num_envs = args.num_envs

    # Use AsyncVectorEnv with context="spawn" to parallelize env stepping.
    # Default fork() causes CUDA/EGL deadlocks; spawn starts fresh interpreters.
    env_fns = [
        lambda i=i: MultiCameraWrapper(
            gym.make(
                "Meta-World/MT1",
                env_name=task_name,
                seed=args.seed + i,
                width=args.width,
                height=args.height,
            ),
            args.policy_cameras,
        )
        for i in range(num_envs)
    ]
    env = gym.vector.AsyncVectorEnv(env_fns, context="spawn")

    try:
        obs, info = env.reset(seed=args.seed)
        camera_views = info["cameras"]
        success = np.zeros(num_envs, dtype=bool)
        cumulative_reward = np.zeros(num_envs)
        steps_to_success = np.full(num_envs, -1, dtype=int)
        action_plan = collections.deque()
        inference_step = 0

        # Per-env reward tracking
        per_step_rewards = [[] for _ in range(num_envs)]
        per_step_success = [[] for _ in range(num_envs)]
        reward_at_last_inference = np.zeros(num_envs)

        task_output_dir = base_output_dir / task_name

        pbar = tqdm(range(args.max_steps), desc=task_name)
        for step in pbar:
            if not action_plan:
                # Keep full observation for metadata extraction
                full_obs = obs.copy()

                # Build observation dict matching the eval pipeline
                obs_dict = {
                    "observation/image": camera_views["corner4"],
                    "observation/wrist_image": camera_views["gripperPOV"],
                    "observation/state": obs.astype(np.float32)[..., :4],
                    "prompt": [prompt] * num_envs,
                }

                result, intermediates = policy.infer_with_intermediates_v2(obs_dict)
                action_chunk = np.clip(result["actions"], -1.0, 1.0).astype(np.float32)

                # Save adaRMS conditioning globally ONCE (not per step)
                if "adarms_cond_saved" not in dir() or not adarms_cond_saved:
                    cond_dir = base_output_dir
                    cond_dir.mkdir(parents=True, exist_ok=True)
                    if "global" not in adarms_cond_saved:
                        np.savez(
                            cond_dir / "adarms_cond_global.npz",
                            adarms_cond_global=intermediates["adarms_cond_global"],
                        )
                        adarms_cond_saved.add("global")
                        logger.info(f"Saved global adaRMS conditioning to {cond_dir / 'adarms_cond_global.npz'}")

                # Save per-env activations for this inference step
                reward_since_last = cumulative_reward - reward_at_last_inference
                for env_id in range(num_envs):
                    episode_dir = task_output_dir / f"episode_000_env_{env_id:03d}"
                    step_dir = episode_dir / f"step_{step:04d}"

                    # Enhanced metadata with proprio_state, object_positions, predicted_actions
                    proprio_state = full_obs[env_id, :4].tolist()  # gripper xyz + angle
                    object_positions = full_obs[env_id, 4:7].tolist()  # object xyz
                    predicted_actions = action_chunk[env_id, 0, :].tolist()  # first action in chunk (4D)

                    step_metadata = {
                        "task_name": task_name,
                        "episode_id": 0,
                        "env_id": env_id,
                        "step": step,
                        "inference_step": inference_step,
                        "prompt": prompt,
                        "cumulative_reward": float(cumulative_reward[env_id]),
                        "success_so_far": bool(success[env_id]),
                        "reward_since_last_inference": float(reward_since_last[env_id]),
                        "proprio_state": proprio_state,
                        "object_positions": object_positions,
                        "predicted_actions": predicted_actions,
                    }
                    save_step_activations_v2(step_dir, intermediates, env_id, step_metadata)

                reward_at_last_inference = cumulative_reward.copy()
                inference_step += 1

                for t in range(args.replan_steps):
                    action_plan.append(action_chunk[:, t, :])

            action = action_plan.popleft()
            obs, reward, terminated, truncated, info = env.step(action)
            camera_views = info["cameras"]
            cumulative_reward += reward

            step_success = np.asarray(info.get("success", np.zeros(num_envs)), dtype=bool)
            for env_id in range(num_envs):
                per_step_rewards[env_id].append(float(reward[env_id]))
                per_step_success[env_id].append(bool(step_success[env_id]))
                if step_success[env_id] and steps_to_success[env_id] == -1:
                    steps_to_success[env_id] = step
            success |= step_success
            if success.all():
                break

            pbar.set_postfix(reward=f"{cumulative_reward.mean():.1f}", success=f"{success.mean():.0%}")

        # Write episode-level metadata and reward trajectories
        total_env_steps = len(per_step_rewards[0])
        for env_id in range(num_envs):
            episode_dir = task_output_dir / f"episode_000_env_{env_id:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            episode_metadata = {
                "task_name": task_name,
                "episode_id": 0,
                "env_id": env_id,
                "episode_success": bool(success[env_id]),
                "total_reward": float(cumulative_reward[env_id]),
                "steps_to_success": int(steps_to_success[env_id]),
                "total_env_steps": total_env_steps,
                "total_inference_steps": inference_step,
                "prompt": prompt,
                "checkpoint_dir": args.policy.dir,
                "config_name": args.policy.config,
                # V2 enhanced episode metadata
                "collection_version": "v2",
                "collected_denoise_steps": list(COLLECT_DENOISE_STEPS),
                "collected_residual_layers": list(RESIDUAL_LAYERS),
                "collected_mlp_layers": list(MLP_LAYERS),
                "collected_attention_layers": list(ATTENTION_LAYERS),
            }
            with open(episode_dir / "metadata.json", "w") as f:
                json.dump(episode_metadata, f, indent=2)

            # Reward trajectory
            rewards_arr = np.array(per_step_rewards[env_id], dtype=np.float32)
            cumulative_arr = np.cumsum(rewards_arr).astype(np.float32)
            success_arr = np.array(per_step_success[env_id], dtype=bool)
            np.savez(
                episode_dir / "rewards.npz",
                per_step_reward=rewards_arr,
                cumulative_reward=cumulative_arr,
                success_at_step=success_arr,
            )

        logger.info(
            f"{task_name}: mean_reward={cumulative_reward.mean():.2f}, "
            f"success_rate={success.mean():.0%}, inference_steps={inference_step}"
        )
    finally:
        env.close()


def run_single_gpu(args: Args) -> None:
    """Run collection on a single GPU (current process)."""
    logger.info("Single-GPU mode (v2)")

    # Ensure PyTorch checkpoint exists before loading policy.
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint

    # Determine tasks
    if args.tasks:
        tasks = args.tasks
    elif args.split == "train":
        tasks = ML45_TRAIN
    else:
        tasks = ML45_TEST

    # Load policy in-process
    train_config = _config.get_config(args.policy.config)
    ensure_pytorch_checkpoint(args.policy.dir, args.policy.config)
    policy = _policy_config.create_trained_policy(train_config, args.policy.dir)
    logger.info(f"Policy loaded: pytorch={policy._is_pytorch_model}")  # noqa: SLF001
    if not policy._is_pytorch_model:  # noqa: SLF001
        raise RuntimeError("Activation collection requires a PyTorch checkpoint (model.safetensors).")

    # Determine checkpoint step from path for output dir structure
    policy_dir = pathlib.Path(args.policy.dir)
    checkpoint_step = policy_dir.name
    base_output_dir = pathlib.Path(args.output_dir) / checkpoint_step

    # Track whether we've saved the global adaRMS conditioning
    adarms_cond_saved = set()

    for task_name in tasks:
        if task_name not in TASK_TO_PROMPT:
            logger.warning(f"Unknown task: {task_name}, skipping")
            continue
        collect_task(policy, task_name, args, base_output_dir, adarms_cond_saved)

    logger.info(f"All activations saved to {base_output_dir}")


def run_multi_gpu(args: Args) -> None:
    """Launch one subprocess per GPU, each handling a subset of tasks.

    Uses `bash -c 'CUDA_VISIBLE_DEVICES=X ...'` to ensure the env var is set
    before any Python/CUDA initialization.
    """
    gpus = args.gpus

    if args.tasks:
        tasks = args.tasks
    elif args.split == "train":
        tasks = ML45_TRAIN
    else:
        tasks = ML45_TEST

    # Ensure PyTorch checkpoint exists once before spawning processes.
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint

    ensure_pytorch_checkpoint(args.policy.dir, args.policy.config)

    # Split tasks across GPUs round-robin
    task_chunks = [[] for _ in gpus]
    for i, task in enumerate(tasks):
        task_chunks[i % len(gpus)].append(task)

    log_dir = pathlib.Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for gpu_id, chunk in zip(gpus, task_chunks, strict=True):
        logger.info(f"GPU {gpu_id}: {len(chunk)} tasks — {chunk}")

    # Launch subprocesses
    processes = []
    for gpu_id, chunk in zip(gpus, task_chunks, strict=True):
        if not chunk:
            continue
        tasks_str = " ".join(chunk)
        # Use bash -c with inline CUDA_VISIBLE_DEVICES to guarantee env propagation
        inner_cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} MUJOCO_GL=egl"
            f" {sys.executable} {__file__}"
            f" --policy.config={args.policy.config}"
            f" --policy.dir={args.policy.dir}"
            f" --num_envs={args.num_envs}"
            f" --max_steps={args.max_steps}"
            f" --replan_steps={args.replan_steps}"
            f" --output_dir={args.output_dir}"
            f" --seed={args.seed}"
            f" --split={args.split}"
            f" --tasks {tasks_str}"
        )
        log_file = log_dir / f"gpu_{gpu_id}.log"
        log_fh = open(log_file, "w")  # noqa: SIM115
        logger.info(f"Starting GPU {gpu_id} (log: {log_file})")
        proc = subprocess.Popen(
            ["bash", "-c", inner_cmd],
            stdout=log_fh,
            stderr=log_fh,
        )
        processes.append((gpu_id, proc, log_fh))

    # Wait for all to finish
    for gpu_id, proc, log_fh in processes:
        proc.wait()
        log_fh.close()
        if proc.returncode != 0:
            logger.error(f"GPU {gpu_id} exited with code {proc.returncode}. See {log_dir / f'gpu_{gpu_id}.log'}")
        else:
            logger.info(f"GPU {gpu_id} completed successfully")


def main(args: Args) -> None:
    if args.gpus:
        if len(args.gpus) == 1:
            # Single GPU specified — just run in this process
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
            run_single_gpu(args)
        else:
            run_multi_gpu(args)
    else:
        run_single_gpu(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
