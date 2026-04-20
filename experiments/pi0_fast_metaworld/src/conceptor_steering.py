#!/usr/bin/env python3
"""
Conceptor-Based Steering for pi0-fast MetaWorld
================================================
In-process steered evaluation (no WebSocket server).  Loads policy once,
sweeps steering conditions, writes summary.json per task.

Usage:
    uv run experiments/pi0_fast_metaworld/src/conceptor_steering.py \
        --task assembly-v3
"""
import collections
import dataclasses
import json
import logging
import os
import pathlib
from typing import List, Optional

import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger(__name__)

OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))

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

CAMERA_IDS = {
    "topview": 0, "corner": 1, "corner2": 2, "corner3": 3,
    "corner4": 4, "behindGripper": 5, "gripperPOV": 6,
}
POLICY_CAMERAS = ["corner", "corner4", "gripperPOV"]


# ── Steering matrix construction ─────────────────────────────────────────


def load_npz(path):
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"Conceptor file not found: {path}")
    return np.load(path, allow_pickle=True)


def get_conceptor(npz, task: str, strategy: str, alpha: float, kind: str) -> np.ndarray:
    key = f"{task}__{strategy}__{alpha}__{kind}"
    if key not in npz:
        raise KeyError(f"Conceptor not in npz: {key}")
    return npz[key]


def build_M(C: np.ndarray, beta: float) -> np.ndarray:
    d = C.shape[0]
    return ((1.0 - beta) * np.eye(d, dtype=C.dtype) + beta * C).astype(np.float32)


def build_M_per_step_combined(
    npz, task: str, alpha: float, beta: float, max_steps: int, kind: str = "C_contrastive",
) -> np.ndarray:
    """Build a (max_steps, d, d) steering tensor that applies position-aware conceptors.

    Tokens 0..T/3 use per_token_first, T/3..2T/3 use per_token_mid, 2T/3..T use per_token_last.
    """
    C_first = get_conceptor(npz, task, "per_token_first", alpha, kind)
    C_mid = get_conceptor(npz, task, "per_token_mid", alpha, kind)
    C_last = get_conceptor(npz, task, "per_token_last", alpha, kind)

    d = C_first.shape[0]
    I = np.eye(d, dtype=np.float32)
    out = np.empty((max_steps, d, d), dtype=np.float32)

    t1 = max_steps // 3
    t2 = 2 * max_steps // 3
    for i in range(max_steps):
        if i < t1:
            C = C_first
        elif i < t2:
            C = C_mid
        else:
            C = C_last
        out[i] = (1.0 - beta) * I + beta * C.astype(np.float32)
    return out


# ── In-process MetaWorld evaluation ──────────────────────────────────────


class MultiCameraWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, camera_names: list):
        super().__init__(env)
        self.camera_names = camera_names

    def _render_cameras(self):
        renderer = self.unwrapped.mujoco_renderer
        images = {}
        for name in self.camera_names:
            cam_id = CAMERA_IDS[name]
            images[name] = renderer.render("rgb_array", camera_id=cam_id)
        return images

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["cameras"] = self._render_cameras()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["cameras"] = self._render_cameras()
        return obs, reward, terminated, truncated, info


def eval_task_steered(
    policy, task_name: str, steering_M, *,
    num_envs: int = 2, max_steps: int = 300, replan_steps: int = 10,
    seed: int = 69_420, width: int = 224, height: int = 224,
) -> float:
    """Run one MetaWorld task with steering and return success rate."""
    prompt = TASK_TO_PROMPT[task_name]

    env_fns = [
        lambda i=i: MultiCameraWrapper(
            gym.make("Meta-World/MT1", env_name=task_name, seed=seed + i, width=width, height=height),
            POLICY_CAMERAS,
        )
        for i in range(num_envs)
    ]
    env = gym.vector.AsyncVectorEnv(env_fns, context="spawn")

    try:
        obs, info = env.reset(seed=seed)
        camera_views = info["cameras"]
        success = np.zeros(num_envs, dtype=bool)
        action_plan = collections.deque()

        for step in range(max_steps):
            if not action_plan:
                obs_dict = {
                    "observation/image": camera_views["corner4"],
                    "observation/wrist_image": camera_views["gripperPOV"],
                    "observation/state": obs.astype(np.float32)[..., :4],
                    "prompt": [prompt] * num_envs,
                }

                if steering_M is not None:
                    result, _ = policy.infer_with_steering(obs_dict, steering_M=steering_M)
                else:
                    result = policy.infer(obs_dict)
                action_chunk = np.clip(result["actions"], -1.0, 1.0).astype(np.float32)
                for t in range(replan_steps):
                    action_plan.append(action_chunk[:, t, :])

            action = action_plan.popleft()
            obs, reward, terminated, truncated, info = env.step(action)
            camera_views = info["cameras"]
            step_success = np.asarray(info.get("success", np.zeros(num_envs)), dtype=bool)
            success |= step_success
            if success.all():
                break

        return float(success.mean())
    finally:
        env.close()


# ── CLI ──────────────────────────────────────────────────────────────────


def baseline_from_activations(task: str, activations_dir: str) -> Optional[float]:
    """Read episode metadata to get baseline success rate without running inference."""
    act_dir = pathlib.Path(activations_dir)
    task_dir = act_dir / task
    if not task_dir.exists():
        return None
    n_success, n_total = 0, 0
    for ep_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        n_total += 1
        if meta.get("episode_success", False):
            n_success += 1
    if n_total == 0:
        return None
    return n_success / n_total


@dataclasses.dataclass
class Args:
    task: str = "assembly-v3"

    config: str = "pi0_fast_metaworld"
    checkpoint_dir: str = "checkpoints/pi0_fast_metaworld/pi0_fast_metaworld_b200_bs512/2500/"

    conceptor_npz: str = ""
    """Path to conceptor .npz. Defaults to $OPENPI_DATA_HOME/pi0fast_metaworld_conceptors.npz."""

    activations_dir: str = ""
    """Path to activations/{ckpt_step}/ for baseline SR. Defaults to
    $OPENPI_DATA_HOME/pi0fast-metaworld-activations-v1-ml45train-16env/2500."""

    global_alphas: List[float] = dataclasses.field(default_factory=lambda: [1.0, 2.0, 10.0])
    per_step_combined_alphas: List[float] = dataclasses.field(default_factory=lambda: [0.1, 0.5, 1.0])
    positive_only_alphas: List[float] = dataclasses.field(default_factory=lambda: [0.5, 1.0, 2.0])
    betas: List[float] = dataclasses.field(default_factory=lambda: [0.1, 0.2, 0.3])
    max_decoding_steps: int = 256

    num_envs: int = 2
    max_steps: int = 300
    replan_steps: int = 10
    seed: int = 69_420

    output_dir: str = "experiments/pi0_fast_metaworld/steering_results"


def main(args: Args) -> None:
    task = args.task
    if task not in TASK_TO_PROMPT:
        raise ValueError(f"Unknown task: {task}")

    npz_path = args.conceptor_npz or str(pathlib.Path(OPENPI_DATA_HOME) / "pi0fast_metaworld_conceptors.npz")
    act_dir = args.activations_dir or str(
        pathlib.Path(OPENPI_DATA_HOME) / "pi0fast-metaworld-activations-v1-ml45train-16env" / "2500"
    )

    task_output_dir = pathlib.Path(args.output_dir) / task
    task_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Task: %s", task)
    logger.info("Output: %s", task_output_dir)

    with open(task_output_dir / "sweep_args.json", "w") as f:
        json.dump(dataclasses.asdict(args), f, indent=2, default=str)

    logger.info("Loading pi0-fast policy...")
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    logger.info("Policy loaded (JAX).")

    npz = load_npz(npz_path)

    all_results: List[dict] = []
    summary_path = task_output_dir / "summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            prev = json.load(f)
        all_results = prev.get("conditions", [])
        done = {r["condition"] for r in all_results}
        logger.info("Resuming with %d previously completed conditions", len(done))
    else:
        done = set()

    def save_progress():
        sorted_results = sorted(all_results, key=lambda x: x.get("success_rate", 0), reverse=True)
        with open(summary_path, "w") as f:
            json.dump({"task": task, "conditions": sorted_results}, f, indent=2)

    def run_condition(condition_name: str, M) -> dict:
        sr = eval_task_steered(
            policy, task, M,
            num_envs=args.num_envs, max_steps=args.max_steps,
            replan_steps=args.replan_steps, seed=args.seed,
        )
        logger.info("  %s: SR=%.3f", condition_name, sr)
        return {"condition": condition_name, "success_rate": sr}

    # ── 1. Baseline (from activation data — no GPU inference) ──
    if "baseline" not in done:
        logger.info("\n[1] Baseline (from activation metadata)...")
        baseline_sr = baseline_from_activations(task, act_dir)
        if baseline_sr is not None:
            logger.info("  baseline: SR=%.3f (from activation data)", baseline_sr)
            all_results.append({"condition": "baseline", "success_rate": baseline_sr})
        else:
            logger.warning("  No activation data for baseline — running inference.")
            all_results.append(run_condition("baseline", None))
        save_progress()

    # ── 2. Global (contrastive) ──
    logger.info("\n[2] Global contrastive conditions...")
    for alpha in args.global_alphas:
        try:
            C = get_conceptor(npz, task, "global", alpha, "C_contrastive")
        except KeyError:
            logger.warning("No global contrastive for %s/a=%s — skipping.", task, alpha)
            continue
        for beta in args.betas:
            cond_name = f"global_a{alpha}_b{beta}"
            if cond_name in done:
                continue
            all_results.append(run_condition(cond_name, build_M(C, beta)))
            save_progress()

    # ── 3. Per-step combined (contrastive, first/mid/last by position) ──
    logger.info("\n[3] Per-step combined conditions...")
    for alpha in args.per_step_combined_alphas:
        for beta in args.betas:
            cond_name = f"per_step_combined_a{alpha}_b{beta}"
            if cond_name in done:
                continue
            try:
                M_stack = build_M_per_step_combined(
                    npz, task, alpha, beta, args.max_decoding_steps, kind="C_contrastive",
                )
            except KeyError as e:
                logger.warning("Missing per_token conceptor for combined: %s — skipping.", e)
                continue
            all_results.append(run_condition(cond_name, M_stack))
            save_progress()

    # ── 4. Positive-only (C_success, global) ──
    logger.info("\n[4] Positive-only conditions...")
    for alpha in args.positive_only_alphas:
        try:
            C = get_conceptor(npz, task, "global", alpha, "C_success")
        except KeyError:
            logger.warning("No C_success for %s/a=%s — skipping.", task, alpha)
            continue
        for beta in args.betas:
            cond_name = f"positive_only_a{alpha}_b{beta}"
            if cond_name in done:
                continue
            all_results.append(run_condition(cond_name, build_M(C, beta)))
            save_progress()

    save_progress()
    logger.info("\n%s", "=" * 70)
    logger.info("%-45s %8s", "Condition", "SR")
    for r in sorted(all_results, key=lambda x: x.get("success_rate", 0), reverse=True):
        logger.info("%-45s %8.3f", r["condition"], r["success_rate"])
    logger.info("Saved to %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main(tyro.cli(Args))
