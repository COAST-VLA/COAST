"""
Conceptor-Based Steering Experiment for π₀.₅

Implements two steering strategies:
  Strategy 3: Global contrastive conceptor C_steer = C_success ∧ ¬C_failure
              applied as soft multiplicative gate at layer 11 across all denoising steps.
  Strategy 5: Per-denoising-step conceptors — same as Strategy 3 but with separate
              conceptor per denoising step t ∈ {0, ..., 9}.

Baselines:
  - No steering (vanilla model)
  - Linear additive steering (mean-diff direction)
  - Random conceptor control

Usage:
    MUJOCO_GL=egl uv run experiments/conceptor_steering.py \
        --policy.config=pi05_metaworld \
        --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
        --tasks assembly-v3 \
        --num_envs 15 \
        --output_dir experiments/steering_results
"""

import collections
import dataclasses
import json
import logging
import os
import pathlib
from typing import Literal

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metaworld  # noqa: F401
import numpy as np
from huggingface_hub import hf_hub_download
from scipy import stats
from tqdm import tqdm
import torch
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

REPO_ID = "brandonyang/ml45-activations-15"
HF_CACHE = "/nlp/data/huggingface_cache"

CAMERA_IDS = {
    "topview": 0, "corner": 1, "corner2": 2, "corner3": 3,
    "corner4": 4, "behindGripper": 5, "gripperPOV": 6,
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

# Layer indices in the captured data: [0, 5, 11, 17]
# layer_idx=2 in the npz corresponds to model layer 11
LAYER_MAP = {0: 0, 5: 1, 11: 2, 17: 3}

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Compute Conceptors from HF Activation Data
# ──────────────────────────────────────────────────────────────────────────────


def find_mixed_outcome_tasks(tasks):
    """Find tasks that have both successful and failed episodes in the 15-env dataset."""
    mixed_tasks = {}
    for task in tasks:
        success_envs = []
        failure_envs = []
        for env_idx in range(15):
            env_name = f"env_{env_idx:03d}"
            try:
                meta_path = hf_hub_download(
                    REPO_ID,
                    f"ckpt_5000/{task}/{env_name}/metadata.json",
                    cache_dir=HF_CACHE,
                )
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta["episode_success"]:
                    success_envs.append(env_name)
                else:
                    failure_envs.append(env_name)
            except Exception:
                continue
        if success_envs and failure_envs:
            mixed_tasks[task] = {"success": success_envs, "failure": failure_envs}
            logger.info(
                f"  {task}: {len(success_envs)} success, {len(failure_envs)} failure envs"
            )
        elif success_envs:
            logger.info(f"  {task}: ALL {len(success_envs)} succeed (no failures)")
        elif failure_envs:
            logger.info(f"  {task}: ALL {len(failure_envs)} fail (no successes)")
    return mixed_tasks


def load_activations_for_episode(task, env_name, layer_idx=2):
    """
    Load residual stream activations for one episode from HuggingFace.

    Args:
        task: e.g., "assembly-v3"
        env_name: e.g., "env_000"
        layer_idx: index into the 4 captured layers [0,5,11,17].
                   layer_idx=2 corresponds to model layer 11.

    Returns:
        all_activations: dict mapping denoising_step -> np.array (n_inference, 32, 1024)
        metadata: episode metadata dict
    """
    meta_path = hf_hub_download(
        REPO_ID,
        f"ckpt_5000/{task}/{env_name}/metadata.json",
        cache_dir=HF_CACHE,
    )
    with open(meta_path) as f:
        meta = json.load(f)

    n_inference = meta["total_inference_steps"]

    all_activations = {t: [] for t in range(10)}

    for step in range(n_inference):
        step_name = f"step_{step:04d}"
        res_path = hf_hub_download(
            REPO_ID,
            f"ckpt_5000/{task}/{env_name}/{step_name}/suffix_residual.npz",
            cache_dir=HF_CACHE,
        )
        data = np.load(res_path)
        residual = data["all_suffix_residual"]  # (10, 4, 32, 1024)

        for t in range(10):
            act = residual[t, layer_idx, :, :]  # (32, 1024)
            all_activations[t].append(act)

    for t in range(10):
        all_activations[t] = np.stack(all_activations[t])  # (n_inference, 32, 1024)

    return all_activations, meta


def collect_outcome_activations(task, env_splits, layer_idx=2):
    """
    Collect activations grouped by success/failure.

    Returns:
        success_acts: dict mapping denoising_step -> np.array (N_success, 1024)
        failure_acts: dict mapping denoising_step -> np.array (N_failure, 1024)
    """
    success_acts = {t: [] for t in range(10)}
    failure_acts = {t: [] for t in range(10)}

    for outcome, env_list in [("success", env_splits["success"]),
                               ("failure", env_splits["failure"])]:
        target = success_acts if outcome == "success" else failure_acts
        for env_name in tqdm(env_list, desc=f"Loading {outcome} envs", leave=False):
            acts, meta = load_activations_for_episode(task, env_name, layer_idx)
            for t in range(10):
                # Flatten across inference steps and action tokens
                flat = acts[t].reshape(-1, 1024)
                target[t].append(flat)

    for t in range(10):
        success_acts[t] = np.concatenate(success_acts[t], axis=0)
        failure_acts[t] = np.concatenate(failure_acts[t], axis=0)

    logger.info(
        f"  Collected: {success_acts[0].shape[0]} success samples, "
        f"{failure_acts[0].shape[0]} failure samples per denoise step"
    )
    return success_acts, failure_acts


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor Math
# ──────────────────────────────────────────────────────────────────────────────


def compute_conceptor(X, alpha=1.0):
    """Compute conceptor C = R @ (R + alpha^{-2} I)^{-1} from data X."""
    d = X.shape[1]
    R = (X.T @ X) / X.shape[0]
    reg = (alpha ** -2) * np.eye(d)
    C = R @ np.linalg.inv(R + reg)
    eigenvalues = np.linalg.eigvalsh(C)[::-1]
    return C, eigenvalues


def boolean_and(C_A, C_B):
    """Soft intersection: C_{A ∧ B} = C_A @ inv(C_A + C_B - C_A @ C_B) @ C_B."""
    d = C_A.shape[0]
    inner = C_A + C_B - C_A @ C_B + 1e-8 * np.eye(d)
    return C_A @ np.linalg.inv(inner) @ C_B


def boolean_not(C):
    """Soft complement: I - C."""
    return np.eye(C.shape[0]) - C


def contrastive_conceptor(C_positive, C_negative):
    """C_positive AND (NOT C_negative): subspace used by positive but not negative."""
    return boolean_and(C_positive, boolean_not(C_negative))


def build_steering_conceptors(task, env_splits, alpha=0.5, layer_idx=2):
    """
    Build conceptors for steering.

    Returns:
        global_conceptor: (1024, 1024) — Strategy 3
        step_conceptors: dict {t: (1024, 1024)} — Strategy 5
        diagnostics: dict with quotas, spectra
    """
    logger.info(f"Collecting activations for {task}...")
    success_acts, failure_acts = collect_outcome_activations(task, env_splits, layer_idx)

    diagnostics = {"quotas": {}, "spectra": {}}

    # Strategy 5: Per-denoising-step conceptors
    step_conceptors = {}
    for t in range(10):
        C_succ, _ = compute_conceptor(success_acts[t], alpha)
        C_fail, _ = compute_conceptor(failure_acts[t], alpha)
        C_steer = contrastive_conceptor(C_succ, C_fail)
        step_conceptors[t] = C_steer

        evals_steer = np.linalg.eigvalsh(C_steer)[::-1]
        diagnostics["quotas"][t] = float(np.trace(C_steer))
        diagnostics["spectra"][t] = evals_steer

    # Strategy 3: Global conceptor (pool across all denoising steps)
    all_success = np.concatenate([success_acts[t] for t in range(10)], axis=0)
    all_failure = np.concatenate([failure_acts[t] for t in range(10)], axis=0)

    C_succ_global, _ = compute_conceptor(all_success, alpha)
    C_fail_global, _ = compute_conceptor(all_failure, alpha)
    global_conceptor = contrastive_conceptor(C_succ_global, C_fail_global)

    evals_global = np.linalg.eigvalsh(global_conceptor)[::-1]
    diagnostics["quotas"]["global"] = float(np.trace(global_conceptor))
    diagnostics["spectra"]["global"] = evals_global

    logger.info(
        f"  Global conceptor quota: {diagnostics['quotas']['global']:.1f}, "
        f"per-step range: [{min(diagnostics['quotas'][t] for t in range(10)):.1f}, "
        f"{max(diagnostics['quotas'][t] for t in range(10)):.1f}]"
    )

    return global_conceptor, step_conceptors, diagnostics, success_acts, failure_acts


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Steering Hooks
# ──────────────────────────────────────────────────────────────────────────────


class ConceptorSteeringHook:
    """PyTorch forward hook that applies conceptor steering to the residual stream."""

    def __init__(self, strategy, global_conceptor=None, step_conceptors=None,
                 beta=0.3, device="cuda"):
        self.strategy = strategy
        self.beta = beta
        self.current_denoise_step = 0

        d = 1024
        I = torch.eye(d, dtype=torch.float32).to(device)

        if strategy == "global" and global_conceptor is not None:
            C = torch.tensor(global_conceptor, dtype=torch.float32).to(device)
            self.M_global = (1 - beta) * I + beta * C

        if strategy == "per_step" and step_conceptors is not None:
            self.M_steps = {}
            for t, C_np in step_conceptors.items():
                C = torch.tensor(C_np, dtype=torch.float32).to(device)
                self.M_steps[t] = (1 - beta) * I + beta * C

        self.intervention_norms = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        t = self.current_denoise_step

        if self.strategy == "global":
            M = self.M_global
        elif self.strategy == "per_step":
            M = self.M_steps.get(t, None)
            if M is None:
                return output
        else:
            return output

        M = M.to(dtype=h.dtype)
        h_steered = torch.matmul(h, M.T)

        delta = h_steered - h
        norm = torch.norm(delta).item()
        self.intervention_norms.append(norm)

        if rest is not None:
            return (h_steered,) + rest
        return h_steered

    def set_denoise_step(self, t):
        self.current_denoise_step = t

    def reset_logs(self):
        self.intervention_norms = []


class LinearSteeringHook:
    """Additive steering hook: h' = h + alpha * direction."""

    def __init__(self, steering_vector, alpha=1.0, device="cuda"):
        self.direction = torch.tensor(
            steering_vector, dtype=torch.float32
        ).to(device)
        self.alpha = alpha
        self.current_denoise_step = 0
        self.intervention_norms = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None

        d = self.direction.to(dtype=h.dtype)
        delta = self.alpha * d.unsqueeze(0).unsqueeze(0)
        h_steered = h + delta
        self.intervention_norms.append(torch.norm(delta).item())

        if rest is not None:
            return (h_steered,) + rest
        return h_steered

    def set_denoise_step(self, t):
        self.current_denoise_step = t

    def reset_logs(self):
        self.intervention_norms = []


def compute_linear_steering_vector(success_acts, failure_acts):
    """Mean-difference steering direction (unit vector)."""
    mu_s = np.mean(success_acts, axis=0)
    mu_f = np.mean(failure_acts, axis=0)
    delta = mu_s - mu_f
    return delta / (np.linalg.norm(delta) + 1e-8)


def compute_random_conceptor(d=1024, quota_target=10, alpha=0.5):
    """Generate a random conceptor with similar quota for control."""
    Q, _ = np.linalg.qr(np.random.randn(d, d))
    raw = np.random.exponential(scale=1.0, size=d)
    raw = np.sort(raw)[::-1]
    eigs = raw / (raw + alpha ** -2)
    C = Q @ np.diag(eigs) @ Q.T
    return C


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Environment and Rollout Infrastructure
# ──────────────────────────────────────────────────────────────────────────────


class MultiCameraWrapper(gym.Wrapper):
    """Wrapper that renders multiple cameras and includes images in info dict."""

    def __init__(self, env, camera_names):
        super().__init__(env)
        self.camera_names = camera_names

    def _render_cameras(self):
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


def make_env(env_name, num_envs, seed, width=224, height=224,
             camera_names=("corner", "corner4", "gripperPOV")):
    env_fns = [
        lambda i=i: MultiCameraWrapper(
            gym.make("Meta-World/MT1", env_name=env_name,
                     seed=seed + i, width=width, height=height),
            list(camera_names),
        )
        for i in range(num_envs)
    ]
    return gym.vector.AsyncVectorEnv(env_fns, context="spawn")


def run_episode(policy, env, task_name, num_envs, max_steps=300,
                replan_steps=10, steering_hooks=None):
    """
    Run a single episode (across num_envs parallel envs) with optional steering.

    Returns list of per-env result dicts.
    """
    prompt = TASK_TO_PROMPT[task_name]
    obs, info = env.reset()
    camera_views = info["cameras"]
    success = np.zeros(num_envs, dtype=bool)
    cumulative_reward = np.zeros(num_envs)
    steps_to_success = np.full(num_envs, -1, dtype=int)
    action_plan = collections.deque()
    all_actions = [[] for _ in range(num_envs)]
    all_intervention_norms = []

    for step in range(max_steps):
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
                result, diag = policy.infer_with_steering(
                    obs_dict, steering_hooks=steering_hooks
                )
                # Collect intervention norms from all hooks
                for _, hook_fn in steering_hooks:
                    all_intervention_norms.extend(hook_fn.intervention_norms)
            else:
                result = policy.infer(obs_dict)

            action_chunk = np.clip(result["actions"], -1.0, 1.0).astype(np.float32)
            for t in range(min(replan_steps, action_chunk.shape[1])):
                action_plan.append(action_chunk[:, t, :])

        action = action_plan.popleft()
        for env_id in range(num_envs):
            all_actions[env_id].append(action[env_id].copy())

        obs, reward, terminated, truncated, info = env.step(action)
        camera_views = info["cameras"]
        cumulative_reward += reward
        step_success = np.asarray(info.get("success", np.zeros(num_envs)), dtype=bool)
        for env_id in range(num_envs):
            if step_success[env_id] and steps_to_success[env_id] == -1:
                steps_to_success[env_id] = step
        success |= step_success
        if success.all():
            break

    results = []
    for env_id in range(num_envs):
        results.append({
            "env_id": env_id,
            "success": bool(success[env_id]),
            "total_reward": float(cumulative_reward[env_id]),
            "steps_to_success": int(steps_to_success[env_id]),
            "trajectory": np.array(all_actions[env_id]),
            "intervention_norms": all_intervention_norms,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Metrics
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(baseline_results, steered_results):
    """Compute comparison metrics between baseline and steered conditions."""
    baseline_sr = np.mean([r["success"] for r in baseline_results])
    steered_sr = np.mean([r["success"] for r in steered_results])

    baseline_rewards = [r["total_reward"] for r in baseline_results]
    steered_rewards = [r["total_reward"] for r in steered_results]

    metrics = {
        "baseline_success_rate": float(baseline_sr),
        "steered_success_rate": float(steered_sr),
        "success_rate_delta": float(steered_sr - baseline_sr),
        "baseline_reward_mean": float(np.mean(baseline_rewards)),
        "baseline_reward_std": float(np.std(baseline_rewards)),
        "steered_reward_mean": float(np.mean(steered_rewards)),
        "steered_reward_std": float(np.std(steered_rewards)),
    }

    # Intervention norms
    all_norms = []
    for r in steered_results:
        all_norms.extend(r["intervention_norms"])
    if all_norms:
        metrics["mean_intervention_norm"] = float(np.mean(all_norms))
        metrics["max_intervention_norm"] = float(np.max(all_norms))

    # Action magnitude comparison
    baseline_mags = [np.mean(np.linalg.norm(r["trajectory"], axis=-1)) for r in baseline_results]
    steered_mags = [np.mean(np.linalg.norm(r["trajectory"], axis=-1)) for r in steered_results]
    metrics["baseline_action_magnitude"] = float(np.mean(baseline_mags))
    metrics["steered_action_magnitude"] = float(np.mean(steered_mags))

    # KL divergence (histogram-based per action dim)
    all_baseline_acts = np.concatenate([r["trajectory"] for r in baseline_results], axis=0)
    all_steered_acts = np.concatenate([r["trajectory"] for r in steered_results], axis=0)

    n_bins = 50
    kl_per_dim = []
    for dim in range(min(all_baseline_acts.shape[1], all_steered_acts.shape[1])):
        combined = np.concatenate([all_baseline_acts[:, dim], all_steered_acts[:, dim]])
        edges = np.linspace(combined.min(), combined.max(), n_bins + 1)
        hist_b, _ = np.histogram(all_baseline_acts[:, dim], bins=edges, density=True)
        hist_s, _ = np.histogram(all_steered_acts[:, dim], bins=edges, density=True)
        eps = 1e-10
        hist_b = hist_b + eps
        hist_s = hist_s + eps
        hist_b = hist_b / hist_b.sum()
        hist_s = hist_s / hist_s.sum()
        kl = float(np.sum(hist_s * np.log(hist_s / hist_b)))
        kl_per_dim.append(kl)
    metrics["kl_divergence_mean"] = float(np.mean(kl_per_dim))

    return metrics


def statistical_tests(baseline_results, steered_results):
    """Fisher's exact test for success rate, Mann-Whitney for reward, bootstrap CI."""
    b_success = sum(r["success"] for r in baseline_results)
    b_fail = len(baseline_results) - b_success
    s_success = sum(r["success"] for r in steered_results)
    s_fail = len(steered_results) - s_success

    _, fisher_p = stats.fisher_exact([[b_success, b_fail], [s_success, s_fail]])

    b_rewards = [r["total_reward"] for r in baseline_results]
    s_rewards = [r["total_reward"] for r in steered_results]
    mwu_stat, mwu_p = stats.mannwhitneyu(b_rewards, s_rewards, alternative="two-sided")

    # Bootstrap CI for success rate difference
    n_boot = 1000
    rng = np.random.default_rng(42)
    diffs = []
    b_arr = np.array([r["success"] for r in baseline_results], dtype=float)
    s_arr = np.array([r["success"] for r in steered_results], dtype=float)
    for _ in range(n_boot):
        b_sample = rng.choice(b_arr, size=len(b_arr), replace=True)
        s_sample = rng.choice(s_arr, size=len(s_arr), replace=True)
        diffs.append(np.mean(s_sample) - np.mean(b_sample))

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    return {
        "fisher_p": float(fisher_p),
        "mannwhitney_p": float(mwu_p),
        "success_diff_95ci": [float(ci_low), float(ci_high)],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: Plotting
# ──────────────────────────────────────────────────────────────────────────────


def plot_success_vs_beta(all_metrics, task, output_dir):
    """Success rate vs steering strength β, one line per strategy."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Group by strategy prefix
    strategies = {}
    for name, m in all_metrics.items():
        if name == "baseline":
            continue
        parts = name.split("_")
        if parts[0] in ("strategy3", "strategy5"):
            strategy_label = parts[0]
            beta_val = float(parts[-1].replace("b", ""))
            if strategy_label not in strategies:
                strategies[strategy_label] = {"betas": [], "sr": []}
            strategies[strategy_label]["betas"].append(beta_val)
            strategies[strategy_label]["sr"].append(m["steered_success_rate"])

    baseline_sr = all_metrics.get("baseline", {}).get("baseline_success_rate",
                    all_metrics[list(all_metrics.keys())[0]]["baseline_success_rate"])

    ax.axhline(baseline_sr, color="gray", linestyle="--", label="Baseline", linewidth=2)

    colors = {"strategy3": "tab:blue", "strategy5": "tab:orange",
              "linear": "tab:green", "random": "tab:red"}
    labels = {"strategy3": "Strategy 3 (Global)", "strategy5": "Strategy 5 (Per-step)",
              "linear": "Linear", "random": "Random Conceptor"}

    for strat, data in strategies.items():
        idx = np.argsort(data["betas"])
        betas = np.array(data["betas"])[idx]
        sr = np.array(data["sr"])[idx]
        ax.plot(betas, sr, "o-", color=colors.get(strat, "tab:purple"),
                label=labels.get(strat, strat), linewidth=2, markersize=8)

    # Add linear and random as single points
    for name, m in all_metrics.items():
        if name.startswith("linear_"):
            alpha_val = name.split("_a")[-1]
            ax.scatter(0.3, m["steered_success_rate"], marker="^", s=100,
                      color=colors["linear"], zorder=5,
                      label=f"Linear (α={alpha_val})")
        elif name == "random_conceptor":
            ax.scatter(0.3, m["steered_success_rate"], marker="x", s=100,
                      color=colors["random"], zorder=5, label="Random Conceptor")

    ax.set_xlabel("Steering Strength β", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(f"Conceptor Steering: {task}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / f"success_vs_beta_{task}.png", dpi=150)
    plt.savefig(output_dir / f"success_vs_beta_{task}.pdf")
    plt.close()


def plot_intervention_vs_success(all_metrics, task, output_dir):
    """Scatter: intervention norm vs success rate (efficiency plot)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for name, m in all_metrics.items():
        if name == "baseline" or "mean_intervention_norm" not in m:
            continue
        ax.scatter(m["mean_intervention_norm"], m["steered_success_rate"],
                  s=80, label=name, alpha=0.8)
        ax.annotate(name, (m["mean_intervention_norm"], m["steered_success_rate"]),
                   fontsize=7, ha="left", va="bottom")

    ax.set_xlabel("Mean Intervention Norm", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(f"Efficiency: {task}", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"intervention_efficiency_{task}.png", dpi=150)
    plt.close()


def plot_per_step_norms(hook, task, condition_name, output_dir):
    """Bar chart of intervention norms per denoising step (Strategy 5)."""
    norms = hook.intervention_norms
    if not norms:
        return

    # Group by denoising step
    norms_by_step = {t: [] for t in range(10)}
    for i, n in enumerate(norms):
        t = i % 10
        norms_by_step[t].append(n)

    means = [np.mean(norms_by_step[t]) for t in range(10)]
    stds = [np.std(norms_by_step[t]) for t in range(10)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(range(10), means, yerr=stds, capsize=3, alpha=0.7)
    ax.set_xlabel("Denoising Step", fontsize=12)
    ax.set_ylabel("Intervention Norm", fontsize=12)
    ax.set_title(f"Per-Step Intervention: {task} ({condition_name})", fontsize=13)
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / f"per_step_norms_{task}_{condition_name}.png", dpi=150)
    plt.close()


def plot_conceptor_spectra(diagnostics, task, output_dir):
    """Plot eigenvalue spectra of the contrastive conceptors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Global spectrum
    ax = axes[0]
    evals = diagnostics["spectra"]["global"]
    ax.plot(evals[:100], "o-", markersize=3)
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Global C_steer Spectrum\n(quota={diagnostics['quotas']['global']:.1f})")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Per-step spectra
    ax = axes[1]
    for t in [0, 4, 9]:
        evals = diagnostics["spectra"][t]
        ax.plot(evals[:100], "o-", markersize=3,
                label=f"step {t} (q={diagnostics['quotas'][t]:.1f})")
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Per-Step C_steer Spectra")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Contrastive Conceptor Spectra: {task}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"conceptor_spectra_{task}.png", dpi=150)
    plt.close()


def generate_summary_table(all_metrics, task, output_dir, stat_tests=None):
    """Generate and save summary table as CSV and log it."""
    rows = []
    for name, m in all_metrics.items():
        row = {
            "condition": name,
            "success_rate": m.get("steered_success_rate", m.get("baseline_success_rate", "")),
            "reward_mean": m.get("steered_reward_mean", m.get("baseline_reward_mean", "")),
            "reward_std": m.get("steered_reward_std", m.get("baseline_reward_std", "")),
            "intervention_norm": m.get("mean_intervention_norm", ""),
            "kl_divergence": m.get("kl_divergence_mean", ""),
            "action_magnitude": m.get("steered_action_magnitude", m.get("baseline_action_magnitude", "")),
        }
        if stat_tests and name in stat_tests:
            row["fisher_p"] = stat_tests[name]["fisher_p"]
            row["mannwhitney_p"] = stat_tests[name]["mannwhitney_p"]
            row["ci_low"] = stat_tests[name]["success_diff_95ci"][0]
            row["ci_high"] = stat_tests[name]["success_diff_95ci"][1]
        rows.append(row)

    import csv
    csv_path = output_dir / f"results_{task}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Also log
    logger.info(f"\n{'='*80}")
    logger.info(f"Results for {task}:")
    logger.info(f"{'Condition':<35s} {'SR':>6s} {'Reward':>12s} {'IntNorm':>10s} {'KL':>8s}")
    logger.info(f"{'-'*75}")
    for row in rows:
        sr = f"{row['success_rate']:.3f}" if isinstance(row['success_rate'], float) else str(row['success_rate'])
        rew = f"{row['reward_mean']:.1f}±{row['reward_std']:.1f}" if isinstance(row['reward_mean'], float) else ""
        inorm = f"{row['intervention_norm']:.2f}" if isinstance(row['intervention_norm'], float) else "-"
        kl = f"{row['kl_divergence']:.4f}" if isinstance(row['kl_divergence'], float) else "-"
        logger.info(f"{row['condition']:<35s} {sr:>6s} {rew:>12s} {inorm:>10s} {kl:>8s}")
    logger.info(f"{'='*80}\n")

    return csv_path


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6: Full Experiment Runner
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class PolicyArgs:
    config: str = "pi05_metaworld"
    dir: str = "checkpoints/pi05_metaworld/pi05_metaworld_test/5000/"


@dataclasses.dataclass
class Args:
    policy: PolicyArgs = dataclasses.field(default_factory=PolicyArgs)

    # Tasks to steer. Must be tasks with mixed outcomes in 15-env dataset.
    tasks: list[str] = dataclasses.field(default_factory=lambda: ["assembly-v3"])

    # Steering parameters
    alphas: list[float] = dataclasses.field(default_factory=lambda: [0.5])
    betas: list[float] = dataclasses.field(default_factory=lambda: [0.1, 0.3, 0.5])
    steering_layer: int = 11
    linear_alphas: list[float] = dataclasses.field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])

    # Env params
    num_envs: int = 15
    max_steps: int = 300
    replan_steps: int = 10
    width: int = 224
    height: int = 224
    policy_cameras: list[str] = dataclasses.field(default_factory=lambda: ["corner", "corner4", "gripperPOV"])
    seed: int = 69_420

    output_dir: str = "experiments/steering_results"


def run_full_experiment(task, env_splits, policy, args, output_dir, device="cuda"):
    """Run the complete steering experiment for one task."""
    output_dir = pathlib.Path(output_dir) / task
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_metrics = {}
    stat_tests = {}

    layer_idx_in_data = LAYER_MAP[args.steering_layer]
    model_layer = args.steering_layer

    # ── Baseline ──
    logger.info(f"[{task}] Running baseline (no steering)...")
    env = make_env(task, args.num_envs, args.seed, args.width, args.height, args.policy_cameras)
    try:
        baseline = run_episode(
            policy, env, task, args.num_envs, args.max_steps, args.replan_steps,
            steering_hooks=None,
        )
    finally:
        env.close()
    all_results["baseline"] = baseline
    all_metrics["baseline"] = {
        "baseline_success_rate": float(np.mean([r["success"] for r in baseline])),
        "baseline_reward_mean": float(np.mean([r["total_reward"] for r in baseline])),
        "baseline_reward_std": float(np.std([r["total_reward"] for r in baseline])),
        "baseline_action_magnitude": float(np.mean([
            np.mean(np.linalg.norm(r["trajectory"], axis=-1)) for r in baseline
        ])),
    }
    logger.info(
        f"  Baseline: SR={all_metrics['baseline']['baseline_success_rate']:.3f}, "
        f"Reward={all_metrics['baseline']['baseline_reward_mean']:.1f}"
    )

    # ── Compute conceptors ──
    for alpha in args.alphas:
        logger.info(f"[{task}] Computing conceptors (α={alpha}, layer={model_layer})...")
        global_C, step_Cs, diagnostics, success_acts, failure_acts = \
            build_steering_conceptors(task, env_splits, alpha=alpha, layer_idx=layer_idx_in_data)

        plot_conceptor_spectra(diagnostics, task, output_dir)

        # Save diagnostics
        diag_path = output_dir / f"diagnostics_a{alpha}.json"
        diag_save = {
            "quotas": {str(k): v for k, v in diagnostics["quotas"].items()},
            "alpha": alpha,
            "layer": model_layer,
        }
        with open(diag_path, "w") as f:
            json.dump(diag_save, f, indent=2)

        for beta in args.betas:
            # ── Strategy 3: Global conceptor ──
            name = f"strategy3_a{alpha}_b{beta}"
            logger.info(f"[{task}] Running {name}...")
            hook = ConceptorSteeringHook(
                strategy="global", global_conceptor=global_C, beta=beta, device=device,
            )
            env = make_env(task, args.num_envs, args.seed, args.width, args.height, args.policy_cameras)
            try:
                results = run_episode(
                    policy, env, task, args.num_envs, args.max_steps, args.replan_steps,
                    steering_hooks=[(model_layer, hook)],
                )
            finally:
                env.close()
            all_results[name] = results
            all_metrics[name] = compute_metrics(baseline, results)
            stat_tests[name] = statistical_tests(baseline, results)
            logger.info(
                f"  {name}: SR={all_metrics[name]['steered_success_rate']:.3f} "
                f"(Δ={all_metrics[name]['success_rate_delta']:+.3f}), "
                f"IntNorm={all_metrics[name].get('mean_intervention_norm', 0):.2f}"
            )

            # ── Strategy 5: Per-step conceptors ──
            name = f"strategy5_a{alpha}_b{beta}"
            logger.info(f"[{task}] Running {name}...")
            hook = ConceptorSteeringHook(
                strategy="per_step", step_conceptors=step_Cs, beta=beta, device=device,
            )
            env = make_env(task, args.num_envs, args.seed, args.width, args.height, args.policy_cameras)
            try:
                results = run_episode(
                    policy, env, task, args.num_envs, args.max_steps, args.replan_steps,
                    steering_hooks=[(model_layer, hook)],
                )
            finally:
                env.close()
            all_results[name] = results
            all_metrics[name] = compute_metrics(baseline, results)
            stat_tests[name] = statistical_tests(baseline, results)
            logger.info(
                f"  {name}: SR={all_metrics[name]['steered_success_rate']:.3f} "
                f"(Δ={all_metrics[name]['success_rate_delta']:+.3f}), "
                f"IntNorm={all_metrics[name].get('mean_intervention_norm', 0):.2f}"
            )
            plot_per_step_norms(hook, task, name, output_dir)

    # ── Linear steering baseline ──
    logger.info(f"[{task}] Computing linear steering vector...")
    # Use pooled activations across denoising steps for mean-diff
    pooled_s = np.concatenate([success_acts[t] for t in range(10)], axis=0)
    pooled_f = np.concatenate([failure_acts[t] for t in range(10)], axis=0)
    steer_vec = compute_linear_steering_vector(pooled_s, pooled_f)

    for alpha_lin in args.linear_alphas:
        name = f"linear_a{alpha_lin}"
        logger.info(f"[{task}] Running {name}...")
        hook = LinearSteeringHook(steer_vec, alpha=alpha_lin, device=device)
        env = make_env(task, args.num_envs, args.seed, args.width, args.height, args.policy_cameras)
        try:
            results = run_episode(
                policy, env, task, args.num_envs, args.max_steps, args.replan_steps,
                steering_hooks=[(model_layer, hook)],
            )
        finally:
            env.close()
        all_results[name] = results
        all_metrics[name] = compute_metrics(baseline, results)
        stat_tests[name] = statistical_tests(baseline, results)
        logger.info(
            f"  {name}: SR={all_metrics[name]['steered_success_rate']:.3f} "
            f"(Δ={all_metrics[name]['success_rate_delta']:+.3f})"
        )

    # ── Random conceptor control ──
    logger.info(f"[{task}] Running random conceptor control...")
    target_quota = diagnostics["quotas"]["global"]
    random_C = compute_random_conceptor(quota_target=target_quota)
    hook = ConceptorSteeringHook(
        strategy="global", global_conceptor=random_C, beta=0.3, device=device,
    )
    env = make_env(task, args.num_envs, args.seed, args.width, args.height, args.policy_cameras)
    try:
        results = run_episode(
            policy, env, task, args.num_envs, args.max_steps, args.replan_steps,
            steering_hooks=[(model_layer, hook)],
        )
    finally:
        env.close()
    all_results["random_conceptor"] = results
    all_metrics["random_conceptor"] = compute_metrics(baseline, results)
    stat_tests["random_conceptor"] = statistical_tests(baseline, results)
    logger.info(
        f"  random_conceptor: SR={all_metrics['random_conceptor']['steered_success_rate']:.3f} "
        f"(Δ={all_metrics['random_conceptor']['success_rate_delta']:+.3f})"
    )

    # ── Generate plots and summary ──
    plot_success_vs_beta(all_metrics, task, output_dir)
    plot_intervention_vs_success(all_metrics, task, output_dir)
    csv_path = generate_summary_table(all_metrics, task, output_dir, stat_tests)

    # Save full results JSON (without trajectories for space)
    results_json = {}
    for name, res_list in all_results.items():
        results_json[name] = [
            {k: v for k, v in r.items() if k != "trajectory"}
            for r in res_list
        ]
    with open(output_dir / f"full_results_{task}.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    # Save stat tests
    with open(output_dir / f"stat_tests_{task}.json", "w") as f:
        json.dump(stat_tests, f, indent=2)

    logger.info(f"[{task}] All results saved to {output_dir}")
    return all_results, all_metrics


def main(args: Args):
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(dataclasses.asdict(args), f, indent=2)

    # Discover mixed-outcome tasks
    logger.info("Discovering mixed-outcome tasks in 15-env dataset...")
    mixed_tasks = find_mixed_outcome_tasks(args.tasks)
    if not mixed_tasks:
        logger.error("No mixed-outcome tasks found! Cannot perform contrastive steering.")
        return

    logger.info(f"Found {len(mixed_tasks)} mixed-outcome tasks: {list(mixed_tasks.keys())}")

    # Load policy
    logger.info("Loading policy...")
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    train_config = _config.get_config(args.policy.config)
    ensure_pytorch_checkpoint(args.policy.dir, args.policy.config)
    policy = _policy_config.create_trained_policy(train_config, args.policy.dir)
    if not policy._is_pytorch_model:  # noqa: SLF001
        raise RuntimeError("Steering requires a PyTorch checkpoint.")
    device = policy._pytorch_device  # noqa: SLF001
    logger.info(f"Policy loaded on {device}")

    # Run experiments for each task
    for task in args.tasks:
        if task not in mixed_tasks:
            logger.warning(f"Skipping {task}: not a mixed-outcome task")
            continue
        env_splits = mixed_tasks[task]
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting experiment for {task}")
        logger.info(f"  Success envs: {env_splits['success']}")
        logger.info(f"  Failure envs: {env_splits['failure']}")
        logger.info(f"{'='*60}")

        run_full_experiment(task, env_splits, policy, args, output_dir, device=device)

    logger.info("All experiments complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = tyro.cli(Args)
    main(args)
