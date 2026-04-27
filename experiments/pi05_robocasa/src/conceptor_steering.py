#!/usr/bin/env python3
"""
Conceptor-Based Steering for pi0.5 RoboCasa
============================================

Loads pre-computed conceptors from robocasa_conceptors.npz and runs steered
policy evaluation on RoboCasa via the WebSocket server + main.py pattern.

One SLURM job per task.  Each job:
  1. Loads the model ONCE
  2. Runs baseline (no steering)
  3. Sweeps all (layer × alpha × beta × strategy) using that task's conceptors
  4. Runs random-conceptor controls
  5. Saves everything under  output_dir/{task}/

Pre-computed conceptor file:
    $OPENPI_DATA_HOME/robocasa_conceptors.npz
    Key pattern: {task}__L{layer}__{alpha_or_per_step_N}__{C_contrastive|C_success|C_failure}

Usage (from repo root):
    uv run experiments/pi05_robocasa/src/conceptor_steering.py \
        --task CloseFridge
"""

import dataclasses
import json
import logging
import os
import pathlib
import re
import subprocess
import threading
import time
from typing import List

import numpy as np
import torch
import tyro

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

OPENPI_DATA_HOME = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
CONCEPTOR_NPZ = pathlib.Path(OPENPI_DATA_HOME) / "robocasa_conceptors.npz"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]  # openpi-new/

ROBOCASA_TASKS = [
    "CloseFridge",
    "CoffeeSetupMug",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "TurnOnElectricKettle",
]


# ──────────────────────────────────────────────────────────────────────────────
# Load Pre-Computed Conceptors from .npz
# ──────────────────────────────────────────────────────────────────────────────


def load_npz():
    """Load the pre-computed conceptor .npz."""
    if not CONCEPTOR_NPZ.exists():
        raise FileNotFoundError(f"Conceptor file not found: {CONCEPTOR_NPZ}")
    return np.load(CONCEPTOR_NPZ, allow_pickle=True)


def get_global_contrastive(npz, task, layer, alpha):
    key = f"{task}__L{layer}__{alpha}__C_contrastive"
    if key not in npz:
        raise KeyError(f"Key not found in npz: {key}")
    return npz[key]


def get_per_step_contrastive(npz, task, layer, step):
    key = f"{task}__L{layer}__per_step_{step}__C_contrastive"
    if key not in npz:
        raise KeyError(f"Key not found in npz: {key}")
    return npz[key]


def get_per_step_contrastive_all(npz, task, layer):
    """Return all per-denoising-step contrastive conceptors at (task, layer).

    Walks ``per_step_{0..N-1}`` until a key is missing. Order matches the
    denoising-step index, so the list can be indexed directly by
    ``current_denoise_step`` inside the hook.
    """
    out = []
    step = 0
    while True:
        key = f"{task}__L{layer}__per_step_{step}__C_contrastive"
        if key not in npz:
            break
        out.append(npz[key])
        step += 1
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Steering Hook
# ──────────────────────────────────────────────────────────────────────────────


class ConceptorSteeringHook:
    """PyTorch forward hook: h' = (1-beta)*h + beta*(h @ C^T).

    Two modes:
      - Static: ``conceptor_matrix`` is a single (d, d) ndarray; the same C is
        applied at every denoising step.
      - Per-step (true time-varying): ``conceptor_matrix`` is a list/tuple of
        (d, d) ndarrays, one per denoising step. ``infer_with_steering`` calls
        ``set_denoise_step(t)`` each step, and the hook selects the matching
        matrix at runtime. If the index exceeds the list length it clamps.
    """

    def __init__(self, conceptor_matrix, beta=0.3, device="cuda"):
        self.beta = beta
        self.current_denoise_step = 0
        if isinstance(conceptor_matrix, (list, tuple)):
            d = conceptor_matrix[0].shape[0]
            I = torch.eye(d, dtype=torch.float32, device=device)
            self._M_per_step = [
                (1 - beta) * I + beta * torch.tensor(C, dtype=torch.float32, device=device)
                for C in conceptor_matrix
            ]
            self.M = None
        else:
            d = conceptor_matrix.shape[0]
            I = torch.eye(d, dtype=torch.float32, device=device)
            C_t = torch.tensor(conceptor_matrix, dtype=torch.float32, device=device)
            self.M = (1 - beta) * I + beta * C_t
            self._M_per_step = None
        self.intervention_norms = []

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            h, rest = output[0], output[1:]
        else:
            h, rest = output, None
        if self._M_per_step is not None:
            idx = min(self.current_denoise_step, len(self._M_per_step) - 1)
            M = self._M_per_step[idx].to(dtype=h.dtype)
        else:
            M = self.M.to(dtype=h.dtype)
        h_steered = torch.matmul(h, M.T)
        self.intervention_norms.append(torch.norm(h_steered - h).item())
        return (h_steered,) + rest if rest is not None else h_steered

    def set_denoise_step(self, t):
        self.current_denoise_step = t

    def reset_logs(self):
        self.intervention_norms = []


def compute_random_conceptor(d=1024, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    raw = np.sort(rng.exponential(1.0, size=d))[::-1]
    eigs = raw / (raw + alpha ** -2)
    return (Q @ np.diag(eigs) @ Q.T).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Steered Server + RoboCasa Client
# ──────────────────────────────────────────────────────────────────────────────


class SteeredPolicyWrapper:
    """Wraps a Policy to route infer() through infer_with_steering()."""

    def __init__(self, policy, steering_hooks):
        self._policy = policy
        self._steering_hooks = steering_hooks

    def update_hooks(self, steering_hooks):
        self._steering_hooks = steering_hooks

    def infer(self, obs):
        if self._steering_hooks:
            for _, hook_fn in self._steering_hooks:
                hook_fn.reset_logs()
            result, _ = self._policy.infer_with_steering(
                obs, steering_hooks=self._steering_hooks
            )
            return result
        return self._policy.infer(obs)

    @property
    def metadata(self):
        return self._policy.metadata


SUCCESS_RATE_RE = re.compile(r"success_rate=([0-9.]+)")


def run_single_task_eval(task_name, num_episodes, port, output_dir):
    """Launch main.py for a SINGLE task via subprocess. Returns success_rate float."""
    robocasa_env_dir = REPO_ROOT / "examples" / "robocasa_env"
    abs_output_dir = str(pathlib.Path(output_dir).resolve())
    cmd = [
        str(robocasa_env_dir / ".venv" / "bin" / "python"),
        str(robocasa_env_dir / "main.py"),
        "--env_name", task_name,
        "--num_episodes", str(num_episodes),
        "--port", str(port),
        "--output_dir", abs_output_dir,
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    logger.info(f"Eval: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(robocasa_env_dir), env=env,
                          capture_output=True, text=True, timeout=14400)
    log_text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        logger.error(f"Eval failed (rc={proc.returncode}):\n{log_text[-3000:]}")
        return None
    matches = SUCCESS_RATE_RE.findall(log_text)
    if matches:
        return float(matches[-1])
    logger.error(f"No success_rate found in main.py output:\n{log_text[-2000:]}")
    return None


def start_server_background(wrapper, port):
    """Start the WebSocket server in a daemon thread. Returns the thread (for reference only)."""
    from openpi.serving import websocket_policy_server

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapper, host="0.0.0.0", port=port, metadata=wrapper.metadata,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(5)  # wait for bind
    logger.info(f"Server started on port {port} (daemon thread)")
    return t


def run_condition(task_name, port, num_episodes,
                  condition_name, task_output_dir):
    """Run one eval against the already-running server. Returns result dict."""
    cond_dir = task_output_dir / condition_name
    cond_dir.mkdir(parents=True, exist_ok=True)

    sr = run_single_task_eval(task_name, num_episodes, port, str(cond_dir))
    if sr is None:
        sr = float("nan")

    logger.info(f"  {condition_name}: SR={sr:.3f}")
    return {"condition": condition_name, "success_rate": sr}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    """Run all steering conditions for one task. Model loads once."""
    # Which task's conceptors to use for steering
    task: str = ROBOCASA_TASKS[0]

    # Policy
    config: str = "pi05_robocasa"
    checkpoint_dir: str = "/vast/projects/ungar/stellar/miaom/openpi-metaworld/checkpoints/pi05_pretrain_human300/multitask_learning/75000"

    # Sweep axes
    layers: List[int] = dataclasses.field(default_factory=lambda: [5, 11, 17])
    alphas: List[float] = dataclasses.field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 10.0])
    betas: List[float] = dataclasses.field(default_factory=lambda: [0.1, 0.3, 0.5])
    strategies: List[str] = dataclasses.field(default_factory=lambda: ["global", "per_step"])
    """Strategies. Valid values:
      - ``global``     : one C per (alpha, layer), applied at every denoising step.
      - ``per_step``   : true time-varying — different C at every denoising step.
                         Alpha-independent (per-step keys in the npz are
                         alpha-free), so it's iterated over (layer, beta) only.
      - ``per_step_N`` : legacy — static C built at denoising step N, applied
                         uniformly across all steps. Kept for ablation runs.
    """

    # Eval
    num_episodes: int = 15
    port: int = 8000

    # Random-conceptor control count (set to 1 to only sample one baseline)
    n_random_controls: int = -1  # -1 means use full layer×beta grid

    output_dir: str = "experiments/pi05_robocasa/steering_results"


def main(args: Args):
    task = args.task
    task_output_dir = pathlib.Path(args.output_dir) / task
    task_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Task: {task}")
    logger.info(f"Output: {task_output_dir}")
    logger.info(f"Sweep: {len(args.layers)} layers × {len(args.alphas)} alphas × "
                f"{len(args.betas)} betas × {len(args.strategies)} strategies")

    # Save args
    with open(task_output_dir / "sweep_args.json", "w") as f:
        json.dump(dataclasses.asdict(args), f, indent=2, default=str)

    # Load model ONCE
    logger.info("Loading policy (one-time cost)...")
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    ensure_pytorch_checkpoint(args.checkpoint_dir, args.config)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    device = str(policy._pytorch_device)  # noqa: SLF001
    logger.info(f"Policy loaded on {device}")

    # Load conceptors ONCE
    npz = load_npz()

    # Load existing results to append to (don't clobber prior sweeps)
    summary_path = task_output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
        all_results = existing.get("conditions", [])
        existing_conds = {r["condition"] for r in all_results if r is not None}
        logger.info(f"Loaded {len(all_results)} existing conditions from summary.json")
    else:
        all_results = []
        existing_conds = set()

    def save_progress():
        """Incremental save with merge-from-disk to avoid concurrent-write clobber.

        Another job may be writing to the same summary.json in parallel (e.g.
        ps0rand + posonly run simultaneously). Before writing, re-read the file
        and union our results with whatever is on disk — last writer doesn't
        lose the other writer's rows.
        """
        merged = {}
        if summary_path.exists():
            try:
                existing = json.load(open(summary_path))
                for r in existing.get("conditions", []):
                    if r is not None and "condition" in r:
                        merged[r["condition"]] = r
            except (json.JSONDecodeError, OSError):
                pass  # mid-write by the other job; our copy is a superset
        for r in all_results:
            if r is not None and "condition" in r:
                merged[r["condition"]] = r
        sorted_results = sorted(merged.values(),
                                key=lambda x: x.get("success_rate", 0), reverse=True)
        tmp = summary_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"task": task, "conditions": sorted_results}, f, indent=2)
        tmp.replace(summary_path)  # atomic

    # Reusable wrapper — we swap hooks between conditions.
    # Server starts ONCE and stays alive; we just mutate the wrapper's hooks.
    wrapper = SteeredPolicyWrapper(policy, steering_hooks=None)
    start_server_background(wrapper, args.port)

    # ── 1. Baseline ──
    if "baseline" in existing_conds:
        logger.info("\n[1/4] Baseline already present, skipping")
    else:
        logger.info("\n[1/4] Baseline (no steering)...")
        wrapper.update_hooks(None)
        r = run_condition(task, args.port, args.num_episodes,
                          "baseline", task_output_dir)
        all_results.append(r)
        existing_conds.add("baseline")
        save_progress()

    # ── 2. Steered conditions ──
    # ``per_step`` is alpha-free (per-step keys in the npz have no alpha slot),
    # so it doesn't enter the alpha loop — we'd just re-run the same condition
    # |alphas| times with no new information. Run it once per (layer, beta) below.
    n_alpha_strats = len([s for s in args.strategies if s != "per_step"])
    total = (
        len(args.layers) * len(args.alphas) * len(args.betas) * n_alpha_strats
        + (len(args.layers) * len(args.betas) if "per_step" in args.strategies else 0)
    )
    logger.info(f"\n[2/4] Steered conditions ({total} total)...")
    idx = 0
    for layer in args.layers:
        for alpha in args.alphas:
            C_global = None  # lazy-load per layer/alpha only if needed
            for beta in args.betas:
                for strategy in args.strategies:
                    if strategy == "per_step":
                        continue  # handled in the dedicated block below
                    idx += 1
                    cond_name = f"{strategy}_L{layer}_a{alpha}_b{beta}"

                    if cond_name in existing_conds:
                        logger.info(f"  [{idx}/{total}] {cond_name} — already exists, skipping")
                        continue

                    logger.info(f"  [{idx}/{total}] {cond_name}")

                    if strategy == "global":
                        if C_global is None:
                            C_global = get_global_contrastive(npz, task, layer, alpha)
                        C = C_global
                    elif strategy.startswith("per_step_"):
                        # legacy: static C built at denoising step N, applied uniformly.
                        step = int(strategy.split("_")[-1])
                        C = get_per_step_contrastive(npz, task, layer, step)
                    else:
                        logger.warning(f"  Unknown strategy {strategy}, skipping")
                        continue

                    hook = ConceptorSteeringHook(C, beta=beta, device=device)
                    wrapper.update_hooks([(layer, hook)])
                    r = run_condition(task, args.port, args.num_episodes,
                                      cond_name, task_output_dir)
                    all_results.append(r)
                    existing_conds.add(cond_name)
                    save_progress()

    # ── 2b. True time-varying per_step (alpha-independent) ──
    # Loads the full list of per-denoising-step conceptors once per layer and
    # passes them all to the hook. infer_with_steering calls
    # hook.set_denoise_step(t) on every step (see pi0_pytorch.py:944), and the
    # hook indexes its M cache to apply a different conceptor each step.
    if "per_step" in args.strategies:
        logger.info(f"\n[2b/4] per_step (true time-varying: one C per denoising step)")
        for layer in args.layers:
            Cs = get_per_step_contrastive_all(npz, task, layer)
            if not Cs:
                logger.warning(f"  no per_step conceptors at L{layer}, skipping")
                continue
            for beta in args.betas:
                idx += 1
                cond_name = f"per_step_L{layer}_b{beta}"
                if cond_name in existing_conds:
                    logger.info(f"  [{idx}/{total}] {cond_name} — already exists, skipping")
                    continue
                logger.info(f"  [{idx}/{total}] {cond_name}  ({len(Cs)} ds-specific Cs)")
                hook = ConceptorSteeringHook(Cs, beta=beta, device=device)
                wrapper.update_hooks([(layer, hook)])
                r = run_condition(task, args.port, args.num_episodes,
                                  cond_name, task_output_dir)
                all_results.append(r)
                existing_conds.add(cond_name)
                save_progress()

    # ── 3. Random controls ──
    random_pairs = [(layer, beta) for layer in args.layers for beta in args.betas]
    if args.n_random_controls >= 0:
        random_pairs = random_pairs[: args.n_random_controls]
    logger.info(f"\n[3/4] Random controls ({len(random_pairs)})...")
    for layer, beta in random_pairs:
        cond_name = f"random_L{layer}_b{beta}"
        if cond_name in existing_conds:
            logger.info(f"  {cond_name} — already exists, skipping")
            continue
        C_rand = compute_random_conceptor(seed=layer * 100 + int(beta * 10))
        hook = ConceptorSteeringHook(C_rand, beta=beta, device=device)
        wrapper.update_hooks([(layer, hook)])
        r = run_condition(task, args.port, args.num_episodes,
                          cond_name, task_output_dir)
        all_results.append(r)
        existing_conds.add(cond_name)
        save_progress()

    # ── 4. Final summary ──
    logger.info(f"\n[4/4] Saving final summary...")
    save_progress()

    logger.info(f"\n{'='*70}")
    logger.info(f"{'Condition':<45s} {'SR':>8s}")
    logger.info(f"{'-'*55}")
    for r in all_results:
        logger.info(f"{r['condition']:<45s} {r['success_rate']:>8.3f}")
    logger.info(f"{'='*70}")
    logger.info(f"All results for {task} saved to {task_output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(tyro.cli(Args))
