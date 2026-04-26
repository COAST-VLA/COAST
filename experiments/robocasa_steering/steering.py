#!/usr/bin/env python3
"""Conceptor + linear steering for diffusion_policy on RoboCasa (in-process).

Runs one RoboCasa task's full steering sweep across five strategies against a
DiffusionTransformerHybridImagePolicy. The policy loads once; each condition
is a self-contained env_runner.run(policy) call with forward hooks installed
on chosen decoder layers and the flow-matching loop monkey-patched to inform
those hooks of the current denoising-step index.

Supported strategies (drop-in with the pi0.5 LIBERO naming):
    linear         h' = h + alpha*v   (v = unit(mean_s - mean_f))
    global         h' = (1-beta)h + beta(h @ C^T)   with C = C_s (I - C_f)
    per_step       Same as global but a DIFFERENT conceptor at each denoising
                   step - sparsely built in the npz, nearest-neighbour lookup
                   at runtime.
    positive_only  C = C_success only (no NOT-C_failure).
    random         Random SPD matrix with matched quota (control).

Pre-reqs:
    1. activations produced by  diffusion_policy/collect_activations_robocasa.py
    2. conceptor .npz produced by  experiments/robocasa_steering/build_conceptors.py
    3. trained checkpoint (same one evaluated by eval_robocasa.py)

Usage (from diffusion_policy repo root):
    python experiments/robocasa_steering/steering.py \\
        --checkpoint checkpoints/latest.ckpt \\
        --conceptor_npz ~/.cache/diffusion_policy/diffusion_policy_conceptors.npz \\
        --task CloseFridge --split pretrain \\
        --layers 5 8 11 --alphas 0.5 1.0 2.0 --betas 0.1 0.3 \\
        --num_rollouts 15 --num_envs 5
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import copy
import dataclasses
import json
import logging
import pathlib
import sys
from typing import Any

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import colored

from diffusion_policy.workspace.base_workspace import BaseWorkspace

import robocasa  # noqa: F401  (registers gym envs)
from robocasa.utils.dataset_registry_utils import get_task_horizon

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SyncVectorEnv shims for n_envs=1 (mirrors smoke_test_eval.py)
# ──────────────────────────────────────────────────────────────────────────────
def _install_sync_vector_env_shims():
    import diffusion_policy.env_runner.robomimic_image_runner as _runner_mod
    from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv as _SyncVectorEnv

    def _SyncVectorEnvWrapper(env_fns, dummy_env_fn=None, **_kwargs):
        return _SyncVectorEnv(env_fns)

    _runner_mod.AsyncVectorEnv = _SyncVectorEnvWrapper

    import diffusion_policy.gym_util.sync_vector_env as _sync_mod
    _real_concatenate = _sync_mod.concatenate

    def _concatenate_old_order(items, out, space):
        return _real_concatenate(space, items, out)

    _sync_mod.concatenate = _concatenate_old_order


# ──────────────────────────────────────────────────────────────────────────────
# n_envs>1 path: same RoboCasa / gym 0.26 fixes as collect_activations_robocasa.py.
#   1. force AsyncVectorEnv shared_memory=False (RoboCasa obs space is raw OrderedDict).
#   2. patch reset_async/reset_wait to accept seed/options kwargs (gym 0.26 calls them).
#   3. swap concatenate to old (items, out, space) arg order.
# ──────────────────────────────────────────────────────────────────────────────
def _install_async_no_shared_memory_shim():
    import diffusion_policy.env_runner.robomimic_image_runner as _runner_mod
    import diffusion_policy.gym_util.async_vector_env as _async_mod

    _gym_concatenate = _async_mod.concatenate

    def _concatenate_old_order(items, out, space):
        return _gym_concatenate(space, items, out)

    _async_mod.concatenate = _concatenate_old_order

    _RealAsync = _runner_mod.AsyncVectorEnv
    _real_reset_async = _RealAsync.reset_async
    _real_reset_wait = _RealAsync.reset_wait

    def _reset_async_compat(self, seed=None, options=None):
        return _real_reset_async(self)

    def _reset_wait_compat(self, seed=None, options=None, timeout=None):
        return _real_reset_wait(self, timeout=timeout)

    _RealAsync.reset_async = _reset_async_compat
    _RealAsync.reset_wait = _reset_wait_compat

    def _AsyncNoSharedMem(env_fns, dummy_env_fn=None, **kwargs):
        kwargs.setdefault("shared_memory", False)
        return _RealAsync(env_fns, dummy_env_fn=dummy_env_fn, **kwargs)

    _runner_mod.AsyncVectorEnv = _AsyncNoSharedMem


# ──────────────────────────────────────────────────────────────────────────────
# Steering specs + hooks (same contract as the pi0.5 bundle)
# ──────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class LayerSteeringSpec:
    """One layer's steering payload.

    Exactly one of C / C_per_step / v / v_per_step should be set.
    C_per_step / v_per_step are (matrix, ds_index)-pair lists. The hook maps
    the current denoising step to the nearest built ds_index.
    """
    C: np.ndarray | None = None
    C_per_step: list[tuple[np.ndarray, int]] | None = None
    v: np.ndarray | None = None
    v_per_step: list[tuple[np.ndarray, int]] | None = None
    beta: float = 0.3
    alpha: float = 1.0

    def mode(self) -> str:
        if self.C is not None:           return "conceptor_global"
        if self.C_per_step is not None:  return "conceptor_per_step"
        if self.v is not None:           return "linear_global"
        if self.v_per_step is not None:  return "linear_per_step"
        raise ValueError("empty LayerSteeringSpec")


class _PerStepLookup:
    """Maps a current denoising step (0..D-1) to the nearest built index in a
    (matrix, ds_index)-pair list. Precomputes the nearest-index lookup once."""

    def __init__(self, payload: list[tuple[np.ndarray, int]], num_steps: int):
        self.matrices = [item[0] for item in payload]
        built = np.array([item[1] for item in payload])
        self.lookup = np.empty(num_steps, dtype=np.int64)
        for s in range(num_steps):
            self.lookup[s] = int(np.argmin(np.abs(built - s)))

    def __call__(self, current_step: int) -> np.ndarray:
        return self.matrices[int(self.lookup[current_step])]


class ConceptorSteeringHook:
    """Forward hook:  h' = (1-beta) h + beta (h @ C.T)."""

    def __init__(self, spec: LayerSteeringSpec, device, num_denoising_steps: int):
        self.spec = spec
        self.beta = spec.beta
        self.device = device
        self.current_step = 0
        if spec.C is not None:
            self._single_M = self._make_M(spec.C)
            self._per_step = None
        elif spec.C_per_step is not None:
            self._single_M = None
            Cs_by_idx = _PerStepLookup(spec.C_per_step, num_denoising_steps)
            # Precompute M for each built ds - cheap (10 matrices) and avoids
            # re-making on every hook call.
            self._per_step_M = [self._make_M(C) for C in Cs_by_idx.matrices]
            self._per_step_lookup = Cs_by_idx.lookup
        else:
            raise ValueError("ConceptorSteeringHook requires C or C_per_step")
        self.intervention_norms: list[float] = []

    def _make_M(self, C: np.ndarray) -> torch.Tensor:
        d = C.shape[0]
        I = torch.eye(d, dtype=torch.float32, device=self.device)
        C_t = torch.tensor(C, dtype=torch.float32, device=self.device)
        return (1 - self.beta) * I + self.beta * C_t

    def set_denoise_step(self, t: int) -> None:
        self.current_step = t

    def reset_logs(self) -> None:
        self.intervention_norms = []

    def __call__(self, module, inputs, output):
        h, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        if self._single_M is not None:
            M = self._single_M
        else:
            idx = int(self._per_step_lookup[self.current_step])
            M = self._per_step_M[idx]
        M = M.to(dtype=h.dtype)
        h_steered = torch.matmul(h, M.T)
        self.intervention_norms.append(float(torch.norm(h_steered - h).item()))
        return (h_steered,) + rest if rest is not None else h_steered


class LinearSteeringHook:
    """Forward hook:  h' = h + alpha * v."""

    def __init__(self, spec: LayerSteeringSpec, device, num_denoising_steps: int):
        self.spec = spec
        self.alpha = spec.alpha
        self.device = device
        self.current_step = 0
        if spec.v is not None:
            self._single_v = torch.tensor(spec.v, dtype=torch.float32, device=device)
            self._per_step_v = None
        elif spec.v_per_step is not None:
            self._single_v = None
            lookup = _PerStepLookup(spec.v_per_step, num_denoising_steps)
            self._per_step_v = [torch.tensor(v, dtype=torch.float32, device=device)
                                for v in lookup.matrices]
            self._per_step_lookup = lookup.lookup
        else:
            raise ValueError("LinearSteeringHook requires v or v_per_step")
        self.intervention_norms: list[float] = []

    def set_denoise_step(self, t: int) -> None:
        self.current_step = t

    def reset_logs(self) -> None:
        self.intervention_norms = []

    def __call__(self, module, inputs, output):
        h, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
        if self._single_v is not None:
            v = self._single_v
        else:
            idx = int(self._per_step_lookup[self.current_step])
            v = self._per_step_v[idx]
        v = v.to(dtype=h.dtype)
        h_steered = h + self.alpha * v
        self.intervention_norms.append(float(torch.norm(h_steered - h).item()))
        return (h_steered,) + rest if rest is not None else h_steered


def compute_random_conceptor(d: int, alpha: float = 1.0, seed: int = 42) -> np.ndarray:
    """Random SPD matrix with a conceptor-shaped eigenvalue profile."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    raw = np.sort(rng.exponential(1.0, size=d))[::-1]
    eigs = raw / (raw + alpha ** -2)
    return (Q @ np.diag(eigs) @ Q.T).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Steering context: install hooks + monkey-patch conditional_sample so the
# hooks know which denoising-step index they're on.
# ──────────────────────────────────────────────────────────────────────────────

class SteeringContext:
    def __init__(self, policy, spec: dict[int, LayerSteeringSpec] | None, device, num_denoising_steps: int):
        self.policy = policy
        self.spec = spec
        self.device = device
        self.num_denoising_steps = num_denoising_steps
        self.hooks: list[tuple[int, Any]] = []
        self._handles: list = []
        self._patched_conditional_sample = False

    def __enter__(self):
        if not self.spec:
            return self

        # Build hooks and register forward hooks on the decoder layers.
        for layer_idx, layer_spec in self.spec.items():
            mode = layer_spec.mode()
            if mode.startswith("conceptor"):
                hook = ConceptorSteeringHook(layer_spec, self.device, self.num_denoising_steps)
            else:
                hook = LinearSteeringHook(layer_spec, self.device, self.num_denoising_steps)
            self.hooks.append((layer_idx, hook))
            layer_mod = self.policy.model.decoder.layers[layer_idx]
            self._handles.append(layer_mod.register_forward_hook(hook))

        # Monkey-patch conditional_sample to update hooks.current_step each ds.
        # We set an *instance* attribute; on exit we delete it so descriptor
        # lookup uncovers the class method again (cleaner than stashing the
        # original bound method).
        self._patched_conditional_sample = True
        policy = self.policy
        hooks = [h for _, h in self.hooks]

        def patched(condition_data, condition_mask, cond=None, generator=None, **kwargs):
            model = policy.model
            scheduler = policy.noise_scheduler
            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator,
            )
            scheduler.set_timesteps(policy.num_inference_steps)
            for ds_idx, t in enumerate(scheduler.timesteps):
                for h in hooks:
                    h.set_denoise_step(ds_idx)
                trajectory[condition_mask] = condition_data[condition_mask]
                model_output = model(trajectory, t, cond)
                trajectory = scheduler.step(
                    model_output, t, trajectory, generator=generator, **kwargs
                ).prev_sample
            trajectory[condition_mask] = condition_data[condition_mask]
            return trajectory

        self.policy.conditional_sample = patched
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._patched_conditional_sample:
            # Drop the instance attribute so the class-level method is visible again.
            try:
                del self.policy.conditional_sample
            except AttributeError:
                pass
            self._patched_conditional_sample = False


# ──────────────────────────────────────────────────────────────────────────────
# Conceptor npz accessors
# ──────────────────────────────────────────────────────────────────────────────

def _npz_has(npz, key: str) -> bool:
    return key in npz.files


def get_contrastive(npz, task, layer, alpha) -> np.ndarray:
    return npz[f"{task}__L{layer}__{alpha}__C_contrastive"]


def get_success_only(npz, task, layer, alpha) -> np.ndarray:
    return npz[f"{task}__L{layer}__{alpha}__C_success"]


def get_linear_contrastive(npz, task, layer) -> np.ndarray:
    return npz[f"{task}__L{layer}__linear__V_contrastive"]


def _per_step_indices_from_npz(npz) -> list[int]:
    if "_per_step_indices" in npz.files:
        return sorted(int(x) for x in npz["_per_step_indices"].tolist())
    # Fallback: derive from key parse.
    import re
    pat = re.compile(r"__per_step_(\d+)__C_contrastive$")
    found = set()
    for k in npz.files:
        m = pat.search(k)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def get_per_step_contrastive(npz, task, layer, alpha: float | None = None) -> list[tuple[np.ndarray, int]]:
    """Return [(C_contrastive, ds_index), ...] for the requested layer.

    If `alpha` is given, look for alpha-aware keys
        {task}__L{layer}__{alpha}__per_step_{ds}__C_contrastive
    first (added by add_per_step_alphas.py), and only fall back to the
    legacy alpha-less keys if none of the alpha-aware ones exist.
    """
    out = []
    for ds in _per_step_indices_from_npz(npz):
        key = None
        if alpha is not None:
            cand = f"{task}__L{layer}__{alpha}__per_step_{ds}__C_contrastive"
            if _npz_has(npz, cand):
                key = cand
        if key is None:
            cand = f"{task}__L{layer}__per_step_{ds}__C_contrastive"
            if _npz_has(npz, cand):
                key = cand
        if key is not None:
            out.append((npz[key], ds))
    return out


def get_per_step_linear_contrastive(npz, task, layer) -> list[tuple[np.ndarray, int]]:
    out = []
    for ds in _per_step_indices_from_npz(npz):
        key = f"{task}__L{layer}__linear_per_step_{ds}__V_contrastive"
        if _npz_has(npz, key):
            out.append((npz[key], ds))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# One condition = one env_runner.run(policy) under a SteeringContext
# ──────────────────────────────────────────────────────────────────────────────

def _load_policy_and_runner(checkpoint: str, task: str, num_rollouts: int, num_envs: int,
                            split: str, device: str, runner_output_dir: str):
    if num_envs == 1:
        _install_sync_vector_env_shims()
    else:
        _install_async_no_shared_memory_shim()

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cfg = copy.deepcopy(OmegaConf.to_container(cfg))
    cfg["task"]["env_runner"]["env_kwargs"] = {
        "split": split, "seed": 1111111, "env_name": task,
    }
    cfg = OmegaConf.create(cfg)

    horizon = get_task_horizon(task=task)
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_test = num_rollouts
    cfg.task.env_runner.max_steps = int(horizon * 1.5)
    cfg.task.env_runner.n_envs = num_envs

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=runner_output_dir)
    assert isinstance(workspace, BaseWorkspace)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    dev = torch.device(device)
    policy.to(dev)
    policy.eval()

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=runner_output_dir)
    return policy, env_runner, dev, cfg


def _extract_success_rate(runner_log: dict, task: str) -> float | None:
    # env_runner.run writes `success_rate/<env_name>` plus `test/mean_score`.
    key = f"success_rate/{task}"
    if key in runner_log:
        return float(runner_log[key])
    if "test/mean_score" in runner_log:
        return float(runner_log["test/mean_score"])
    return None


def run_condition(policy, env_runner, device, task: str, num_denoising_steps: int,
                  cond_name: str, spec: dict[int, LayerSteeringSpec] | None) -> dict[str, Any]:
    with SteeringContext(policy, spec, device, num_denoising_steps):
        with torch.no_grad():
            runner_log = env_runner.run(policy)
    sr = _extract_success_rate(runner_log, task)
    logger.info(f"  {cond_name}: SR={sr if sr is None else f'{sr:.3f}'}")
    return {"condition": cond_name, "success_rate": sr if sr is not None else float("nan")}


# ──────────────────────────────────────────────────────────────────────────────
# Summary I/O (resume-friendly merge; same contract as the pi0.5 bundle)
# ──────────────────────────────────────────────────────────────────────────────

def _load_summary(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path) as fh:
            return json.load(fh).get("conditions", []) or []
    except (json.JSONDecodeError, OSError):
        return []


def _save_summary(path: pathlib.Path, task: str, results: list[dict[str, Any]]) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for r in _load_summary(path):
        if r and "condition" in r:
            merged[r["condition"]] = r
    for r in results:
        if r and "condition" in r:
            merged[r["condition"]] = r
    sorted_results = sorted(merged.values(),
                            key=lambda x: (x.get("success_rate")
                                           if isinstance(x.get("success_rate"), (int, float))
                                              and not (isinstance(x.get("success_rate"), float) and np.isnan(x["success_rate"]))
                                           else -1.0),
                            reverse=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump({"task": task, "conditions": sorted_results}, fh, indent=2)
    tmp.replace(path)


# ──────────────────────────────────────────────────────────────────────────────
# Main: sweep strategies for one task
# ──────────────────────────────────────────────────────────────────────────────

def _build_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--conceptor_npz", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--split", default="pretrain")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_rollouts", type=int, default=15)
    p.add_argument("--num_envs", type=int, default=5)
    p.add_argument("--layers", type=int, nargs="+", default=[5, 8, 11])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.3])
    p.add_argument("--linear_alphas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    p.add_argument("--strategies", nargs="+",
                   default=["linear", "global", "per_step", "positive_only", "random"])
    p.add_argument("--n_random_controls", type=int, default=-1,
                   help="-1 = full layer*beta grid, 0 = skip, N = first N")
    p.add_argument("--output_dir", default="experiments/robocasa_steering/steering_results")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip the no-steering baseline condition (only run --strategies).")
    p.add_argument("--per-step-static-ds", type=int, default=None,
                   help="If set, the per_step strategy uses ONE conceptor at this ds index "
                        "applied uniformly across denoising steps (matches the pi05_libero / "
                        "pi05_robocasa 'per_step_K' semantics). Default: None = our "
                        "time-varying per_step (different conceptor at each ds).")
    return p.parse_args()


def main():
    args = _build_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    task_short = args.task[:60]
    task_output_dir = pathlib.Path(args.output_dir) / task_short
    task_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Task:       {args.task}")
    logger.info(f"Output:     {task_output_dir}")
    logger.info(f"Strategies: {args.strategies}")
    logger.info(f"Sweep:      layers={args.layers}  alphas={args.alphas}  "
                f"betas={args.betas}  linear_alphas={args.linear_alphas}")

    with open(task_output_dir / "sweep_args.json", "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    # ── Load policy + env_runner once ──────────────────────────────────────
    logger.info("Loading policy + env_runner (one-time cost) ...")
    runner_scratch = str(task_output_dir / "_runner_scratch")
    pathlib.Path(runner_scratch).mkdir(parents=True, exist_ok=True)
    policy, env_runner, device, _cfg = _load_policy_and_runner(
        checkpoint=args.checkpoint,
        task=args.task,
        num_rollouts=args.num_rollouts,
        num_envs=args.num_envs,
        split=args.split,
        device=args.device,
        runner_output_dir=runner_scratch,
    )
    num_denoising_steps = int(policy.num_inference_steps)
    hidden_dim = int(policy.model.decoder.layers[0].linear1.in_features)
    logger.info(f"Policy loaded: num_inference_steps={num_denoising_steps}, hidden_dim={hidden_dim}")

    # ── Conceptor npz ──────────────────────────────────────────────────────
    npz_path = pathlib.Path(args.conceptor_npz)
    if not npz_path.is_file():
        raise FileNotFoundError(f"conceptor npz not found: {npz_path}")
    npz = np.load(npz_path, allow_pickle=False)
    if args.task not in {k.split("__", 1)[0] for k in npz.files if not k.startswith("_")}:
        raise ValueError(f"task {args.task!r} has no conceptor entries in {npz_path}")

    # ── Resume-friendly summary ───────────────────────────────────────────
    summary_path = task_output_dir / "summary.json"
    all_results = _load_summary(summary_path)
    done_conds = {r["condition"] for r in all_results if r and "condition" in r}
    if done_conds:
        logger.info(f"Resuming - {len(done_conds)} conditions already on disk.")

    def record(r: dict):
        all_results.append(r)
        done_conds.add(r["condition"])
        _save_summary(summary_path, args.task, all_results)

    def _run(cond_name: str, spec: dict[int, LayerSteeringSpec] | None) -> None:
        if cond_name in done_conds:
            logger.info(f"  {cond_name} already present, skipping")
            return
        record(run_condition(policy, env_runner, device, args.task,
                             num_denoising_steps, cond_name, spec))

    try:
        # 1. Baseline (skippable via --skip-baseline)
        if args.skip_baseline:
            logger.info("\n[baseline]  skipped (--skip-baseline)")
        else:
            logger.info("\n[baseline]")
            _run("baseline", None)

        # Count for progress logging.
        total = 0
        if "linear" in args.strategies:        total += len(args.layers) * len(args.linear_alphas)
        if "global" in args.strategies:        total += len(args.layers) * len(args.alphas) * len(args.betas)
        if "per_step" in args.strategies:      total += len(args.layers) * len(args.alphas) * len(args.betas)
        if "positive_only" in args.strategies: total += len(args.layers) * len(args.alphas) * len(args.betas)
        if "random" in args.strategies:
            total += (len(args.layers) * len(args.betas) if args.n_random_controls < 0
                      else args.n_random_controls)
        logger.info(f"\nTotal steered conditions to run: {total}")
        idx = 0

        # 2. linear
        if "linear" in args.strategies:
            logger.info("\n[linear]  h' = h + alpha*v")
            for L in args.layers:
                try:
                    v = get_linear_contrastive(npz, args.task, L)
                except KeyError:
                    logger.warning(f"  no linear V at L{L}, skipping"); continue
                for a in args.linear_alphas:
                    idx += 1
                    cond = f"linear_L{L}_la{a}"
                    logger.info(f"  [{idx}/{total}] {cond}")
                    _run(cond, {L: LayerSteeringSpec(v=v, alpha=a)})

        # 3. global
        if "global" in args.strategies:
            logger.info("\n[global]  h' = (1-beta)h + beta(h @ C_contr.T)")
            for L in args.layers:
                for alpha in args.alphas:
                    try:
                        C = get_contrastive(npz, args.task, L, alpha)
                    except KeyError:
                        logger.warning(f"  no C_contrastive at L{L}/alpha={alpha}, skipping"); continue
                    for beta in args.betas:
                        idx += 1
                        cond = f"global_L{L}_a{alpha}_b{beta}"
                        logger.info(f"  [{idx}/{total}] {cond}")
                        _run(cond, {L: LayerSteeringSpec(C=C, beta=beta)})

        # 4. per_step
        # Two semantics:
        #   * default (--per-step-static-ds is None): TIME-VARYING — different conceptor
        #     at each ds via _PerStepLookup. Condition name `per_step_L{L}_a{α}_b{β}`.
        #   * static (--per-step-static-ds K):        STATIC at ds=K — one conceptor
        #     applied uniformly (matches pi05_libero/robocasa "per_step_K" semantics).
        #     Condition name `per_step_dsK_L{L}_a{α}_b{β}`.
        if "per_step" in args.strategies:
            if args.per_step_static_ds is not None:
                K = int(args.per_step_static_ds)
                logger.info(f"\n[per_step]  STATIC at ds={K}  "
                            f"(pi05-style: one C_contr applied uniformly across denoising steps)")
                for L in args.layers:
                    for alpha in args.alphas:
                        Cs = get_per_step_contrastive(npz, args.task, L, alpha=alpha)
                        if not Cs:
                            logger.warning(f"  no per-step C_contr at L{L}/alpha={alpha}, skipping")
                            continue
                        # Pick the (matrix, ds) pair whose ds == K (or nearest).
                        builds = sorted(Cs, key=lambda mC_ds: abs(mC_ds[1] - K))
                        C_static, ds_used = builds[0]
                        if ds_used != K:
                            logger.warning(f"  ds={K} not built at L{L}/alpha={alpha}; "
                                           f"using nearest ds={ds_used}")
                        for beta in args.betas:
                            idx += 1
                            cond = f"per_step_ds{ds_used}_L{L}_a{alpha}_b{beta}"
                            logger.info(f"  [{idx}/{total}] {cond}")
                            _run(cond, {L: LayerSteeringSpec(C=C_static, beta=beta)})
            else:
                logger.info("\n[per_step]  TIME-VARYING (one C_contr per built denoising step, "
                            "nearest-neighbour lookup)")
                for L in args.layers:
                    for alpha in args.alphas:
                        Cs = get_per_step_contrastive(npz, args.task, L, alpha=alpha)
                        if not Cs:
                            logger.warning(f"  no per-step C_contr at L{L}/alpha={alpha}, skipping")
                            continue
                        for beta in args.betas:
                            idx += 1
                            cond = f"per_step_L{L}_a{alpha}_b{beta}"
                            logger.info(f"  [{idx}/{total}] {cond}")
                            _run(cond, {L: LayerSteeringSpec(C_per_step=Cs, beta=beta)})

        # 5. positive_only
        if "positive_only" in args.strategies:
            logger.info("\n[positive_only]  C = C_success (no NOT-C_failure)")
            for L in args.layers:
                for alpha in args.alphas:
                    try:
                        C = get_success_only(npz, args.task, L, alpha)
                    except KeyError:
                        logger.warning(f"  no C_success at L{L}/alpha={alpha}, skipping"); continue
                    for beta in args.betas:
                        idx += 1
                        cond = f"posonly_L{L}_a{alpha}_b{beta}"
                        logger.info(f"  [{idx}/{total}] {cond}")
                        _run(cond, {L: LayerSteeringSpec(C=C, beta=beta)})

        # 6. random control
        if "random" in args.strategies:
            logger.info("\n[random]  matched-quota random SPD control")
            random_pairs = [(L, b) for L in args.layers for b in args.betas]
            if args.n_random_controls >= 0:
                random_pairs = random_pairs[: args.n_random_controls]
            for L, beta in random_pairs:
                idx += 1
                cond = f"random_L{L}_b{beta}"
                logger.info(f"  [{idx}/{total}] {cond}")
                C_rand = compute_random_conceptor(d=hidden_dim, seed=L * 100 + int(beta * 10))
                _run(cond, {L: LayerSteeringSpec(C=C_rand, beta=beta)})
    finally:
        env_runner.close()

    # ── Final summary ──────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 70}")
    logger.info(f"{'Condition':<45s} {'SR':>8s}")
    logger.info(f"{'-' * 55}")
    for r in sorted(all_results,
                    key=lambda x: (x.get("success_rate")
                                   if isinstance(x.get("success_rate"), (int, float))
                                      and not (isinstance(x.get("success_rate"), float) and np.isnan(x["success_rate"]))
                                   else -1.0),
                    reverse=True):
        sr = r.get("success_rate")
        sr_str = f"{sr:.3f}" if isinstance(sr, (int, float)) and not (isinstance(sr, float) and np.isnan(sr)) else "  nan"
        logger.info(f"{r['condition']:<45s} {sr_str:>8s}")
    logger.info(f"{'=' * 70}")
    logger.info(f"Results: {summary_path}")


if __name__ == "__main__":
    main()
