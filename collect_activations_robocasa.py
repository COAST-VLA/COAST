"""Collect per-step / per-episode activations from a diffusion_policy
rollout on RoboCasa, in the same on-disk layout as openpi's server-side
collector (see /home/kim34/projects/openpi-metaworld/docs/activation_collection.md).

Output tree:
    <output_root>/<checkpoint_step>/<task_name>/
        episode_NNN_env_NNN/
            metadata.json, rewards.npz
            step_NNNN/
                metadata.json
                denoising.npz       all_x_t (D, H, A), all_v_t (D, H, A)         fp32
                adarms_cond.npz     all_adarms_cond (D, T_cond, C)               fp32
                suffix_residual.npz all_suffix_residual (D, L, H, C)             fp32

Schema identifier: "dp_v1" (stamped in step metadata.json). Differs from pi0.5's
`v1` in that `all_adarms_cond` is the full per-step cond token sequence
(shape (D, T_cond, C)) rather than a pooled (D, C) vector, because diffusion_policy's
TransformerForDiffusion feeds conditioning as cross-attention tokens, not AdaLN.

Usage:
    # Single task (smoke-test scale, n_envs=1 → uses SyncVectorEnv):
    python collect_activations_robocasa.py \\
        --checkpoint /home/kim34/projects/diffusion_policy/checkpoints/latest.ckpt \\
        --activations_output_dir ./activations \\
        --task CloseStandMixerHead --split pretrain \\
        --num_rollouts 1 --num_envs 1

    # Full task set:
    python collect_activations_robocasa.py \\
        --checkpoint .../latest.ckpt \\
        --activations_output_dir ./activations \\
        --task_set atomic_seen --split pretrain \\
        --num_rollouts 15 --num_envs 5
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import copy
import json
import math
import pathlib
import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import dill
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from termcolor import colored

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import robocasa  # noqa: F401  (registers gym envs; required by env_runner)
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon


# --------------------------------------------------------------------------- #
# n_envs=1 only: swap AsyncVectorEnv for SyncVectorEnv and fix gym-0.26
# concatenate arg order. Mirrors smoke_test_eval.py so this script also works as
# a single-env smoke test.
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# n_envs>1 path: the repo's custom AsyncVectorEnv predates gym 0.26.
#   1. robocasa envs publish observation_space as raw collections.OrderedDict;
#      the gym shared_memory path rejects that → force shared_memory=False
#      (the pipe-based _worker path supports custom spaces).
#   2. gym 0.26 VectorEnv.reset() calls reset_async(seed=, options=) and
#      reset_wait(seed=, options=); the custom reset_async/reset_wait take no
#      kwargs → wrap them to accept & ignore.
#   3. async_vector_env.py calls concatenate(items, out, space) (old gym-0.21
#      arg order); gym 0.26 expects concatenate(space, items, out) → patch the
#      module-level `concatenate` name.
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Activation capture
# --------------------------------------------------------------------------- #
class TransformerActivationCapture:
    """Forward-hook instrumentation for DiffusionTransformerHybridImagePolicy.

    Captures, per denoising step:
      - model input sample (x_t) and output (v_t)
      - encoder output (cond sequence: time + obs tokens after the cond encoder)
      - each TransformerDecoderLayer's output (residual stream)
    """

    def __init__(self, policy):
        transformer = policy.model
        if not (hasattr(transformer, "decoder") and hasattr(transformer, "encoder")):
            raise NotImplementedError(
                "TransformerActivationCapture only supports "
                "DiffusionTransformerHybridImagePolicy (TransformerForDiffusion). "
                "Got model type: {}".format(type(transformer).__name__)
            )
        if transformer.decoder is None:
            raise NotImplementedError(
                "Encoder-only TransformerForDiffusion is not yet supported. "
                "This script assumes obs_as_cond=True (decoder-based model)."
            )
        self.policy = policy
        self.transformer = transformer
        self.num_layers = len(transformer.decoder.layers)
        self.handles: list = []
        self._reset_buffers()
        self._install_hooks()

    def _reset_buffers(self):
        self.x_t: list = []
        self.v_t: list = []
        self.cond: list = []
        self.residual: list[list] = [[] for _ in range(self.num_layers)]

    def _install_hooks(self):
        def model_hook(module, inputs, output):
            self.x_t.append(inputs[0].detach().to("cpu").numpy())
            self.v_t.append(output.detach().to("cpu").numpy())

        self.handles.append(self.transformer.register_forward_hook(model_hook))

        def enc_hook(module, inputs, output):
            self.cond.append(output.detach().to("cpu").numpy())

        self.handles.append(self.transformer.encoder.register_forward_hook(enc_hook))

        for i, layer in enumerate(self.transformer.decoder.layers):
            def make_residual_hook(idx):
                def fn(module, inputs, output):
                    self.residual[idx].append(output.detach().to("cpu").numpy())
                return fn

            self.handles.append(layer.register_forward_hook(make_residual_hook(i)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def reset(self):
        self._reset_buffers()

    def stacked(self) -> dict:
        """Stack per-denoising-step buffers. Shapes: (D, B, ...) or (D, L, B, ...)."""
        all_x_t = np.stack(self.x_t, axis=0)  # (D, B, H, A)
        all_v_t = np.stack(self.v_t, axis=0)  # (D, B, H, A)
        all_cond = np.stack(self.cond, axis=0)  # (D, B, T_cond, C)
        # residual[l]: D entries of shape (B, H, C) -> stack over D, then over L.
        residual = np.stack(
            [np.stack(xs, axis=0) for xs in self.residual], axis=1
        )  # (D, L, B, H, C)
        return {
            "all_x_t": all_x_t,
            "all_v_t": all_v_t,
            "all_adarms_cond": all_cond,
            "all_suffix_residual": residual,
        }


# --------------------------------------------------------------------------- #
# On-disk writers (match openpi/serving/activation_collector.py byte-for-byte
# for the v1 schema; adarms_cond carries a 3-D array here vs pi0.5's 2-D).
# --------------------------------------------------------------------------- #
def save_step_activations(step_dir: pathlib.Path, intermediates: dict, env_id: int, step_metadata: dict):
    step_dir.mkdir(parents=True, exist_ok=True)

    all_x_t = intermediates["all_x_t"][:, env_id].astype(np.float32)
    all_v_t = intermediates["all_v_t"][:, env_id].astype(np.float32)
    all_adarms_cond = intermediates["all_adarms_cond"][:, env_id].astype(np.float32)
    all_suffix_residual = intermediates["all_suffix_residual"][:, :, env_id].astype(np.float32)

    np.savez(step_dir / "denoising.npz", all_x_t=all_x_t, all_v_t=all_v_t)
    np.savez(step_dir / "adarms_cond.npz", all_adarms_cond=all_adarms_cond)
    np.savez(step_dir / "suffix_residual.npz", all_suffix_residual=all_suffix_residual)

    with open(step_dir / "metadata.json", "w") as f:
        json.dump(step_metadata, f, indent=2)


def save_episode_files(episode_dir: pathlib.Path, episode_metadata: dict, per_step_reward, per_step_success):
    episode_dir.mkdir(parents=True, exist_ok=True)
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(episode_metadata, f, indent=2)

    rewards_arr = np.array(per_step_reward, dtype=np.float32)
    cumulative_arr = np.cumsum(rewards_arr).astype(np.float32) if len(rewards_arr) else rewards_arr
    success_arr = np.array(per_step_success, dtype=bool)
    np.savez(
        episode_dir / "rewards.npz",
        per_step_reward=rewards_arr,
        cumulative_reward=cumulative_arr,
        success_at_step=success_arr,
    )


# --------------------------------------------------------------------------- #
# Collection loop — mirrors RobomimicImageRunner.run() but captures activations
# on every policy call and writes per-env-per-step files.
# --------------------------------------------------------------------------- #
def _episode_dir(output_root, checkpoint_step, task_name, episode_id, env_id):
    return (
        pathlib.Path(output_root)
        / checkpoint_step
        / task_name
        / "episode_{:03d}_env_{:03d}".format(episode_id, env_id)
    )


def run_collection(
    policy,
    env_runner,
    task_name: str,
    checkpoint_step: str,
    output_root: pathlib.Path,
    policy_dir: str,
    config_name: str,
    prompt: str,
    tqdm_interval_sec: float = 5.0,
):
    device = policy.device
    env = env_runner.env
    n_envs = len(env_runner.env_fns)
    n_inits = len(env_runner.env_init_fn_dills)
    n_chunks = math.ceil(n_inits / n_envs)
    n_action_steps = env_runner.n_action_steps
    max_steps = env_runner.max_steps
    abs_action = env_runner.abs_action

    capture = TransformerActivationCapture(policy)
    episode_success_out: list[bool] = []

    try:
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_n_active = end - start

            this_init_fns = env_runner.env_init_fn_dills[start:end]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns = this_init_fns + [env_runner.env_init_fn_dills[0]] * n_diff

            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])
            obs = env.reset()
            policy.reset()

            episode_ids = [start + i for i in range(this_n_active)]
            per_step_reward = [[] for _ in range(this_n_active)]
            per_step_success = [[] for _ in range(this_n_active)]
            cumulative_reward = [0.0] * this_n_active
            reward_since_last_inference = [0.0] * this_n_active
            steps_to_success = [-1] * this_n_active
            episode_success = [False] * this_n_active
            total_inference_steps = [0] * this_n_active
            active = [True] * this_n_active
            prev_reward_len = [0] * this_n_active  # len(wrapper.reward) last observed

            env_step_counter = 0

            pbar = tqdm.tqdm(
                total=max_steps,
                desc=f"Collect {task_name} {chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=tqdm_interval_sec,
            )
            done_all = False
            while not done_all:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

                capture.reset()
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                intermediates = capture.stacked()  # (D, B, ...) across envs

                np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
                action = np_action_dict["action"]

                # Save activations for every still-active env at the current env step.
                for eid in range(this_n_active):
                    if not active[eid]:
                        continue
                    step_meta = {
                        "task_name": task_name,
                        "episode_id": int(episode_ids[eid]),
                        "env_id": int(eid),
                        "step": int(env_step_counter),
                        "inference_step": int(total_inference_steps[eid]),
                        "prompt": prompt,
                        "cumulative_reward": float(cumulative_reward[eid]),
                        "success_so_far": bool(episode_success[eid]),
                        "reward_since_last_inference": float(reward_since_last_inference[eid]),
                        "collection_version": "dp_v1",
                    }
                    step_dir = _episode_dir(
                        output_root, checkpoint_step, task_name, episode_ids[eid], eid
                    ) / "step_{:04d}".format(env_step_counter)
                    save_step_activations(step_dir, intermediates, env_id=eid, step_metadata=step_meta)
                    total_inference_steps[eid] += 1
                    reward_since_last_inference[eid] = 0.0

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                env_action = action
                if abs_action:
                    env_action = env_runner.undo_transform_action(action)
                obs, reward, done, info = env.step(env_action)

                # Pull the full per-sub-step reward history per env from the
                # MultiStepWrapper so we get true per-env-step granularity (the
                # aggregated scalar in `reward` is just max()).
                full_reward_lists = env.call("get_attr", "reward")  # list[list[float]], len n_envs
                for eid in range(this_n_active):
                    if not active[eid]:
                        continue
                    new_rewards = full_reward_lists[eid][prev_reward_len[eid]:]
                    for r in new_rewards:
                        r_f = float(r)
                        per_step_reward[eid].append(r_f)
                        cumulative_reward[eid] += r_f
                        reward_since_last_inference[eid] += r_f
                        # robocasa tasks: success is encoded as reward > 0.
                        succ = r_f > 0.0
                        per_step_success[eid].append(succ)
                        if succ and steps_to_success[eid] == -1:
                            steps_to_success[eid] = len(per_step_reward[eid]) - 1
                            episode_success[eid] = True
                    prev_reward_len[eid] = len(full_reward_lists[eid])

                env_step_counter += n_action_steps
                pbar.update(n_action_steps)

                # Per-env termination: mark inactive once done or success seen,
                # but keep stepping the vector env until all are inactive.
                for eid in range(this_n_active):
                    if not active[eid]:
                        continue
                    this_info = info[eid] if isinstance(info, (list, tuple)) else info
                    env_done = bool(np.asarray(done[eid]).any()) if hasattr(done, "__len__") else bool(done)
                    succ_flags = this_info.get("success", []) if isinstance(this_info, dict) else []
                    succ_any = bool(np.any(np.asarray(succ_flags))) if len(succ_flags) else False
                    if env_done or succ_any or episode_success[eid] or env_step_counter >= max_steps:
                        active[eid] = False

                done_all = (not any(active)) or env_step_counter >= max_steps
            pbar.close()

            # Finalize per-env episode files (only the initially-active envs).
            for eid in range(this_n_active):
                ep_dir = _episode_dir(
                    output_root, checkpoint_step, task_name, episode_ids[eid], eid
                )
                ep_meta = {
                    "task_name": task_name,
                    "episode_id": int(episode_ids[eid]),
                    "env_id": int(eid),
                    "episode_success": bool(episode_success[eid]),
                    "total_reward": float(cumulative_reward[eid]),
                    "steps_to_success": int(steps_to_success[eid]),
                    "total_env_steps": int(len(per_step_reward[eid])),
                    "total_inference_steps": int(total_inference_steps[eid]),
                    "prompt": prompt,
                    "checkpoint_dir": str(policy_dir),
                    "config_name": config_name,
                }
                save_episode_files(ep_dir, ep_meta, per_step_reward[eid], per_step_success[eid])
                episode_success_out.append(episode_success[eid])

        env.reset()
    finally:
        capture.remove()

    return episode_success_out


# --------------------------------------------------------------------------- #
# Orchestration (mirrors eval_robocasa.eval_task)
# --------------------------------------------------------------------------- #
def collect_task(
    checkpoint: str,
    activations_output_dir: str,
    device: str,
    task: str,
    num_rollouts: int,
    num_envs: int,
    split: str,
    prompt: str = "",
    runner_output_dir: str | None = None,
):
    if num_envs == 1:
        _install_sync_vector_env_shims()
    else:
        _install_async_no_shared_memory_shim()

    checkpoint_path = pathlib.Path(checkpoint)
    output_root = pathlib.Path(activations_output_dir).resolve()
    checkpoint_step = checkpoint_path.stem  # e.g. "latest"
    if runner_output_dir is None:
        runner_output_dir = str(output_root / checkpoint_step / task / "_runner_scratch")
    pathlib.Path(runner_output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cfg = copy.deepcopy(OmegaConf.to_container(cfg))
    cfg["task"]["env_runner"]["env_kwargs"] = {
        "split": split,
        "seed": 1111111,
        "env_name": task,
    }
    cfg = OmegaConf.create(cfg)

    horizon = get_task_horizon(task=task)
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_test = num_rollouts
    cfg.task.env_runner.max_steps = int(horizon * 1.5)
    cfg.task.env_runner.n_envs = num_envs

    config_name = cfg.get("name", None) or cfg.policy._target_

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=runner_output_dir)
    assert isinstance(workspace, BaseWorkspace)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    dev = torch.device(device)
    policy.to(dev)
    policy.eval()

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=runner_output_dir)

    try:
        successes = run_collection(
            policy=policy,
            env_runner=env_runner,
            task_name=task,
            checkpoint_step=checkpoint_step,
            output_root=output_root,
            policy_dir=str(checkpoint_path.parent),
            config_name=str(config_name),
            prompt=prompt,
        )
    finally:
        env_runner.close()

    n = len(successes)
    sr = (sum(bool(x) for x in successes) / n) if n else 0.0
    print(colored(f"[{task}] success {sum(successes)}/{n} = {sr:.3f}", "green"))
    return successes


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-a", "--activations_output_dir", required=True,
                        help="Root directory for the activation tree "
                             "(<root>/<checkpoint_stem>/<task>/...).")
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("-s", "--split", required=True)
    parser.add_argument("-n", "--num_rollouts", default=15, type=int)
    parser.add_argument("-e", "--num_envs", default=5, type=int)
    parser.add_argument("--prompt", default="",
                        help="Optional task-instruction string stamped into step metadata.")
    parser.add_argument("--runner_output_dir", default=None,
                        help="Scratch directory for env_runner outputs "
                             "(videos, etc.). Defaults to <activations>/<ckpt>/<task>/_runner_scratch.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--task", help="Single task name (e.g. CloseStandMixerHead).")
    group.add_argument("-T", "--task_set", nargs="+",
                       help="One or more task-set names from TASK_SET_REGISTRY.")
    args = parser.parse_args()

    if args.task is not None:
        tasks = [args.task]
    else:
        tasks = []
        for s in args.task_set:
            tasks.extend(TASK_SET_REGISTRY[s])
        tasks = sorted(set(tasks))

    for i, task in enumerate(tasks):
        print(colored(f"[{i + 1}/{len(tasks)}] collecting activations for {task}", "yellow"))
        collect_task(
            checkpoint=args.checkpoint,
            activations_output_dir=args.activations_output_dir,
            device=args.device,
            task=task,
            num_rollouts=args.num_rollouts,
            num_envs=args.num_envs,
            split=args.split,
            prompt=args.prompt,
            runner_output_dir=args.runner_output_dir,
        )


if __name__ == "__main__":
    main()
