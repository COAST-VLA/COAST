"""
Benchmark activation collection pipeline to identify bottlenecks.

Measures each stage of the pipeline at different batch sizes (num_envs)
to understand scaling behavior and guide optimization.

Usage:
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl uv run scripts/benchmark_activations.py \
        --policy.config=pi05_metaworld \
        --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
        --num_envs_list 1 2 5 10 15
"""

from collections import defaultdict
from contextlib import contextmanager
import dataclasses
import json
import logging
import pathlib
import tempfile
import time

import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
import torch
import tyro

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.policies.policy import collate_transformed_singles
from openpi.training import config as _config

logger = logging.getLogger(__name__)

# Inline imports from collect_activations to avoid 'scripts' package issue
CAMERA_IDS = {"topview": 0, "corner": 1, "corner2": 2, "corner3": 3, "corner4": 4, "behindGripper": 5, "gripperPOV": 6}

TASK_TO_PROMPT = {"reach-v3": "reach the goal position", "door-open-v3": "open the door"}


class MultiCameraWrapper(gym.Wrapper):
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


def save_step_activations(step_dir, intermediates, env_id, step_metadata):
    step_dir.mkdir(parents=True, exist_ok=True)
    all_x_t = intermediates["all_x_t"][:, env_id]
    all_v_t = intermediates["all_v_t"][:, env_id]
    all_adarms_cond = intermediates["all_adarms_cond"][:, env_id]
    all_suffix_residual = intermediates["all_suffix_residual"][:, :, env_id]
    all_suffix_mlp_hidden = intermediates["all_suffix_mlp_hidden"][:, :, env_id]
    np.savez_compressed(step_dir / "denoising.npz", all_x_t=all_x_t, all_v_t=all_v_t)
    np.savez_compressed(step_dir / "adarms_cond.npz", all_adarms_cond=all_adarms_cond)
    np.savez_compressed(step_dir / "suffix_residual.npz", all_suffix_residual=all_suffix_residual)
    np.savez_compressed(step_dir / "suffix_mlp_hidden.npz", all_suffix_mlp_hidden=all_suffix_mlp_hidden)
    with open(step_dir / "metadata.json", "w") as f:
        json.dump(step_metadata, f, indent=2)


class Timer:
    """Accumulates timing measurements."""

    def __init__(self):
        self.timings: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def time(self, name: str):
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.timings[name].append(time.perf_counter() - start)

    def mean_ms(self, name: str) -> float:
        vals = self.timings[name]
        return (sum(vals) / len(vals)) * 1000 if vals else 0.0

    def reset(self):
        self.timings.clear()


@dataclasses.dataclass
class PolicyArgs:
    config: str = "pi05_metaworld"
    dir: str = "checkpoints/pi05_metaworld/pi05_metaworld_test/5000/"


@dataclasses.dataclass
class Args:
    policy: PolicyArgs = dataclasses.field(default_factory=PolicyArgs)
    # Batch sizes to benchmark.
    num_envs_list: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 5, 10, 15])
    # Inference calls per batch size (averaged).
    num_inference_calls: int = 3
    # Task to benchmark on.
    task: str = "reach-v3"
    seed: int = 69_420


def make_env(task: str, num_envs: int, seed: int) -> gym.vector.VectorEnv:
    env_fns = [
        lambda i=i: MultiCameraWrapper(
            gym.make("Meta-World/MT1", env_name=task, seed=seed + i, width=224, height=224),
            ["corner", "corner4", "gripperPOV"],
        )
        for i in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)


def build_obs_dict(obs, camera_views, prompt, num_envs):
    return {
        "observation/image": camera_views["corner4"],
        "observation/wrist_image": camera_views["gripperPOV"],
        "observation/state": obs.astype(np.float32)[..., :4],
        "prompt": [prompt] * num_envs,
    }


def benchmark_model_internals(policy, obs_dict, timer, num_calls):
    """Time internal model stages: preprocess, prefix pass, denoising loop, numpy conversion."""
    import jax

    model = policy._model  # noqa: SLF001
    device = policy._pytorch_device  # noqa: SLF001

    # Prepare inputs once (same as infer_with_intermediates)
    inputs = jax.tree.map(lambda x: x, obs_dict)
    singles = [{k: v[i] for k, v in inputs.items()} for i in range(int(inputs["observation/state"].shape[0]))]
    singles = [policy._input_transform(ex) for ex in singles]  # noqa: SLF001
    inputs = collate_transformed_singles(singles)
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device), inputs)
    observation = _model.Observation.from_dict(inputs)

    for _ in range(num_calls):
        bsize = observation.state.shape[0]
        noise = model.sample_noise((bsize, model.config.action_horizon, model.config.action_dim), device)

        # Preprocess
        with timer.time("model_preprocess"):
            images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(  # noqa: SLF001
                observation, train=False
            )

        # Prefix pass
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        with timer.time("model_prefix_pass"):
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
            model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_key_values = model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        # Denoising loop (10 steps, no hooks — just forward passes)
        x_t = noise
        dt_tensor = torch.tensor(-1.0 / 10, dtype=torch.float32, device=device)
        time_val = torch.tensor(1.0, dtype=torch.float32, device=device)

        with timer.time("model_denoising_loop"):
            while time_val >= -dt_tensor / 2:
                expanded_time = time_val.expand(bsize)
                suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(
                    state, x_t, expanded_time
                )
                suffix_len = suffix_pad_masks.shape[1]
                prefix_len = prefix_pad_masks.shape[1]
                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
                suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
                full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
                full_att_2d_masks_4d = model._prepare_attention_masks_4d(full_att_2d_masks)  # noqa: SLF001
                model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
                outputs_embeds, _ = model.paligemma_with_expert.forward(
                    attention_mask=full_att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=[None, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                )
                suffix_out = outputs_embeds[1][:, -model.config.action_horizon :]
                v_t = model.action_out_proj(suffix_out.to(dtype=torch.float32))
                x_t = x_t + dt_tensor * v_t
                time_val += dt_tensor

        # Numpy conversion (simulate what sample_actions_with_intermediates does)
        # Create dummy tensors matching the sizes
        dummy_x = [torch.randn(bsize, 32, 32) for _ in range(10)]
        dummy_res = [torch.randn(4, bsize, 32, 1024) for _ in range(10)]
        dummy_mlp = [torch.randn(4, bsize, 32, 4096) for _ in range(10)]
        dummy_cond = [torch.randn(bsize, 1024) for _ in range(10)]

        with timer.time("model_numpy_conversion"):
            torch.stack(dummy_x).float().numpy()
            torch.stack(dummy_x).float().numpy()  # v_t same shape
            torch.stack(dummy_cond).float().numpy()
            torch.stack(dummy_res).float().numpy()
            torch.stack(dummy_mlp).float().numpy()


def run_benchmark(num_envs: int, policy, args: Args) -> dict:
    """Run benchmark for a single num_envs value. Returns timing dict."""
    timer = Timer()
    prompt = TASK_TO_PROMPT[args.task]

    # Create envs
    env = make_env(args.task, num_envs, args.seed)
    try:
        # Reset
        with timer.time("env_reset"):
            obs, info = env.reset(seed=args.seed)
            camera_views = info["cameras"]

        obs_dict = build_obs_dict(obs, camera_views, prompt, num_envs)

        # Warmup (untimed)
        print(f"  Warmup inference (num_envs={num_envs})...")
        result, intermediates = policy.infer_with_intermediates(obs_dict)
        action_chunk = np.clip(result["actions"], -1.0, 1.0).astype(np.float32)
        torch.cuda.synchronize()

        # Benchmark loop
        for call_idx in range(args.num_inference_calls):
            # Env stepping (10 steps)
            with timer.time("env_stepping"):
                for t in range(10):
                    action = action_chunk[:, t, :]
                    obs, reward, terminated, truncated, info = env.step(action)
                    camera_views = info["cameras"]

            obs_dict = build_obs_dict(obs, camera_views, prompt, num_envs)

            # Full inference call
            with timer.time("inference_total"):
                result, intermediates = policy.infer_with_intermediates(obs_dict)

            action_chunk = np.clip(result["actions"], -1.0, 1.0).astype(np.float32)

            # Disk I/O (compressed)
            with tempfile.TemporaryDirectory() as tmpdir, timer.time("disk_io_compressed"):
                for env_id in range(num_envs):
                    step_dir = pathlib.Path(tmpdir) / f"env_{env_id}"
                    metadata = {"task_name": args.task, "env_id": env_id, "step": call_idx}
                    save_step_activations(step_dir, intermediates, env_id, metadata)

            # Disk I/O (uncompressed — np.savez instead of np.savez_compressed)
            with tempfile.TemporaryDirectory() as tmpdir, timer.time("disk_io_uncompressed"):
                for env_id in range(num_envs):
                    step_dir = pathlib.Path(tmpdir) / f"env_{env_id}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    x_t = intermediates["all_x_t"][:, env_id]
                    v_t = intermediates["all_v_t"][:, env_id]
                    cond = intermediates["all_adarms_cond"][:, env_id]
                    res = intermediates["all_suffix_residual"][:, :, env_id]
                    mlp = intermediates["all_suffix_mlp_hidden"][:, :, env_id]
                    np.savez(step_dir / "denoising.npz", all_x_t=x_t, all_v_t=v_t)
                    np.savez(step_dir / "adarms_cond.npz", all_adarms_cond=cond)
                    np.savez(step_dir / "suffix_residual.npz", all_suffix_residual=res)
                    np.savez(step_dir / "suffix_mlp_hidden.npz", all_suffix_mlp_hidden=mlp)

        # Model internals breakdown (separate from the main loop)
        # Free GPU memory first to avoid OOM
        torch.cuda.empty_cache()
        obs_dict = build_obs_dict(obs, camera_views, prompt, num_envs)
        try:
            benchmark_model_internals(policy, obs_dict, timer, args.num_inference_calls)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM during model internals benchmark for num_envs={num_envs}, skipping")

    finally:
        env.close()

    return timer


def print_results(timer: Timer, num_envs: int, num_calls: int):
    """Print timing breakdown table."""
    env_step = timer.mean_ms("env_stepping")
    inference = timer.mean_ms("inference_total")
    disk_compressed = timer.mean_ms("disk_io_compressed")
    disk_uncompressed = timer.mean_ms("disk_io_uncompressed")

    prefix = timer.mean_ms("model_prefix_pass")
    denoising = timer.mean_ms("model_denoising_loop")
    numpy_conv = timer.mean_ms("model_numpy_conversion")
    preprocess = timer.mean_ms("model_preprocess")

    # Use uncompressed for the "total" since that's the optimization path
    total_compressed = env_step + inference + disk_compressed
    total_uncompressed = env_step + inference + disk_uncompressed

    def pct(val, total):
        return val / total * 100 if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"num_envs={num_envs}, {num_calls} inference calls (mean)")
    print(f"{'=' * 60}")
    print(f"{'Component':<35} {'Time (ms)':>10} {'%':>8}")
    print(f"{'─' * 55}")
    print(f"{'Env stepping (10 steps)':<35} {env_step:>10.0f} {pct(env_step, total_compressed):>7.1f}%")
    print(f"{'Inference total':<35} {inference:>10.0f} {pct(inference, total_compressed):>7.1f}%")
    print(f"{'  Preprocess':<35} {preprocess:>10.0f}")
    print(f"{'  Prefix pass':<35} {prefix:>10.0f}")
    print(f"{'  Denoising loop (10 iters)':<35} {denoising:>10.0f}")
    print(f"{'  Numpy conversion':<35} {numpy_conv:>10.0f}")
    print(f"{'Disk I/O compressed':<35} {disk_compressed:>10.0f} {pct(disk_compressed, total_compressed):>7.1f}%")
    print(f"{'Disk I/O uncompressed':<35} {disk_uncompressed:>10.0f}")
    print(f"{'─' * 55}")
    print(f"{'Total (w/ compressed IO)':<35} {total_compressed:>10.0f}")
    print(f"{'Total (w/ uncompressed IO)':<35} {total_uncompressed:>10.0f}")

    # ~30 inference calls per task (300 steps / 10 replan_steps)
    est_compressed_min = 30 * total_compressed / 60_000
    est_uncompressed_min = 30 * total_uncompressed / 60_000
    print(
        f"\nWith compressed IO:   {60_000 / total_compressed:.1f} calls/min, 1 task: {est_compressed_min:.1f} min, 45 tasks: {est_compressed_min * 45 / 60:.1f}h"
    )
    print(
        f"With uncompressed IO: {60_000 / total_uncompressed:.1f} calls/min, 1 task: {est_uncompressed_min:.1f} min, 45 tasks: {est_uncompressed_min * 45 / 60:.1f}h"
    )

    return {
        "num_envs": num_envs,
        "env_stepping_ms": env_step,
        "inference_ms": inference,
        "disk_compressed_ms": disk_compressed,
        "disk_uncompressed_ms": disk_uncompressed,
        "prefix_ms": prefix,
        "denoising_ms": denoising,
        "numpy_ms": numpy_conv,
        "total_compressed_ms": total_compressed,
        "total_uncompressed_ms": total_uncompressed,
    }


def print_summary(all_results: list[dict]):
    """Print comparison table across all num_envs values."""
    print(f"\n{'=' * 90}")
    print("SUMMARY: Scaling across batch sizes")
    print(f"{'=' * 90}")
    print(
        f"{'envs':>5} {'Env (ms)':>9} {'Infer (ms)':>11} {'Disk-C (ms)':>12} {'Disk-U (ms)':>12} "
        f"{'Total-C (ms)':>13} {'Total-U (ms)':>13}"
    )
    print(f"{'─' * 90}")
    for r in all_results:
        print(
            f"{r['num_envs']:>5} {r['env_stepping_ms']:>9.0f} {r['inference_ms']:>11.0f} "
            f"{r['disk_compressed_ms']:>12.0f} {r['disk_uncompressed_ms']:>12.0f} "
            f"{r['total_compressed_ms']:>13.0f} {r['total_uncompressed_ms']:>13.0f}"
        )
    print("\n(C = compressed npz, U = uncompressed npz)")


def main(args: Args):
    # Load policy once
    train_config = _config.get_config(args.policy.config)
    policy = _policy_config.create_trained_policy(train_config, args.policy.dir)
    print(f"Policy loaded (pytorch={policy._is_pytorch_model})")  # noqa: SLF001

    all_results = []
    for num_envs in args.num_envs_list:
        print(f"\n--- Benchmarking num_envs={num_envs} ---")
        timer = run_benchmark(num_envs, policy, args)
        result = print_results(timer, num_envs, args.num_inference_calls)
        all_results.append(result)
        torch.cuda.empty_cache()

    if len(all_results) > 1:
        print_summary(all_results)

    # Save results
    out_path = pathlib.Path("benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
