# Plan: Activation Collection Speed Benchmark

## Context

We need 15 envs per task (45 tasks) for meaningful mech-interp analysis, but the current pipeline is slow. With 2 envs, one task takes ~2-5 min. With 15 envs, we observed ~2.5 min per inference call (vs ~25s with batch=2). Before optimizing, we need to know exactly where time is spent.

The benchmark will measure each pipeline stage at different `num_envs` values to identify the bottleneck and guide optimization.

## File to create

`scripts/benchmark_activations.py`

## Design

### What to measure

One "inference cycle" = 10 env steps + 1 policy inference + disk I/O. The benchmark times each stage:

| Stage | What it measures |
|-------|-----------------|
| Env stepping (10 steps) | SyncVectorEnv step + 3-camera rendering per step |
| Input transforms | Unbatch → per-example transform → collate → GPU transfer |
| Model inference (total) | Full `sample_actions_with_intermediates` with cuda sync |
| — Prefix pass | embed_prefix + KV cache forward (once per inference) |
| — Denoising loop (10x) | embed_suffix + expert forward + hook captures per step |
| — Numpy conversion | torch.stack + .float().numpy() for all intermediates |
| Output transforms | GPU→CPU detach + unnormalize |
| Disk I/O | save 4 compressed npz + json per env to tmpdir |

### Approach

1. Load policy once, create SyncVectorEnv with `num_envs`
2. Do 1 warmup inference call (untimed) to warm CUDA kernels
3. For `num_inference_calls` iterations (default 3):
   - Time env stepping: 10 `env.step()` calls with the previous action chunk
   - Time full `policy.infer_with_intermediates()` call (with `torch.cuda.synchronize()` before/after for accurate GPU timing)
   - Time disk I/O: `save_step_activations` to a tempdir
4. For model internals breakdown, do separate isolated calls:
   - Prepare observation tensors once
   - Time prefix pass only (embed_prefix + KV cache forward) with cuda sync
   - Time denoising-only (reuse cached KV, run 10 denoise iterations) with cuda sync
   - Time numpy conversion only (stack + float + numpy on pre-collected tensors)
5. Print results table for this `num_envs` value
6. Repeat for each `num_envs` in the list

### Key implementation details

- **`torch.cuda.synchronize()`** before every `time.perf_counter()` call — GPU ops are async, without sync the timings are wrong
- **Model internals**: Access `policy._model` directly. Call `_preprocess_observation`, `embed_prefix`, `embed_suffix`, and the denoising loop stages individually. This avoids modifying the model code.
- **Disk I/O**: Use `tempfile.TemporaryDirectory()` so we don't pollute the filesystem
- **Reuse from collect_activations.py**: Import `MultiCameraWrapper`, `TASK_TO_PROMPT`, `save_step_activations` to avoid duplication
- **Args**: Use `dataclasses` + `tyro` for consistency with other scripts. Accept `--num_envs_list` (default `1 2 5 10 15`), `--num_inference_calls` (default 3), `--policy.config`, `--policy.dir`

### Output format

```
=== num_envs=2, 3 inference calls (mean) ===

Component                       Time (ms)    %
──────────────────────────────────────────────
Env stepping (10 steps)            XXX      X.X%
Input transforms                   XXX      X.X%
Model inference (total)            XXX      X.X%
  Prefix pass                      XXX      X.X%
  Denoising loop (10 iters)        XXX      X.X%
  Numpy conversion                 XXX      X.X%
Output transforms                  XXX      X.X%
Disk I/O (per env × N envs)       XXX      X.X%
──────────────────────────────────────────────
Total per inference cycle          XXX    100.0%

Throughput: X.X calls/min
Est. 1 task (30 calls): X.X min
Est. 45 tasks × 15 envs: X.X hours
```

Then a final summary table comparing all `num_envs` values.

## Verification

```bash
export CUDA_VISIBLE_DEVICES=1
MUJOCO_GL=egl uv run scripts/benchmark_activations.py \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/ \
    --num_envs_list 1 2 5 10 15
```

Check that:
- All timing rows are non-zero
- Percentages sum to ~100%
- Model inference dominates (expected >90%)
- Larger batch sizes show sublinear scaling in model time
- `uv run ruff check scripts/benchmark_activations.py` passes
