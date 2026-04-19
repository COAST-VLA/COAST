# MetaWorld ML45 Steering Sweep

This directory produces `best_configs.json` — the tuned per-task
`(layer, α, β, strategy)` quadruple to use with
`examples/metaworld/eval_all.py --steer --steering_config experiments/metaworld/best_configs.json`.

**This README is for researchers reproducing the sweep.** If you just want
to run a steered eval, see `examples/metaworld/README.md`.

## What this produces

`best_configs.json` — one entry per MetaWorld env_name with the
`(layer, α, β, strategy)` tuple that maximized success rate over the sweep,
plus raw baseline / steered success rates for context.

## Prereqs

1. **Conceptor NPZ**:

   ```bash
   hf download brandonyang/metaworld-conceptors metaworld_conceptors.npz \
       --repo-type dataset --local-dir conceptors/
   ```

   This NPZ is the canonical source. If you need to rebuild from fresh
   activations (e.g., a new checkpoint), see "Rebuilding the conceptor NPZ"
   below.

2. **GPU**: one GPU — inference only.

   ```bash
   nvidia-smi
   export CUDA_VISIBLE_DEVICES=<lowest-util-id>
   ```

3. **PyTorch checkpoint**: pi0.5 MetaWorld checkpoint at
   `checkpoints/openpi-metaworld-5000` (the first invocation auto-converts
   from JAX if needed).

4. **Root venv only** — unlike LIBERO, MetaWorld has no sub-venv; everything
   runs from the repo-root `uv run`.

## Running the sweep

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python experiments/metaworld/find_best_configs.py
```

Default grid: 5 tasks × (6 strategies × 3 alphas × 2 betas × 1 layer) + 1
baseline = 185 eval runs × 10 episodes. MetaWorld stepping is fast
(`AsyncVectorEnv` with 10 parallel envs per task), so plan for **~2–3 hours**
wall-clock on one GPU.

Partial results stream to
`experiments/metaworld/steering_results/<timestamp>/partial_results.jsonl`
and `per_task_results.json` updates after every task. Neither file is
committed.

## Customizing the sweep

```bash
# Subset of tasks
uv run python experiments/metaworld/find_best_configs.py --tasks reach-v3 pick-place-v3

# Different grid
uv run python experiments/metaworld/find_best_configs.py \
    --layers 5 11 17 --alphas 0.1 1.0 --betas 0.3 --strategies global per_step

# Fast smoke run
uv run python experiments/metaworld/find_best_configs.py --num_episodes 3 \
    --tasks reach-v3
```

## Interpreting outputs

- `best_configs.json` is the canonical output and is committed.
  `baseline_sr` / `steered_sr` are informational.
- `steering_results/<timestamp>/` holds raw per-condition runs; gitignored.
- Sweep driver logs the whole grid to stdout.

## After the sweep

```bash
# Terminal 1 — steering-aware server
uv run scripts/serve_policy.py --env METAWORLD --pytorch --steer \
    --conceptor_npz conceptors/metaworld_conceptors.npz \
    policy:checkpoint \
    --policy.config pi05_metaworld --policy.dir checkpoints/openpi-metaworld-5000

# Terminal 2 — eval all tasks using the tuned configs
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --split train --num_episodes 10 \
    --steer --steering_config experiments/metaworld/best_configs.json
```

## Rebuilding the conceptor NPZ (advanced)

MetaWorld collection is **in-process** (unlike LIBERO/RoboCasa's server-side
collection), but the on-disk layout is identical.

1. Collect activations:

   ```bash
   CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
       --collect --split train --num_envs 16 --num_episodes 10 \
       --policy.config=pi05_metaworld \
       --policy.dir=checkpoints/openpi-metaworld-5000
   ```

2. Compute conceptors:

   ```bash
   uv run python experiments/metaworld/compute_conceptors.py \
       --activation_root activations/openpi-metaworld-5000 \
       --output_path conceptors/metaworld_conceptors_fresh.npz
   ```

The output NPZ uses the same key format as LIBERO/RoboCasa
(`{task}__L{layer}__{alpha|per_step_N}__{C_success|C_failure|C_contrastive}`)
and is drop-in usable with `serve_policy.py --steer`.
