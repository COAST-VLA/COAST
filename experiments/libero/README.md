# LIBERO-10 Steering Sweep

This directory produces `best_configs.json` — the tuned per-task
`(layer, α, β, strategy)` quadruple to use with
`examples/libero_env/eval_all.py --steer --steering_config experiments/libero/best_configs.json`.

**This README is for researchers reproducing the sweep.** If you just want to
run a steered eval, see `examples/libero_env/README.md`.

## What this produces

`best_configs.json` — one entry per LIBERO-10 task with the `(layer, α, β, strategy)`
tuple that maximized success rate over the sweep, plus the raw baseline /
steered success rates for context:

```json
{
  "task_suite": "libero_10",
  "source_sweep": "experiments/libero/steering_results/2026-04-14_180000/",
  "generated_at": "2026-04-14T18:00:00Z",
  "num_episodes_per_condition": 10,
  "defaults": {"layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global"},
  "tasks": {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": {
      "layer": 11, "alpha": 0.1, "beta": 0.3, "strategy": "global",
      "baseline_sr": 0.60, "steered_sr": 1.00
    }
  }
}
```

## Prereqs

1. **Conceptor NPZ**:

   ```bash
   hf download brandonyang/libero-conceptors libero_conceptors.npz \
       --repo-type dataset --local-dir conceptors/
   ```

   This NPZ is the canonical source. If you need to rebuild it from fresh
   activations (e.g., a new checkpoint), see the "Rebuilding the conceptor NPZ"
   section near the bottom of this file.

2. **GPU**: one GPU — inference only.

   ```bash
   nvidia-smi
   export CUDA_VISIBLE_DEVICES=<lowest-util-id>
   ```

3. **PyTorch checkpoint**: pi0.5 LIBERO checkpoint at
   `checkpoints/openpi-libero-2000` (the first sweep invocation auto-converts
   from JAX if needed).

4. **LIBERO sub-venv**: `examples/libero_env/.venv` must exist (it's the
   Python 3.8 LIBERO / robosuite env). Build it from
   `examples/libero_env/pyproject.toml` if missing.

## Running the sweep

```bash
CUDA_VISIBLE_DEVICES=0 uv run python experiments/libero/find_best_configs.py
```

Default grid: 10 tasks × (3 strategies × 3 alphas × 2 betas × 1 layer) + 1 baseline
= 190 eval runs × 10 episodes. At ~2 min/eval (LIBERO-10 long-horizon), plan
for **~6 hours** wall-clock on one GPU.

Partial results stream to
`experiments/libero/steering_results/<timestamp>/partial_results.jsonl`
(append-only, one JSON record per completed condition). The script also saves
`per_task_results.json` after every completed task. Neither file is committed.

## Customizing the sweep

```bash
# Subset of tasks
uv run python experiments/libero/find_best_configs.py \
    --tasks KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it

# Different grid
uv run python experiments/libero/find_best_configs.py \
    --layers 5 11 17 --alphas 0.1 1.0 --betas 0.3 --strategies global per_step

# Fewer episodes for a fast smoke run
uv run python experiments/libero/find_best_configs.py --num_episodes 5 \
    --tasks KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
```

## Interpreting outputs

- `best_configs.json` is the canonical output and is committed to the repo.
  `baseline_sr` / `steered_sr` are informational — they show *why* a config was
  chosen but are not consumed by the server or `eval_all.py`.
- `steering_results/<timestamp>/` holds the raw per-task, per-condition runs
  and is gitignored.
- Server-side logs live inside each per-condition video output dir.

## After the sweep

Verify the winners reproduce under `eval_all.py`:

```bash
# Terminal 1 — steering-aware server
uv run scripts/serve_policy.py --pytorch --steer \
    --conceptor_npz conceptors/libero_conceptors.npz \
    policy:checkpoint \
    --policy.config pi05_libero --policy.dir checkpoints/openpi-libero-2000

# Terminal 2 — eval all tasks using the tuned configs
cd examples/libero_env
MUJOCO_GL=egl uv run eval_all.py \
    --task_suite_name libero_10 --num_episodes 10 \
    --steer --steering_config ../../experiments/libero/best_configs.json
```

If the mean success rate matches the sum of per-task `steered_sr` values in
`best_configs.json` (within ±0.05 run-to-run variance), the sweep is valid.

## Rebuilding the conceptor NPZ (advanced)

The shipped `conceptors/libero_conceptors.npz` from HuggingFace is canonical.
We keep `compute_conceptors.py` here for reproducibility and for rebuilding
when the checkpoint changes. Normal sweep workflow (above) never needs it.

**Pipeline**:

1. Start a collection-mode server (writes per-step hidden states to disk):

   ```bash
   uv run scripts/serve_policy.py --pytorch --collect_activations \
       --output_dir activations \
       policy:checkpoint --policy.config pi05_libero \
       --policy.dir checkpoints/openpi-libero-2000
   ```

2. Run rollouts with `--collect` on as many episodes as you can afford (more
   episodes → better-conditioned correlation matrices → closer match to the
   HF reference):

   ```bash
   cd examples/libero_env
   MUJOCO_GL=egl uv run python eval_all.py \
       --task_suite_name libero_10 --num_episodes 25 --collect
   ```

3. Compute conceptors from the collected activations:

   ```bash
   uv run python experiments/libero/compute_conceptors.py \
       --activation_root activations/openpi-libero-2000 \
       --output_path conceptors/libero_conceptors_fresh.npz
   ```

The output NPZ uses the same key format as the HF reference (`{task}__L{layer}__{alpha|per_step_N}__{C_success|C_failure|C_contrastive}`)
and is drop-in usable: pass `--conceptor_npz conceptors/libero_conceptors_fresh.npz`
to `serve_policy.py --steer`. All math lives in `src/openpi/serving/conceptors.py`.
