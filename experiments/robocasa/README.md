# RoboCasa Steering Sweep

Produces `best_configs.json` with per-env `(layer, α, β, strategy)` tuples for
use with `examples/robocasa_env/eval_all.py --steer --steering_config
experiments/robocasa/best_configs.json`.

**For researchers reproducing the sweep.** For running a steered eval, see
`examples/robocasa_env/README.md`.

## What this produces

One entry per RoboCasa env name keyed under `"tasks"`. RoboCasa uses the short
env name (e.g., `CloseFridge`) as both the conceptor NPZ key and the
`--env_name` flag — no translation needed.

```json
{
  "task_suite": "robocasa_pretrain",
  "tasks": {
    "CloseFridge": {
      "layer": 11, "alpha": 0.5, "beta": 0.1, "strategy": "per_step_0",
      "baseline_sr": 0.40, "steered_sr": 0.80
    }
  }
}
```

## Prereqs

1. **Conceptor NPZ**:

   ```bash
   hf download brandonyang/robocasa-conceptors robocasa_conceptors.npz \
       --repo-type dataset --local-dir conceptors/
   ```

2. **GPU**: one GPU.

3. **PyTorch checkpoint**: pi0.5 RoboCasa checkpoint at
   `checkpoints/pi05_pretrain_human300/multitask_learning/75000`.

4. **RoboCasa sub-venv**: `examples/robocasa_env/.venv` (Python 3.8).

## Running the sweep

```bash
CUDA_VISIBLE_DEVICES=0 uv run python experiments/robocasa/find_best_configs.py
```

**Time estimate**: RoboCasa envs step ~2x slower than LIBERO. Default grid is
7 envs × 19 conditions × 10 episodes ≈ **~8 hours**.

The sweep can be resumed from partial failure by inspecting
`experiments/robocasa/steering_results/<timestamp>/partial_results.jsonl` and
re-running with `--tasks` restricted to remaining envs.

## Customizing

```bash
uv run python experiments/robocasa/find_best_configs.py \
    --tasks CloseFridge OpenDrawer \
    --alphas 0.1 1.0 --betas 0.1 0.3 --strategies global per_step_0 \
    --num_episodes 5
```

## After the sweep

```bash
# Server
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config pi05_robocasa \
    --policy.dir checkpoints/pi05_pretrain_human300/multitask_learning/75000 \
    --env ROBOCASA --pytorch --steer

# Eval
cd examples/robocasa_env
MUJOCO_GL=egl uv run eval_all.py \
    --task_set atomic_seen --num_episodes 10 \
    --steer --steering_config ../../experiments/robocasa/best_configs.json
```

## Rebuilding the conceptor NPZ (advanced)

The shipped `conceptors/robocasa_conceptors.npz` from HuggingFace is canonical.
`compute_conceptors.py` in this directory can rebuild it from scratch if you
change the checkpoint; normal sweep workflow never needs it.

Pipeline is identical to LIBERO's (see `experiments/libero/README.md` →
"Rebuilding the conceptor NPZ"), with the RoboCasa-specific CLI:

```bash
# 1. Collection server
uv run scripts/serve_policy.py --env ROBOCASA --pytorch --collect_activations \
    --output_dir activations \
    policy:checkpoint --policy.config pi05_robocasa \
    --policy.dir checkpoints/pi05_pretrain_human300/multitask_learning/75000

# 2. Roll out with --collect
cd examples/robocasa_env
MUJOCO_GL=egl uv run eval_all.py --task_set atomic_seen --num_episodes 25 --collect

# 3. Compute
uv run python experiments/robocasa/compute_conceptors.py \
    --activation_root activations/75000 \
    --output_path conceptors/robocasa_conceptors_fresh.npz
```

All math lives in `src/openpi/serving/conceptors.py` (shared with LIBERO).
