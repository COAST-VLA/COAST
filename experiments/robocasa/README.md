# RoboCasa Steering — Full End-to-End Pipeline

This directory contains the scripts that produce `best_configs.json` — the
per-task `(layer, α, β, strategy)` quadruple consumed by
`examples/robocasa_env/eval_all.py --steer --steering_config ...`.

Same 3-server + 1-build structure as LIBERO, run on one GPU. RoboCasa's
`--seed` is passed to `gym.make(..., seed=seed)` at env construction; the
env's internal RNG then draws different scene configurations per reset.
Different seeds → genuinely different kitchen layouts / object positions, so
collection (seed `0`), sweep (seed `15`), and final eval (seed `30`) sample
disjoint scene distributions.

## Commands

```bash
# (a) Start collection server on GPU 0, port 8200
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output_dir activations --port 8200 \
    policy:checkpoint --policy.config pi05_robocasa \
    --policy.dir checkpoints/pi05_pretrain_human300/multitask_learning/75000

# (b) Collect activations on every task in the task_set: seed=0
cd examples/robocasa_env && MUJOCO_GL=egl uv run python eval_all.py \
    --task_set atomic_seen \
    --num_episodes 15 --seed 0 --collect --port 8200 \
    --num_workers 5 \
    --output_dir /tmp/robocasa_collect_seed0

# (c) Kill the collection server
pkill -f "scripts/serve_policy.py.*port 8200"

# (d) Build the conceptor NPZ (CPU-only, ~5 min). Output path MUST be
#     `conceptors/robocasa_conceptors.npz` — this is the hardcoded path the
#     sweep driver in (e) loads.
CUDA_VISIBLE_DEVICES="" uv run python experiments/robocasa/compute_conceptors.py \
    --activation_root activations \
    --output_path conceptors/robocasa_conceptors.npz

# (e) Sweep hyperparameters: seed=15 → scene draws disjoint from collection.
#     The sweep driver loads the policy itself and starts its own in-process
#     steering server on `--port` (default 8203). Produces best_configs.json.
CUDA_VISIBLE_DEVICES=0 uv run python experiments/robocasa/find_best_configs.py \
    --num_episodes 15 --seed 15

# (f) Start a steering server for the final held-out eval (the sweep driver
#     exited after (e) and took its in-process server with it).
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --pytorch --steer \
    --conceptor_npz conceptors/robocasa_conceptors.npz --port 8201 \
    policy:checkpoint --policy.config pi05_robocasa \
    --policy.dir checkpoints/pi05_pretrain_human300/multitask_learning/75000

# (g) Final held-out eval with per-task tuned configs: seed=30 → another disjoint
#     scene-draw population. Run TWICE — once unsteered for baseline, once steered.
cd examples/robocasa_env && MUJOCO_GL=egl uv run python eval_all.py \
    --task_set atomic_seen \
    --num_episodes 15 --seed 30 --port 8201 \
    --num_workers 5 \
    --output_dir /tmp/robocasa_eval_seed30_baseline

cd examples/robocasa_env && MUJOCO_GL=egl uv run python eval_all.py \
    --task_set atomic_seen \
    --num_episodes 15 --seed 30 --port 8201 \
    --num_workers 5 \
    --steer --steering_config experiments/robocasa/best_configs.json \
    --output_dir /tmp/robocasa_eval_seed30_steered
```

## What each step produces

| Step | Output | Notes |
|------|--------|-------|
| (b) | `activations/75000/<env_name>/episode_NNN_env_000/step_NNNN/*.npz` | 15 eps × N tasks in the task_set |
| (d) | `conceptors/robocasa_conceptors.npz` | `{env_name}__L{L}__{α}__C_{kind}` + per-step + `linear_direction` |
| (f) | `experiments/robocasa/steering_results/<ts>/partial_results.jsonl` + `per_task_results.json` | Streaming SR |
| (f) | `experiments/robocasa/best_configs.json` | Per-task winners |
| (g) | `/tmp/robocasa_eval_seed30/results.json` | Final mean SR per task |

## Customizing the sweep

`find_best_configs.py` Args:

| Flag | Default | Notes |
|------|---------|-------|
| `--tasks`      | 7 atomic_seen envs from the collaborator NPZ | Pass env-names to override |
| `--layers`     | `(11,)` | Which layer(s) to hook |
| `--alphas`     | `(0.1, 0.5, 1.0)` | Conceptor aperture; ignored for `per_step`/`linear` |
| `--betas`      | `(0.1, 0.3)` | Interpolation weight; ignored for `linear` |
| `--strategies` | `(global, per_step, positive_only, random_matched, linear)` | Any subset |
| `--split`      | `pretrain` | RoboCasa split for the underlying tasks |
| `--num_episodes` | 10 | Eps per (task, condition) |
| `--seed`       | 7 | Forwarded to each main.py subprocess; controls RoboCasa scene-RNG seed |

Default grid: 7 tasks × strategy-gated grid × 1 layer + 1 baseline. RoboCasa
env steps ~2× slower than LIBERO — plan ~**8-10 hours** for a full sweep at
`num_episodes=10`.

## Skipping activation collection

If you have a pre-built NPZ, skip (a)-(d):

```bash
hf download brandonyang/robocasa-conceptors robocasa_conceptors.npz \
    --repo-type dataset --local-dir conceptors/
```

Then start at (e). As with LIBERO, the held-out split in (f)/(g) is only
scientifically clean if you know the pre-built NPZ's collection seed and
pick disjoint sweep/eval seeds.

## See also

- `examples/robocasa_env/README.md` — end-user `--steer` flag documentation.
- `src/openpi/serving/steering.py` — the runtime (hooks + wrapper).
- `src/openpi/serving/conceptors.py` — the NPZ builder.
