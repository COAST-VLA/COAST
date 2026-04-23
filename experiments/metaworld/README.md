# MetaWorld Steering — Full End-to-End Pipeline

This directory contains the scripts that produce `best_configs.json` — the
per-task `(layer, α, β, strategy)` quadruple consumed by
`examples/metaworld/eval_all.py --steer --steering_config ...`.

MetaWorld differs from LIBERO/RoboCasa in the collection phase: `--collect`
is **in-process** (no server needed — the script loads the pi0.5 PyTorch
policy directly). Steering itself still requires a WebSocket server
(`--steer` is incompatible with in-process `--collect`), so the eval phases
(sweep + final held-out) use the server-client path.

MetaWorld's `--seed` feeds `env.reset(seed=args.seed + episode)`, so
different base seeds yield different per-episode seeds and thus different
object placements / joint initial angles. Collection (seed `0`), sweep
(seed `15`), and final eval (seed `30`) sample disjoint start distributions.

## Commands

```bash
# (a) Collect activations IN-PROCESS on every ML45 train task: seed=0
#     No server needed — --collect loads the policy from --policy.dir.
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --collect --split train --num_envs 16 --seed 0 \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/openpi-metaworld-5000 \
    --collect_output_dir activations

# (b) Build the conceptor NPZ (CPU-only, ~10 min). Output path MUST be
#     `conceptors/metaworld_conceptors.npz` — this is the hardcoded path the
#     sweep driver in (c) loads. Tasks with <2 failures (all-success at this
#     checkpoint) are skipped with a warning — conceptor steering needs both
#     classes. Pick harder tasks or collect more episodes if too many skip.
CUDA_VISIBLE_DEVICES="" uv run python experiments/metaworld/compute_conceptors.py \
    --activation_root activations \
    --output_path conceptors/metaworld_conceptors.npz

# (c) Sweep hyperparameters: seed=15 → disjoint env seeds vs collection.
#     The sweep driver loads the policy itself and starts its own in-process
#     steering server on `--port` (default 8103). Spawns main.py per
#     (task, condition) via WebSocket to that local server.
CUDA_VISIBLE_DEVICES=0 uv run python experiments/metaworld/find_best_configs.py \
    --num_episodes 10 --num_envs 10 --seed 15

# (d) Start a steering server for the final held-out eval (the sweep driver
#     exited after (c) and took its in-process server with it).
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --pytorch --steer \
    --conceptor_npz conceptors/metaworld_conceptors.npz --port 8301 \
    policy:checkpoint --policy.config pi05_metaworld \
    --policy.dir checkpoints/openpi-metaworld-5000

# (e) Final held-out eval with per-task tuned configs: seed=30 → another disjoint
#     env-seed population. Run TWICE — once unsteered for baseline, once steered.
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --split train --num_episodes 15 --seed 30 --port 8301 \
    --output_dir /tmp/metaworld_eval_seed30_baseline

MUJOCO_GL=egl uv run examples/metaworld/eval_all.py \
    --split train --num_episodes 15 --seed 30 --port 8301 \
    --steer --steering_config experiments/metaworld/best_configs.json \
    --output_dir /tmp/metaworld_eval_seed30_steered
```

## What each step produces

| Step | Output | Notes |
|------|--------|-------|
| (a) | `activations/openpi-metaworld-5000/<env_name>/episode_NNN_env_NNN/step_NNNN/*.npz` | 16 envs × 45 ML45-train tasks |
| (b) | `conceptors/metaworld_conceptors.npz` | `{env_name}__L{L}__{α}__C_{kind}` + per-step + `linear_direction` |
| (d) | `experiments/metaworld/steering_results/<ts>/partial_results.jsonl` + `per_task_results.json` | Streaming SR |
| (d) | `experiments/metaworld/best_configs.json` | Per-task winners |
| (e) | `/tmp/metaworld_eval_seed30/results.json` | Final mean SR per task |

## Customizing the sweep

`find_best_configs.py` Args:

| Flag | Default | Notes |
|------|---------|-------|
| `--tasks`      | 5 representative ML45 train tasks | Pass env-names to override |
| `--layers`     | `(11,)` | Which layer(s) to hook |
| `--alphas`     | `(0.1, 0.5, 1.0)` | Ignored for `per_step`/`linear` |
| `--betas`      | `(0.1, 0.3)` | Ignored for `linear` |
| `--strategies` | `(global, per_step, positive_only, random_matched, linear)` | Any subset |
| `--num_episodes` | 10 | Eps per (task, condition) |
| `--num_envs`   | 10 | In-process AsyncVectorEnv width (keeps policy calls batched) |
| `--max_steps`  | 300 | Per-episode cap |
| `--seed`       | 69420 | Forwarded to main.py; base seed for `env.reset(seed=args.seed + episode)` |

Default grid: 5 tasks × strategy-gated grid × 1 layer + 1 baseline.
MetaWorld batches `num_envs` envs in one process, so wall-clock scales with
conditions not episodes — plan ~**3-4 hours** for a default sweep on one
GPU.

## Skipping activation collection

If you have a pre-built NPZ, skip (a)-(b):

```bash
hf download brandonyang/metaworld-conceptors metaworld_conceptors.npz \
    --repo-type dataset --local-dir conceptors/
```

Then start at (c). As with LIBERO / RoboCasa, the held-out split in (d)/(e)
is only scientifically clean if you know the pre-built NPZ's collection
seed and pick disjoint sweep/eval seeds.

## Steering is WebSocket-only

`--steer` requires a steering-capable server and is incompatible with
in-process `--collect` (which bypasses `SteeredPolicyWrapper` entirely by
design). If you need both, do collection first (step a), then start the
steering server (step c), then run the WebSocket-based sweep / eval
(steps d-e).

## See also

- `examples/metaworld/README.md` — end-user `--steer` flag documentation.
- `src/openpi/serving/steering.py` — the runtime (hooks + wrapper).
- `src/openpi/serving/conceptors.py` — the NPZ builder.
