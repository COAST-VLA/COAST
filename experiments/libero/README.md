# LIBERO-10 Steering — Full End-to-End Pipeline

This directory contains the scripts that produce `best_configs.json` — the
per-task `(layer, α, β, strategy)` quadruple consumed by
`examples/libero_env/eval_all.py --steer --steering_config ...`.

The pipeline is **3 server phases + 1 build phase**, run on one GPU, with
three disjoint `--seed` values chosen so collection, sweep, and final eval
operate on disjoint LIBERO initial-state slots. Under PR #48's seeding
semantics, episode `k` pulls `initial_states[(seed + k) % N]`, so seeds
`0 / 15 / 30` with `--num_episodes 15` give three non-overlapping 15-state
windows on every libero_10 task (N = 50 canonical states per task).

## One-liner (wraps all commands below)

```bash
bash experiments/libero/run_end_to_end.sh
# or override the defaults via env vars:
#   GPU=1 NUM_EPISODES=10 SEED_COLLECT=0 SEED_SWEEP=15 SEED_EVAL=30 \
#       bash experiments/libero/run_end_to_end.sh
```

Logs land in `experiments/libero/run_logs/`. Runs the 7 stages below sequentially and prints a final baseline-vs-steered SR line. Read on for the stage-by-stage commands the script wraps.

## Commands

```bash
# (a) Start collection server on GPU 0, port 8100
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output_dir activations/libero --port 8100 \
    policy:checkpoint --policy.config pi05_libero \
    --policy.dir checkpoints/openpi-libero-2000

# (b) Collect activations on every libero_10 task: seed=0 → init-state slots 0..14.
#     The server's --output_dir is the authoritative sink; the client's per-rollout
#     videos default under examples/libero_env/output/ (see examples/libero_env/README.md).
cd examples/libero_env && MUJOCO_GL=egl uv run python eval_all.py \
    --task_suite_name libero_10 \
    --num_episodes 15 --seed 0 --collect --port 8100 \
    --num_workers 5

# (c) Kill the collection server
pkill -f "scripts/serve_policy.py.*port 8100"

# (d) Build the conceptor NPZ from the collected activations (CPU-only, ~5 min).
#     Output path MUST be `conceptors/libero_conceptors.npz` — this is the
#     hardcoded path the sweep driver in (e) loads.
CUDA_VISIBLE_DEVICES="" uv run python experiments/libero/compute_conceptors.py \
    --activation_root activations/libero \
    --output_path conceptors/libero_conceptors.npz

# (e) Sweep hyperparameters: seed=15 → init-state slots 15..29 (DISJOINT from
#     collection). The sweep driver loads the policy itself and starts its own
#     in-process steering server on `--port` (default 8003) — no separate
#     serve_policy.py invocation needed. Produces best_configs.json.
CUDA_VISIBLE_DEVICES=0 uv run python experiments/libero/find_best_configs.py \
    --num_episodes 15 --seed 15

# (f) Start a steering server for the final held-out eval (the sweep driver
#     exited after (e) and took its in-process server with it).
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --pytorch --steer \
    --conceptor_npz conceptors/libero_conceptors.npz --port 8101 \
    policy:checkpoint --policy.config pi05_libero \
    --policy.dir checkpoints/openpi-libero-2000

# (g) Final held-out eval with per-task tuned configs: seed=30 → slots 30..44
#     (disjoint from both (b) and (e)). Run TWICE — once without --steer for
#     baseline, once with the --steer / --steering_config pair.
#     Videos land under the default examples/libero_env/output/ tree.
cd examples/libero_env && MUJOCO_GL=egl uv run python eval_all.py \
    --task_suite_name libero_10 \
    --num_episodes 15 --seed 30 --port 8101 \
    --num_workers 5

cd examples/libero_env && MUJOCO_GL=egl uv run python eval_all.py \
    --task_suite_name libero_10 \
    --num_episodes 15 --seed 30 --port 8101 \
    --num_workers 5 \
    --steer --steering_config experiments/libero/best_configs.json
```

## What each step produces

| Step | Output | Notes |
|------|--------|-------|
| (b) | `activations/libero/openpi-libero-2000/<task>/episode_NNN_env_000/step_NNNN/*.npz` | Per-step intermediates for each of 10 tasks × 15 eps |
| (d) | `conceptors/libero_conceptors.npz` | ~2k keys: `{task}__L{L}__{α}__C_{kind}` + per-step + `linear_direction` |
| (f) | `experiments/libero/steering_results/<ts>/partial_results.jsonl` + `per_task_results.json` | Streaming per-condition SR |
| (f) | `experiments/libero/best_configs.json` | Per-task `(layer, α, β, strategy)` + baseline and steered SR |
| (g) | `examples/libero_env/output/libero_10/results.json` | Final mean SR per task (rewritten by each of the two runs; copy between invocations to retain both) |

## Customizing the sweep

`find_best_configs.py` Args (selected):

| Flag | Default | Notes |
|------|---------|-------|
| `--tasks`      | all 10 libero_10 tasks | Restrict to a subset |
| `--layers`     | `(11,)` | Which transformer layer(s) to hook |
| `--alphas`     | `(0.1, 0.5, 1.0)` | Ignored for `per_step` (baked in as 1.0) and `linear` |
| `--betas`      | `(0.1, 0.3)` | Ignored for `linear` (baked in as 0.0) |
| `--strategies` | `(global, per_step, positive_only, random_matched, linear)` | Drop any to shrink grid |
| `--num_episodes` | 10 | Eps per (task, condition) |
| `--seed`       | 7 | Forwarded to each main.py subprocess; controls init-state window |

Default grid: 10 tasks × strategy-gated grid × 1 layer + 1 baseline =
**190 eval runs × 10 eps**. ~2 min/eval → **~6 hours** wall-clock on one GPU.

## Skipping activation collection

If you trust a pre-built NPZ (e.g. the published checkpoint's), skip (a)-(d):

```bash
hf download brandonyang/libero-conceptors libero_conceptors.npz \
    --repo-type dataset --local-dir conceptors/
```

Then jump straight to (e). Note: the held-out split in (f)/(g) is only
meaningful if you know the seed used for the NPZ's underlying collection —
pick a sweep/eval seed disjoint from it. When in doubt, re-collect (steps
(a)-(d)) yourself for scientifically clean results.

## Notes

- **Partial sweep recovery.** If a sweep crashes partway, the partial
  JSONL at `experiments/libero/steering_results/<ts>/partial_results.jsonl`
  is valid — re-aggregate by grouping on task, picking argmax steered SR
  per task, and emitting `best_configs.json` from the survivors.
- **Old NPZs may be missing per-step keys 1-8.** The current
  `DEFAULT_PER_STEP_INDICES` is all 10 denoising steps, but NPZs built
  before that change have only `per_step_0` / `per_step_9` → `per_step`
  strategy will NaN. Rebuild via step (d) if you hit this.

## See also

- `examples/libero_env/README.md` — end-user `--steer` flag documentation.
- `src/openpi/serving/steering.py` — the runtime (hooks + wrapper).
- `src/openpi/serving/conceptors.py` — the NPZ builder.
