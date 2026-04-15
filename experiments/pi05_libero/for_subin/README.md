# pi0.5 LIBERO Conceptor Steering — Handoff Bundle

This directory contains a standalone pipeline for running conceptor-based
steering experiments on a pi0.5 policy evaluated on LIBERO-10. It's the
packaged version of the same pipeline used for the GR00T N1.5 / RoboCasa and
pi0.5 / RoboCasa experiments, re-tuned for pi0.5's LIBERO activation schema.

## TL;DR

```bash
# 0. Pre-req: activations already collected and on disk at
#    $OPENPI_DATA_HOME/activations/pi05_libero_2000_15env/openpi-libero-2000/
# 1. Build conceptors.
uv run experiments/pi05_libero/for_subin/build_conceptors.py
# 2. (Optional) run diagnostics / narrow the sweep.
uv run experiments/pi05_libero/for_subin/diagnostic.py
uv run experiments/pi05_libero/for_subin/select_parameters.py \
    --conceptor-npz $OPENPI_DATA_HOME/libero_conceptors.npz \
    --output-json experiments/pi05_libero/for_subin/selected_params.json
# 3. Run the full steering sweep (one SLURM job per task).
# Edit CHECKPOINT_DIR in run_steering.sh first.
bash experiments/pi05_libero/for_subin/run_steering.sh
```

## Files

| File | What it does |
|------|-------------|
| `build_conceptors.py` | Step 1. Reads LIBERO suffix-residual activations, computes per-task success/failure/contrastive conceptors on a (layer × α) grid plus linear-direction vectors. Per-denoising-step conceptors are built at **all 10 denoising steps** (not just 0 and 9). Writes `$OPENPI_DATA_HOME/libero_conceptors.npz`. |
| `diagnostic.py` | Step 2a. Plots of eigenvalue spectra, pairwise conceptor similarity, AND/NOT boolean ops, linear probes (task-id and success/failure), and quota-vs-α across layers. Use for sanity-checking conceptors and picking layers/α. |
| `select_parameters.py` | Step 2b. Automated narrow-sweep picker. Reads the npz, picks the best layer by mean quota, picks α's whose success/failure overlap lands in the sweet-spot band `[0.85, 0.95]`, and writes a JSON you can wire into the steering launcher. |
| `conceptor_steering.py` | Step 3. Runs ONE task's full steering sweep end-to-end. Loads pi0.5 once, starts a WebSocket policy server in a daemon thread, and sequentially runs one LIBERO-client subprocess per condition. Supports five strategies — see below. |
| `run_steering.sh` | SLURM launcher. Fans out one sbatch per LIBERO-10 task, each job runs the full strategy grid against one loaded policy. |

## Input data — LIBERO activations

`build_conceptors.py` expects the activation cache at
`$OPENPI_DATA_HOME/activations/pi05_libero_2000_15env/openpi-libero-2000/`.
Schema (produced by the pi0.5 collection server, one directory per episode, one
subdirectory per inference step):

```
<task>/episode_XXX_env_XXX/
├── metadata.json                              # episode-level: episode_success, total_inference_steps, …
└── step_YYYY/
    └── suffix_residual.npz
        ├── key:    "all_suffix_residual"
        ├── shape:  (10 denoising_steps, 4 captured_layers, 10 action_tokens, 1024 hidden)
        └── layers captured at:  [0, 5, 11, 17]   (pi0.5 suffix-model layer indices)
```

`build_conceptors.py` mean-pools over the 10 action tokens so each inference
step contributes **one 1024-dim vector** per (layer, denoising_step). It filters
to mixed-outcome tasks (≥ 3 successes AND ≥ 3 failures) — contrastive
conceptors need both classes.

If you need to pull the reference activation dataset:
```bash
huggingface-cli download brandonyang/pi05-libero-activations-v1-2000-15env \
    --repo-type dataset \
    --local-dir $OPENPI_DATA_HOME/activations/pi05_libero_2000_15env
```

## The five steering strategies

All five plug into `conceptor_steering.py --strategies ...` and can be run
together or individually. They all apply their intervention via a PyTorch
forward hook on the suffix-model transformer layer selected by `--layers`.

| Strategy | What it applies | Sweep axes | Role |
|----------|----------------|-----------|------|
| `linear` | `h' = h + α · v`, where `v = unit(mean_success − mean_failure)` | `layer × linear_alpha` | ActAdd-style control — simplest possible success/failure probe. |
| `global` | `h' = (1−β) h + β (h @ Cᵀ)`, where `C = C_success · (I − C_failure)` | `layer × α × β` | Main experiment — one conceptor applied every denoising step. |
| `per_step` | Same as `global` but **a different conceptor at each denoising step 0..9** | `layer × β` (α is baked into the npz) | Tests whether the optimal steering direction drifts through the flow-matching trajectory. |
| `positive_only` | `h' = (1−β) h + β (h @ C_sᵀ)` — uses `C_success` directly, no `NOT C_failure` | `layer × α × β` | Ablation: does the NOT term matter, or is projecting onto the success subspace enough? |
| `random` | Random SPD matrix with matched quota | `layer × β` | Control — isolates the effect of structure vs. any random rotation at matched trace. |

**Per-step mechanism.** Each hook has a `set_denoise_step(t)` method. The
policy's `infer_with_steering` calls this at the top of each denoising
iteration so the hook knows which of the 10 matrices/vectors to apply. For
`global`, `linear`, `positive_only`, and `random` the counter is ignored
(they're single-matrix specs); only `per_step` / `linear_per_step` consume it.

## Output layout

```
experiments/pi05_libero/steering_results/
├── <task_short_name>/
│   ├── sweep_args.json                    # full CLI args for reproducibility
│   ├── summary.json                       # condition → success_rate, sorted
│   ├── baseline/                          # per-condition subdirs (per-condition client logs)
│   ├── global_L11_a1.0_b0.1/
│   ├── per_step_L11_b0.1/
│   ├── linear_L11_la0.5/
│   ├── posonly_L11_a1.0_b0.1/
│   └── random_L11_b0.1/
└── logs/                                  # SLURM stdout/stderr
```

`summary.json` is resume-friendly: if a job dies partway through, just
resubmit and already-completed conditions are skipped (it merges from disk
before writing, so concurrent writers don't clobber each other).

## Knobs for a generous sweep

Defaults in `run_steering.sh` assume you have GPU budget:

- `layers = [5, 11, 17]` — the three deepest captured suffix layers.
- `alphas = [0.1, 0.5, 1.0, 2.0, 10.0]` — full aperture range.
- `betas = [0.1, 0.3]` — β = 0.5 was universally harmful in prior pi0.5
  experiments; keep it off unless you have a reason.
- `linear_alphas = [0.1, 0.5, 1.0]` — ActAdd scale.
- All five strategies.

Per task this is ~ `baseline + 9 linear + 30 global + 6 per_step + 30 posonly
+ 6 random ≈ 82 conditions`. At 15 episodes × ~30 s per episode that's
roughly 10 hours/task on one GPU. Scale with your available GPUs by reducing
`--num-episodes` for iteration or dropping strategies with `--strategies`.

## Integrating with your openpi fork

`conceptor_steering.py` imports:

```python
from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.serving import websocket_policy_server
```

These are part of the openpi PyTorch path and are already exposed in this
repo's `pi05_libero` branch. If your fork has drifted, you may need to:

1. Verify `policy.infer_with_steering(obs, steering_hooks=[(layer, hook)])`
   exists on the returned Pi0 policy. This is a custom method on the pi0.5
   PyTorch wrapper that registers forward hooks, runs `sample_actions`, then
   unregisters.
2. Verify the denoising loop inside `sample_actions_with_intermediates`
   calls `hook.set_denoise_step(t)` at the top of each iteration. If it
   doesn't, per_step / linear_per_step will reduce to the t=0 conceptor for
   every step. Grep your fork for `set_denoise_step` — if it's only defined
   and never called, patch the sampler.
3. Confirm the LIBERO client venv lives at
   `examples/libero_env/.venv/bin/python` and that `main.py` accepts
   `--task_suite_name`, `--task_id`, `--num_episodes`, `--port`,
   `--output_dir`. If the paths differ, edit `run_single_task_eval`.

## Task registry

LIBERO-10 tasks are referenced by their full scene-and-description name
throughout. The name → benchmark `task_id` mapping is hard-coded in
`conceptor_steering.py::LIBERO_TASK_IDS` and `run_steering.sh::TASKS`. Must
stay in sync with the `libero_10` benchmark suite — if you add more tasks or
rename, update both.

## Common gotchas

- **"success_rate not found in client output"** — the client exited 0 but
  `main.py` didn't print `success_rate=...`. Usually a MUJOCO_GL / EGL
  rendering problem; make sure `MUJOCO_GL=egl` is exported and the node has
  a GPU visible. The server-side run is fine — the issue is the LIBERO env
  failing to render.
- **Per-step conceptors all look identical to global** — your openpi fork's
  sampler isn't calling `hook.set_denoise_step(t)`. See item 2 above.
- **build_conceptors.py skips all tasks** — all tasks in your collection are
  either all-success or all-failure (`MIN_PER_CLASS = 3` on both sides).
  This usually means either the policy is too strong/weak or episodes
  weren't labelled correctly. Check `metadata.json["episode_success"]`.
- **npz file is ~1 GB for LIBERO-10** — normal with all 10 denoising steps ×
  4 layers × 5 α's × 3 matrices per task. Pass `--skip-per-step` to cut it
  roughly in half if you only need the `global` strategy.
