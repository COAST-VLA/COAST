# pi0.5 DROID Conceptor Steering

Real-robot counterpart to `experiments/pi05_libero/for_subin/`. Same pipeline
(build contrastive conceptors → select parameters → apply steering at
inference) applied to pi0.5 DROID activations. The differences versus LIBERO:

- **Real robot, no simulator.** There is no LIBERO-style client subprocess or
  automated sweep — `conceptor_steering.py` serves ONE steering condition at
  a time and the human operator runs the physical DROID client against it.
- **Action token count is 15** (LIBERO uses 10). `build_conceptors.py`
  mean-pools across the sequence axis, so this is handled transparently.
- **Activations default to `./activations/pi05_droid/`** — the directory
  written by the collection server below.

## TL;DR

```bash
# 0. Collect activations on the real robot (server + physical client).
#    Start this from the repo root with a free GPU visible.
uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi05_droid \
    --policy.dir="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid"
# (run your DROID client separately — episodes land in ./activations/pi05_droid/<task>/ )

# 1. Build conceptors from the collected activations.
uv run experiments/pi05_droid/build_conceptors.py

uv run experiments/pi05_droid/build_conceptors.py --min-step 300      

# 2. (Optional) pick the best layer / alpha band.
uv run experiments/pi05_droid/select_parameters.py \
    --conceptor-npz "${OPENPI_DATA_HOME:-$HOME/.cache/openpi}/droid_conceptors.npz" \
    --output-json  experiments/pi05_droid/selected_params.json

# 3. Serve the policy with ONE steering condition applied.
#    Stop (Ctrl-C) and rerun with different knobs to sweep.
# per_step
CUDA_VISIBLE_DEVICES=0 STRATEGY=global LAYER=11 ALPHA=0.1 BETA=0.1 \
    bash experiments/pi05_droid/run_steering.sh
```

## Files

| File | What it does |
|------|-------------|
| `build_conceptors.py` | Reads DROID suffix-residual activations, computes per-task success/failure/contrastive conceptors on a (layer × α) grid plus linear-direction vectors. Per-denoising-step conceptors at all 10 denoising steps. Writes `$OPENPI_DATA_HOME/droid_conceptors.npz`. |
| `select_parameters.py` | Picks the best layer by mean quota and alphas whose success/failure overlap lands in the sweet-spot band `[0.85, 0.95]`. Writes a JSON of selected parameters. |
| `conceptor_steering.py` | Loads pi0.5 DROID, installs a forward hook for one strategy/layer/α/β/linear-α combination, and serves the steered policy over WebSocket for the physical robot client. Blocks until Ctrl-C. |
| `run_steering.sh` | Thin wrapper for `conceptor_steering.py` — all knobs passed via env vars so it's easy to relaunch per condition. |

## Input data — DROID activations

`build_conceptors.py` expects the activation cache at
`./activations/pi05_droid/` (relative to repo root), as produced by the
collection server command shown above. Schema:

```
<task>/episode_XXX_env_XXX_<timestamp>/
├── metadata.json                              # episode_success, total_inference_steps, prompt, …
├── rewards.npz
└── step_YYYY/
    ├── suffix_residual.npz
    │   ├── key:    "all_suffix_residual"
    │   └── shape:  (10 denoising_steps, 4 captured_layers, 15 action_tokens, 1024 hidden)
    ├── suffix_mlp_hidden.npz
    ├── denoising.npz
    └── adarms_cond.npz
```

Captured suffix-model layer indices: `[0, 5, 11, 17]` (same as pi0.5 LIBERO).

`build_conceptors.py` filters to tasks with ≥ 3 success AND ≥ 3 failure
episodes (`MIN_PER_CLASS`). Episode success is read from
`metadata.json["episode_success"]` — if your collection run didn't label
successes properly, no conceptors will be built.

## The five steering strategies

Same semantics as the LIBERO version — the hooks are identical.

| Strategy | What it applies | Knobs | Role |
|----------|-----------------|-------|------|
| `baseline` | No steering. | — | Parity with raw `serve_policy.py`. |
| `linear` | `h' = h + α · v`, `v = unit(mean_s − mean_f)` | `layer`, `linear_alpha` | ActAdd-style control. |
| `global` | `h' = (1−β) h + β (h @ C^T)`, `C = C_s · (I − C_f)` | `layer`, `alpha`, `beta` | Main experiment. |
| `per_step` | Same as `global`, different conceptor each denoising step 0..9. | `layer`, `beta` (α baked into npz) | Tests drift along the flow-matching trajectory. |
| `positive_only` | `h' = (1−β) h + β (h @ C_s^T)` — no NOT term | `layer`, `alpha`, `beta` | Ablation of the contrastive term. |
| `random` | Random SPD matrix with matched quota | `layer`, `alpha`, `beta`, `random_seed` | Control — isolates structure vs. random rotation. |

Per-step mechanism: each hook exposes `set_denoise_step(t)`; the pi0.5
sampler calls it at the top of each denoising iteration so `per_step` swaps
the correct matrix. See the "Common gotchas" section in
`../pi05_libero/for_subin/README.md` for how to verify this in a fork.

## Integration notes

`conceptor_steering.py` imports the same openpi paths as the LIBERO version:

```python
from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.serving import websocket_policy_server
```

Required:

1. `policy.infer_with_steering(obs, steering_hooks=[(layer, hook)])` exists on
   the returned pi0.5 PyTorch policy.
2. The denoising loop inside `sample_actions_with_intermediates` calls
   `hook.set_denoise_step(t)` at the top of each iteration. Without this,
   `per_step` reduces to the t=0 conceptor on every step.
3. The DROID client knows how to connect to `ws://<host>:<port>` — same
   protocol as `scripts/serve_policy.py`.

## Common gotchas

- **`build_conceptors.py` skips all tasks.** All episodes are labelled success
  or all failure. Check `metadata.json["episode_success"]` in a few episodes
  and make sure your collection client sets it correctly for DROID rollouts.
- **Per-step conceptors all look identical to global.** Your fork's sampler
  isn't calling `hook.set_denoise_step(t)`. Patch the sampler to call it at
  the top of each denoising iteration.
- **npz size.** With only one DROID task this is small (~ a few hundred MB).
  Pass `--skip-per-step` to cut it in half if you only need `global`.
