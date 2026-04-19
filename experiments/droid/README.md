# DROID Steering (Real Robot)

DROID is a real-robot evaluation harness — there is **no simulator**, and
success is labeled by the human operator after each rollout. That rules out
the automated sweep pattern LIBERO / RoboCasa / MetaWorld use (those produce
`best_configs.json` via a subprocess grid search). DROID's tuning tool is
`select_parameters.py`: a diagnostic-based narrower that picks a short list
of promising `(layer, α, β)` values from the conceptor NPZ alone, and the
operator evaluates each condition manually.

**This README is for researchers running real-robot eval.** If you just want
to understand the `--steer` client flag, see `examples/droid/README.md`.

## What this produces

`selected_params.json` — a short list of `(best_layer, selected_alphas, selected_betas)`
picked by two diagnostics:

1. **Layer** — highest mean quota `tr(C)/d` across tasks at `α = 10.0`.
2. **Alphas** — α values whose mean success/failure overlap
   `tr(C_s C_f) / sqrt(tr(C_s²) tr(C_f²))` falls in the band `[0.85, 0.95]`
   (empirically the sweet spot where success and failure subspaces are
   distinguishable but not disjoint).

Unlike LIBERO's `best_configs.json`, this file is NOT a per-task best — it's
a shortlist you then evaluate by hand on the real robot.

## Prereqs

1. **Conceptor NPZ**:

   ```bash
   hf download brandonyang/droid-conceptors droid_conceptors.npz \
       --repo-type dataset --local-dir conceptors/
   ```

2. **DROID hardware**: Panda arm, 3 cameras (left / right Zed stereo +
   wrist), DROID codebase installed on the control laptop.

3. **PyTorch checkpoint**: pi0.5 DROID checkpoint. `--collect`-mode activation
   collection requires the checkpoint to be loadable in PyTorch on the remote
   server.

## Workflow

### 1. Collect activations (if rebuilding the NPZ)

Real-robot rollouts with the operator labeling success after each:

```bash
# Terminal 1 — remote server with activation collection
uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output_dir activations/pi05_droid \
    policy:checkpoint --policy.config pi05_droid --policy.dir checkpoints/pi05_droid

# Terminal 2 — DROID laptop
python examples/droid/main.py --collect \
    --external_camera left \
    --left_camera_id <ID> --right_camera_id <ID> --wrist_camera_id <ID> \
    --remote_host <server-ip>
# ... run many rollouts per instruction, label success at prompt
```

### 2. Build conceptors

```bash
uv run python experiments/droid/compute_conceptors.py \
    --activation_root activations/pi05_droid \
    --output_path conceptors/droid_conceptors_fresh.npz
```

### 3. Narrow the parameter grid

```bash
uv run python experiments/droid/select_parameters.py \
    --conceptor_npz conceptors/droid_conceptors.npz \
    --output_json experiments/droid/selected_params.json
```

### 4. Manual eval

Launch a steering server for each `(strategy, layer, α, β)` you want to test:

```bash
# Server (one condition per launch — restart to change params)
uv run scripts/serve_policy.py --env DROID --pytorch --steer \
    --conceptor_npz conceptors/droid_conceptors.npz \
    policy:checkpoint --policy.config pi05_droid --policy.dir checkpoints/pi05_droid

# DROID laptop — each rollout uses the server's currently-loaded conditions;
# pass the matching flags so obs[__steering__] is set correctly:
python examples/droid/main.py \
    --external_camera left \
    --left_camera_id <ID> --right_camera_id <ID> --wrist_camera_id <ID> \
    --remote_host <server-ip> \
    --steer --steering_layer 11 --steering_alpha 0.1 --steering_beta 0.3 \
    --steering_strategy global
```

Log success rates by hand, compare conditions. There is intentionally no
automated accounting — the data is too precious for a blind argmax.

## Why no `find_best_configs.py`?

Because:
- Real-robot rollouts cost ~5 minutes each of operator time.
- Success labels are human judgment calls, often partial credit.
- Environment drift (object placement, lighting, wear) matters more than
  hyperparameter choice.

`select_parameters.py` cuts the grid 10× so the operator can ablate a
realistic number of conditions in one session.
