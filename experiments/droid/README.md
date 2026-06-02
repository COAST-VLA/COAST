# DROID Steering — Full End-to-End Pipeline (Real Robot)

DROID is a real-robot harness — there is **no simulator**, success is
labeled by the operator after each rollout, and there is no "task suite" in
the LIBERO / RoboCasa / MetaWorld sense. That rules out the automated
subprocess-grid sweep pattern used for the sim envs. Instead, tuning uses
`select_parameters.py`: a diagnostic-based narrower that picks a short
shortlist of `(layer, α, β)` candidates from the conceptor NPZ alone,
which the operator then evaluates manually per free-form instruction.

The workflow is **1 server + 1 build + 1 diagnostic + N manual rollouts**,
with collection, diagnostic narrowing, and steered eval all separated in
time (collection typically happens on day 0; steered eval on day 1+).

## Semi-automated driver (wraps all commands below)

```bash
bash experiments/droid/run_end_to_end.sh
```

Automates the GPU-host steps (server start/stop, NPZ build, diagnostic
narrower) and pauses with `ENTER`-to-continue prompts at each manual
step (operator rollouts on the DROID laptop). Logs land in
`experiments/droid/run_logs/`. Read on for the stage-by-stage commands
it wraps.

## Commands

```bash
# (a) Start collection server on the GPU host, port 8000
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output_dir activations/droid --port 8000 \
    policy:checkpoint --policy.config pi05_droid \
    --policy.dir gs://openpi-assets/checkpoints/pi05_droid

# (b) On the DROID control laptop, run collection rollouts per instruction.
#     Operator enters the free-form instruction, executes the episode, and
#     labels success at the prompt. Each instruction should be repeated ~15-30
#     times for enough success/failure volume to build a stable conceptor.
python3 scripts/main.py --remote_host=<server_ip> --remote_port=8000 \
    --external_camera="left" --collect

# (c) Kill the collection server (on the GPU host)
pkill -f "scripts/serve_policy.py.*port 8000"

# (d) Build the conceptor NPZ from the collected activations (CPU-only)
CUDA_VISIBLE_DEVICES="" uv run python experiments/droid/compute_conceptors.py \
    --activation_root activations/droid \
    --output_path conceptors/droid_conceptors.npz

# (e) Run the diagnostic narrower — no robot time, no eval rollouts. Picks
#     the best layer via per-layer quota and narrows α to values whose
#     success/failure overlap falls inside a target band.
CUDA_VISIBLE_DEVICES="" uv run python experiments/droid/select_parameters.py \
    --conceptor-npz conceptors/droid_conceptors.npz \
    --output-json experiments/droid/selected_params.json

# (f) Start the steering server, port 8001
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --pytorch --steer \
    --conceptor_npz conceptors/droid_conceptors.npz --port 8001 \
    policy:checkpoint --policy.config pi05_droid \
    --policy.dir gs://openpi-assets/checkpoints/pi05_droid

# (g) Operator runs manual steered eval for each shortlisted config.
#     Example — one (layer=11, α=0.1, β=0.3) condition, one instruction.
#     Repeat for every (layer, α, β) in selected_params.json, ~15-30 rollouts each.
python3 scripts/main.py --remote_host=<server_ip> --remote_port=8001 \
    --external_camera="left" \
    --steer --steering_layer 11 --steering_alpha 0.1 --steering_beta 0.3 \
    --steering_strategy global
#     The operator labels success per rollout; main.py saves results/eval_<ts>.csv
#     which is the authoritative per-condition record.

# (h) Baseline (unsteered) rollouts for comparison — same instruction, same
#     rollout count, drop the steering flags.
python3 scripts/main.py --remote_host=<server_ip> --remote_port=8001 \
    --external_camera="left"
```

## What each step produces

| Step | Output | Notes |
|------|--------|-------|
| (b) | `activations/droid/<checkpoint>/<instruction-slug>/episode_NNN_env_000/step_NNNN/*.npz` | One dir per instruction; 15-30 rollouts recommended per class |
| (d) | `conceptors/droid_conceptors.npz` | `{slug}__L{L}__{α}__C_{kind}` + per-step + `linear_direction` |
| (e) | `experiments/droid/selected_params.json` | `{best_layer, selected_alphas, selected_betas, overlap_band, diagnostics}` |
| (g) | `results/eval_<ts>.csv` (on the DROID laptop), one per condition | `success` (0/1), `duration`, `video_filename` per rollout |
| (h) | `results/eval_<ts>.csv` (baseline) | Same schema |

## Customizing `select_parameters.py`

| Flag | Default | Notes |
|------|---------|-------|
| `--conceptor-npz` | — | Path to the NPZ built in (d) |
| `--output-json`   | `experiments/droid/selected_params.json` | Narrower's output |
| `--overlap-band`  | `(0.85, 0.95)` | Keep α whose mean s/f overlap falls in this band |
| `--beta-shortlist` | `(0.1, 0.3)` | Candidate β values (orthogonal to the α narrower) |

Lower overlap → less room for steering to help; higher overlap → too much
noise. 0.85-0.95 is the band the diagnostic ships with and matches what
was used in published DROID work.

## Skipping activation collection

No pre-built DROID conceptor NPZ is published. Collection is mandatory —
there's no equivalent of the sim HF downloads for real-robot rollouts.

## Notes

- **Why no `find_best_configs.py`.** Real-robot eval doesn't have enough
  throughput to support the LIBERO-style grid (10 tasks × ~20 conditions
  × 10 eps = 2000 rollouts ≈ 40+ hours of operator time).
  `select_parameters.py` uses only the conceptor NPZ — no rollouts, no
  GPU — to cut the search space down to ~3-5 (layer, α, β) candidates
  that the operator can actually test.
- **Held-out separation is physical, not seed-based.** DROID is
  interactive — there's no canonical initial-state list to offset into
  like LIBERO. Each rollout starts from whatever physical scene the
  operator has set up. Held-out separation between collection and eval
  means different real-world setups / scenes / objects between
  collection day and eval day. Budget for this explicitly when planning
  the operator schedule.
- **Old NPZs may be missing per-step keys 1-8.** The current
  `DEFAULT_PER_STEP_INDICES` is all 10 denoising steps, but NPZs built
  before that change have only `per_step_0` / `per_step_9` → `per_step`
  strategy will NaN. Rebuild via step (d) if you hit this.

## See also

- `examples/droid/README.md` — end-user `--steer` flag documentation + the
  interactive main.py loop.
- `experiments/shared/select_parameters.py` — the underlying diagnostic
  math (shared between DROID and any future real-robot envs).
- `src/openpi/serving/steering.py` — the runtime (hooks + wrapper).
- `src/openpi/serving/conceptors.py` — the NPZ builder.
