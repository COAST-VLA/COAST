# pi0.5 RoboCasa — conceptor steering

Per-task conceptor steering for the pi0.5 RoboCasa policy. One SLURM job per
task; the policy loads once and we sweep `(layer × alpha × beta × strategy)`
against pre-computed conceptors stored in
`$OPENPI_DATA_HOME/robocasa_conceptors.npz`.

## Strategies

The default is `--strategies global per_step`. Three strategies are wired up:

### `global` — static conceptor

`h' = (1 - β) h + β (h @ Cᵀ)` with `C = C_contrastive` built at a chosen
alpha. The same `C` is applied at every denoising step. Sweeps over
`(layer, alpha, beta)`.

### `per_step` — true time-varying (recommended)

**Different conceptor at every denoising step.** This is what `per_step`
*should* do — and what it now does after the fix in commit `a84a6f0`. The
hook holds the full list of per-step conceptors (one per denoising step
0..9 for pi0.5, one per built denoising step for other architectures); the
policy's `infer_with_steering` calls `hook.set_denoise_step(t)` each step
(see `src/openpi/models_pytorch/pi0_pytorch.py:944`) and the hook indexes
into its `_M_per_step` cache to apply the matching matrix.

The per-step keys in the npz are alpha-free (`C_contrastive` only, built
at the build script's fixed alpha), so this strategy is iterated over
`(layer, beta)` only. Condition names: `per_step_L{layer}_b{beta}`.

**Why "per denoising step" matters.** Activations at ds=0 (near-noise)
and ds=9 (near-clean) have very different geometry. A single conceptor
fit to the union of all steps is a compromise that fits neither end well.
Building one conceptor per step and switching on the fly tracks the
denoising trajectory's drifting subspace.

### `per_step_N` — legacy static-at-step-N (ablation)

Same as `global` but `C` is built using activations from a single
denoising step `N`. Static — applied uniformly across all denoising
steps. Kept as an ablation: it answers *"does the conceptor's quality
matter as much as the per-step swapping does?"* If `per_step` beats
`per_step_N` for any N, the time-varying behaviour is doing real work.

### Positive-only — `positive_only_steering.py`

`C = C_success` (no `NOT C_failure` term) instead of contrastive. Uses
the same hook math; just a different matrix. Run it via the dedicated
script. Answers: *"is the contrastive AND-NOT actually helping, or is
push-toward-success enough?"*

## Required `.npz` keys

The build script must write at least:

```
{task}__L{layer}__{alpha}__C_contrastive       # global, per (layer, alpha)
{task}__L{layer}__{alpha}__C_success           # for positive_only
{task}__L{layer}__per_step_{ds}__C_contrastive # per_step + per_step_N (per ds)
```

`per_step` walks `ds = 0, 1, 2, ...` until a key is missing, so the
build script should produce a contiguous range starting at 0.

## Output and resume behaviour

Results land at `steering_results/{task}/summary.json`. Each row is
`{"condition": "<name>", "success_rate": <float|nan>}`. `_save_summary`
merges-from-disk before writing, so two parallel jobs (e.g. `per_step` +
`positive_only`) can write to the same file safely. To re-run a single
condition, delete its row and re-launch — the script's skip logic only
re-executes missing rows.

Condition naming is backward-compatible:

- legacy static-at-step-N rows: `per_step_0_L5_a1.0_b0.3`
- new time-varying rows:        `per_step_L5_b0.3`

so old `summary.json` files keep working alongside re-runs with the new
code.

## Quick run

```bash
# default sweep: global + true per_step
uv run experiments/pi05_robocasa/src/conceptor_steering.py --task CloseFridge

# only true time-varying per_step
uv run experiments/pi05_robocasa/src/conceptor_steering.py \
    --task CloseFridge --strategies per_step

# legacy static-at-step-N for comparison
uv run experiments/pi05_robocasa/src/conceptor_steering.py \
    --task CloseFridge --strategies per_step_0 per_step_9

# positive-only (different script)
uv run experiments/pi05_robocasa/src/positive_only_steering.py --task CloseFridge
```
