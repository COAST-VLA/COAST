# SAE-ActAdd Steering Baseline

A drop-in baseline for the conceptor / linear-ActAdd steering pipeline. Trains a
per-task TopK Sparse Autoencoder on the captured residual-stream activations,
selects contrastive features using two filters, and materializes a single
steering vector `v_sae ∈ R^d` that is injected via the same additive path
already used by `linear_final` in `conceptor_steering.py`. No new model code,
no new server plumbing — just a different source of `v`.

## What this baseline is

Given activations `h ∈ R^d` collected from positive (success) and negative
(failure) episodes of one task, train a TopK SAE such that

    f       = TopK(ReLU(W_enc (h - b_dec) + b_enc))     # f ∈ R^{d_sae}, non-negative, k-sparse
    h_hat   = W_dec @ f + b_dec

Then build a steering vector by averaging surviving feature differences:

    μ⁺[k]   = mean of f_k over success samples
    μ⁻[k]   = mean of f_k over failure samples
    fire_rate[k] = fraction of (pos ∪ neg) samples with f_k > 0
    keep[k] = (fire_rate >= 0.005)  AND  (min(μ⁺,μ⁻) / max(μ⁺,μ⁻) <= 0.5)
    v_latent[k]  = (μ⁺[k] - μ⁻[k]) if keep[k] else 0
    v_sae   = W_decᵀ @ v_latent
    v_sae  /= ||v_sae||

Inject at eval time by

    h ← h + α · v_sae

`α` is the only hyperparameter we sweep (matched to the existing `linear_final`
grid: `{0.5, 1.0}`). No top-K feature count, no per-task SAE size tuning.

This is the additive version of SAE steering — the simplest defensible
baseline. We deliberately do not implement feature *clamping* (set f_k to a
target value, decode, re-mix); clamping mixes in SAE reconstruction error and
adds free hyperparameters. If reviewers ask for it, it's a small additional
PR on top of this code.

### Why two filters

* **Filter A — rare features (`fire_rate < 0.005`).** TopK SAEs always have a
  long dead-feature tail. With `TopK=64` and `d_sae=4096–8192`, the *expected*
  per-feature fire rate is `64 / d_sae ≈ 0.8–1.6%`, so 0.5% drops only the
  bottom tail where μ̂⁺ and μ̂⁻ are statistically unreliable.
* **Filter B — bilaterally active features (`min(μ⁺,μ⁻)/max(μ⁺,μ⁻) > 0.5`).**
  Features that fire similarly in both classes carry no contrastive signal.
  This is the operationalization of Khan et al.'s "drop features that activate
  in both classes" criterion.

After both filters, we use *all surviving features* — no top-K-by-Cohen's-d
selection, because Khan-style steering injects the full averaged Δz over
surviving features. The "feature 4909" stories in monosemanticity papers are
post-hoc *interpretation*, not what's actually injected.

### Why unit-normalize v_sae

Both `fit_linear_vectors.py` and `build_conceptors.py` (`for_subin/`) return
unit-norm contrastive directions. The existing `linear_final` α grid is
calibrated against unit vectors. If the SAE side skipped normalization,
`||v_sae||` would vary across tasks (more surviving features → larger raw
norm), and the same nominal α would steer different tasks at different
effective strengths. Normalize for an apples-to-apples α sweep.

## Repo layout

```
experiments/sae/
  README.md                     ← you are here
  src/
    sae_module.py               ← TopK SAE class (~50 lines)
    train_sae.py                ← per-task SAE trainer (PyTorch, single GPU)
    fit_sae_vectors.py          ← Khan-style filter + v_sae builder

experiments/{pi0_fast_libero, pi0_fast_metaworld, pi05_libero, pi05_robocasa}/
  src/
    sae_steering.py             ← thin per-experiment driver, reuses
                                   conceptor_steering.py infrastructure
    run_sae_steering.sh         ← SLURM driver, one job per task

  sae_steering_results/         ← per-task summary.json output (created on first run)
    {task_name}/
      summary.json
      sweep_args.json
      logs/, scripts/
```

Per-task SAE checkpoints and steering-vector NPZs live under
`$OPENPI_DATA_HOME` (the same place the conceptor and linear-vector NPZs live):

```
$OPENPI_DATA_HOME/
  sae_checkpoints/
    pi0fast_libero/{task}.pt
    pi0fast_metaworld/{task}.pt
    pi05_libero/{task}__L{0,5,11,17}.pt
    pi05_robocasa/{task}__L{0,5,11,17}.pt
  pi0fast_libero_sae_vectors.npz
  pi0fast_metaworld_sae_vectors.npz
  libero_sae_vectors.npz
  robocasa_pi05_sae_vectors.npz
```

NPZ key naming (for both pi05 and pi0-fast schemas, single global vector per task):

```
pi05_libero / pi05_robocasa:  {task}__L{layer}__sae__V_contrastive   (1024,)
pi0_fast_libero / metaworld:  {task}__sae__V_contrastive             (2048,)
```

Per-denoising-step variants (e.g. `per_step_0`, `per_step_9`) are deliberately
*not* computed for the SAE baseline. Unlike conceptors — where each per-step
variant is mathematically a different matrix capturing a different slice of
covariance — SAE per-step variants would just reuse the same `W_dec` with
contrastive means estimated from 1/10 the samples. That adds noise without
probing a different mechanism. Mirrors the `linear_final` baseline (one global
ActAdd vector per task).

## Hyperparameters (fixed across all four experiments)

| | value | rationale |
|---|---|---|
| TopK k                 | **64** | Standard for residual-stream SAEs at this scale |
| d_sae                  | **4 × hidden** | 4096 (pi05) / 8192 (pi0-fast). Conservative — per-task data is small (5k–45k samples per (task, layer)) |
| Encoder                | ReLU → TopK | Guarantees non-negative `f`, required by Filter B |
| Decoder norm           | unit columns after each step | Standard |
| Optimizer              | AdamW, lr=3e-4, β=(0.9, 0.999) | |
| Steps                  | 30k | Plenty for 5k–45k samples |
| Batch                  | 4096 | |
| Filter A `rare_thresh`        | 0.005 | Drops bottom of fire-rate distribution |
| Filter B `bilateral_thresh`   | 0.5   | Drops features that fire similarly in both classes |
| Var-explained floor (gate)    | 0.80  | Skip the task if SAE recon is too poor to trust |
| Sweep α                | **{0.25, 0.5, 1.0, 2.0}** | Wider than `linear_final` to capture per-task α sensitivity |

We deliberately do not sweep over k, d_sae, expansion factor, etc. — this is a
baseline, not a method paper. If the result is competitive, we can revisit.

## Activation schemas

| Experiment | Schema | Capture file | Tensor key | Shape |
|---|---|---|---|---|
| pi05_libero       | PyTorch flow-matching | `step_*/suffix_residual.npz` | `all_suffix_residual` | (10, 4, 10, 1024) |
| pi05_robocasa     | PyTorch flow-matching | `step_*/suffix_residual.npz` | `all_suffix_residual` | (10, 4, 50, 1024) |
| pi0_fast_libero   | JAX autoregressive    | `step_*/hidden_states.npz`  | `token_pre_logits`    | (n_tokens, 2048) |
| pi0_fast_metaworld| JAX autoregressive    | `step_*/hidden_states.npz`  | `token_pre_logits`    | (n_tokens, 2048) |

For pi05 (PyTorch), `LAYER_MAP = {0: 0, 5: 1, 11: 2, 17: 3}` selects the
correct slice along the layer axis. For pi0-fast there is no layer axis —
single intervention point.

## End-to-end recipe (per experiment)

Three steps. Each step has both a direct CLI invocation and a SLURM submitter
script under `experiments/{exp}/src/`:

| Step | Script | What it does |
|---|---|---|
| 1 | `run_sae_train.sh` | Train per-task TopK SAEs (1 GPU, ~1–6h depending on data size). |
| 2 | `run_sae_fit.sh` | Encode activations through trained SAEs, apply Khan filters, write per-task v_sae to NPZ. |
| 3 | `run_sae_steering.sh` | Per-task SLURM jobs: load policy, sweep `α ∈ {0.25, 0.5, 1.0, 2.0}` against v_sae, write `summary.json` per task. |

Everything below shows the direct CLI form. The SLURM scripts wrap the same
commands and bake in the appropriate resource asks. All commands run from the
openpi-new repo root with the **root venv** active (`uv run ...`).

### Step 1 — train per-task SAEs

```bash
# pi0_fast LIBERO  (~10 tasks, ~5 min/task on a single B200)
uv run python experiments/sae/src/train_sae.py \
  --schema pi0fast \
  --activations-dir $OPENPI_DATA_HOME/pi0fast-libero-activations-v1-2000-15env/2000 \
  --output-dir $OPENPI_DATA_HOME/sae_checkpoints/pi0fast_libero \
  --d-sae-mult 4 --k 64

# pi0_fast MetaWorld  (~45 tasks)
uv run python experiments/sae/src/train_sae.py \
  --schema pi0fast \
  --activations-dir $OPENPI_DATA_HOME/pi0fast-metaworld-activations-v1-ml45train-16env/2500 \
  --output-dir $OPENPI_DATA_HOME/sae_checkpoints/pi0fast_metaworld \
  --d-sae-mult 4 --k 64

# pi0.5 LIBERO  (10 tasks × 1 layer at L=11)
uv run python experiments/sae/src/train_sae.py \
  --schema pi05 \
  --activations-dir $OPENPI_DATA_HOME/activations/pi05_libero_2000_15env/openpi-libero-2000 \
  --output-dir $OPENPI_DATA_HOME/sae_checkpoints/pi05_libero \
  --layers 11 --d-sae-mult 4 --k 64

# pi0.5 RoboCasa  (7 tasks × 1 layer at L=11)
uv run python experiments/sae/src/train_sae.py \
  --schema pi05 \
  --activations-dir $OPENPI_DATA_HOME/huggingface/lerobot/ksb21st/robocasa-activations-75000 \
  --output-dir $OPENPI_DATA_HOME/sae_checkpoints/pi05_robocasa \
  --layers 11 --d-sae-mult 4 --k 64
```

The trainer writes:
* `{output_dir}/{task}.pt` (pi0fast) or `{output_dir}/{task}__L{L}.pt` (pi05)
* `{output_dir}/training_summary.json` — per-task `holdout_var_explained`

### Step 2 — fit steering vectors with Khan-style filters

```bash
uv run python experiments/sae/src/fit_sae_vectors.py \
  --schema pi0fast \
  --activations-dir $OPENPI_DATA_HOME/pi0fast-libero-activations-v1-2000-15env/2000 \
  --sae-dir $OPENPI_DATA_HOME/sae_checkpoints/pi0fast_libero \
  --output-npz $OPENPI_DATA_HOME/pi0fast_libero_sae_vectors.npz

uv run python experiments/sae/src/fit_sae_vectors.py \
  --schema pi05 \
  --activations-dir $OPENPI_DATA_HOME/activations/pi05_libero_2000_15env/openpi-libero-2000 \
  --sae-dir $OPENPI_DATA_HOME/sae_checkpoints/pi05_libero \
  --layers 11 \
  --output-npz $OPENPI_DATA_HOME/libero_sae_vectors.npz
```

(Analogous commands for the other two experiments; full set is at the end of
this README.)

The fitter writes:
* `{output_npz}` — keyed `{task}__L{L}__sae__V_contrastive` (pi05) or
  `{task}__sae__V_contrastive` (pi0fast). One vector per task (pi0-fast) or
  per (task, layer) (pi05).
* `{output_npz}.diagnostics.json` — per-task `n_after_A`, `n_after_B`,
  `raw_v_norm` (pre-normalization), top-5 contributing feature indices.

### Step 3 — run steered eval

Per-experiment thin drivers under `experiments/{exp}/src/sae_steering.py`. Each
loads `v_sae`, starts the same WebSocket policy server used by
`conceptor_steering.py`, and sweeps α. Results land in
`experiments/{exp}/sae_steering_results/{task}/summary.json`.

The matching SLURM submitter is `experiments/{exp}/src/run_sae_steering.sh` —
one job per task, sweeps `α ∈ {0.25, 0.5, 1.0, 2.0}` plus a no-steering
baseline (5 conditions per task). Already-completed conditions in an existing
`summary.json` are skipped on re-submit.

```bash
DRY_RUN=true bash experiments/pi0_fast_libero/src/run_sae_steering.sh    # preview
              bash experiments/pi0_fast_libero/src/run_sae_steering.sh    # submit
```

For pi05 experiments you'll likely need to override the policy checkpoint path
(see Troubleshooting below):

```bash
CHECKPOINT_DIR=/full/path/to/pi05_libero/.../2000 \
  bash experiments/pi05_libero/src/run_sae_steering.sh
```

## Diagnostics to check before trusting the result

Open `{output_npz}.diagnostics.json` after Step 2. Per task:

* `holdout_var_explained` (in `training_summary.json`) — gate at **0.80**.
  If lower, the SAE didn't learn a useful basis on this task; the steering
  result is uninterpretable. The `--var-explained-floor` flag in
  `fit_sae_vectors.py` skips these tasks.
* `n_after_A`, `n_after_B` — number of features surviving each filter. If
  `n_after_B < 30` for any task, the SAE didn't separate the classes well —
  the v_sae for that task is a few-feature average and probably noisy.
* `raw_v_norm` — pre-normalize ‖v_sae‖. If this varies wildly across tasks
  (say, 10× spread), unit-normalization is doing the right thing.
* `top_features` — sanity-check by interpreting the top decoder column with
  whatever interpretation tools you have.

## Expected cost

| Stage | Cost |
|---|---|
| SAE training | ~70 SAEs × ~5 min/SAE on one GPU = **6 GPU-hours total**, parallelizable across SLURM |
| Vector fitting | minutes — encoding + simple stats |
| Eval sweep | ~144 conditions × 10–20 min each ≈ same wall-clock as one `linear_final` sweep |

## Full Step 2 / Step 3 commands

### Step 2 (vector fitting)

```bash
# pi0_fast LIBERO
uv run python experiments/sae/src/fit_sae_vectors.py \
  --schema pi0fast \
  --activations-dir $OPENPI_DATA_HOME/pi0fast-libero-activations-v1-2000-15env/2000 \
  --sae-dir $OPENPI_DATA_HOME/sae_checkpoints/pi0fast_libero \
  --output-npz $OPENPI_DATA_HOME/pi0fast_libero_sae_vectors.npz

# pi0_fast MetaWorld
uv run python experiments/sae/src/fit_sae_vectors.py \
  --schema pi0fast \
  --activations-dir $OPENPI_DATA_HOME/pi0fast-metaworld-activations-v1-ml45train-16env/2500 \
  --sae-dir $OPENPI_DATA_HOME/sae_checkpoints/pi0fast_metaworld \
  --output-npz $OPENPI_DATA_HOME/pi0fast_metaworld_sae_vectors.npz

# pi0.5 LIBERO
uv run python experiments/sae/src/fit_sae_vectors.py \
  --schema pi05 \
  --activations-dir $OPENPI_DATA_HOME/activations/pi05_libero_2000_15env/openpi-libero-2000 \
  --sae-dir $OPENPI_DATA_HOME/sae_checkpoints/pi05_libero \
  --layers 11 \
  --output-npz $OPENPI_DATA_HOME/libero_sae_vectors.npz

# pi0.5 RoboCasa
uv run python experiments/sae/src/fit_sae_vectors.py \
  --schema pi05 \
  --activations-dir $OPENPI_DATA_HOME/huggingface/lerobot/ksb21st/robocasa-activations-75000 \
  --sae-dir $OPENPI_DATA_HOME/sae_checkpoints/pi05_robocasa \
  --layers 11 \
  --output-npz $OPENPI_DATA_HOME/robocasa_pi05_sae_vectors.npz
```

### Step 3 (eval)

```bash
bash experiments/pi0_fast_libero/src/run_sae_steering.sh
bash experiments/pi0_fast_metaworld/src/run_sae_steering.sh

# pi0.5 experiments need CHECKPOINT_DIR pointing to the real checkpoint —
# the committed default is a relative path (anonymity), the real checkpoints
# may live elsewhere on your machine.
CHECKPOINT_DIR=/path/to/pi05_libero/libero_b200_bs512/2000 \
  bash experiments/pi05_libero/src/run_sae_steering.sh
CHECKPOINT_DIR=/path/to/pi05_pretrain_human300/multitask_learning/75000 \
  bash experiments/pi05_robocasa/src/run_sae_steering.sh
```

(Same applies to `run_sae_train.sh` and `run_sae_fit.sh` for pi05 — the SAE
training itself only needs `OPENPI_DATA_HOME` because it just reads activations,
but the steering eval has to load the policy checkpoint.)

## Troubleshooting

* **`FileNotFoundError: Metadata file (named _METADATA) does not exist at .../checkpoints/pi05_*/...`**:
  the pi05 driver's default `--checkpoint-dir` is a relative path that may
  not exist in your environment (we keep it generic for anonymity). Either
  (a) symlink your real pi05 checkpoint into `openpi-new/checkpoints/<config>/...`,
  or (b) export `CHECKPOINT_DIR=/full/path` before running `run_sae_steering.sh` —
  the bash submitter bakes `${CHECKPOINT_DIR}` into the generated SLURM scripts.
  The pi0-fast scripts don't have this problem because their checkpoints live
  under `openpi-new/checkpoints/` already.
* **`key not in NPZ`** in `sae_steering.py`: the fitter skipped this task —
  check `training_summary.json` for var-explained or `diagnostics.json` for
  n_pos / n_neg.
* **All features fail Filter B** for a task: usually means the SAE found no
  features that activate predominantly in one class. The task likely has too
  few or too imbalanced episodes — check `episode_success` counts.
* **JIT recompile every condition** in pi0-fast: should not happen — the SAE
  driver pads `C_stack` to identity (β=0) and reuses the same JIT signature
  as the conceptor sweep. If you see recompilation, check that
  `max_decoding_steps` matches across runs.
* **PyTorch hook fires but SR doesn't change**: the pi05 `LinearSteeringHook`
  defined in `pi05_libero/src/sae_steering.py` (or imported from
  `pi05_robocasa/src/conceptor_steering.py`) registers on
  `paligemma_with_expert.gemma_expert.model.layers[layer_idx]`. If the layer
  index is out of range (>17), `register_forward_hook` succeeds silently but
  the layer never runs.

## Provenance / why the design is the way it is

Most SAE-steering literature pre-2025 (Templeton, Bricken, Marks) selects a
small top-K of features by Cohen's d or absolute Δμ, then injects a single
decoder column. We deliberately do not do that here: Khan et al. is the
faithful reference for "use the full averaged Δz with the rare and bilaterally-
active features filtered out", and that's what this baseline reports. K is
not a hyperparameter; the only knob is α.
