# diffusion_policy / RoboCasa Steering Pipeline

End-to-end pipeline for **conceptor-based steering** on
`DiffusionTransformerHybridImagePolicy`. Takes activation captures from the
eval/activation sweep, builds per-task contrastive conceptors at multiple
alphas, optionally selects per-(task, checkpoint) hyperparameters, and runs a
SLURM-fanned-out steering evaluation sweep with success-rate aggregation.

## Pipeline at a glance

```
collect_activations_robocasa.py   ──▶ /<ACTIVATIONS_ROOT>/<ckpt>/<task>/episode_*/step_*/suffix_residual.npz
                                                              │
                                          build_conceptors.py │  (per-ckpt, multi-alpha global + per_step)
                                          add_per_step_alphas.py  (post-hoc add per_step alphas via eigendecomp)
                                                              ▼
                                  experiments/robocasa_steering/conceptors/<ckpt>/conceptors.npz
                                                              │
                                  select_parameters_per_task.py │  (per-(task, ckpt) hyperparam selection)
                                  build_and_select_all.py         (driver: walks all ckpts + writes manifest)
                                                              ▼
                                  experiments/robocasa_steering/conceptors/manifest.json
                                  experiments/robocasa_steering/conceptors/common_params__per_task.json
                                                              │
                                  steering_sweep/build_recipes.py │  (per-(ckpt, task) recipe table)
                                                              ▼
                                  steering_sweep/recipes.json
                                  steering_sweep/ablation_recipes.json   (per-(task, epoch) lists for ablations)
                                                              │
                                  submit_steering_sweep.sh    │  (SLURM fan-out)
                                  steering_sweep.sbatch         (per-ckpt SLURM job: loops tasks, calls steering.py)
                                                              ▼
                                  /<OUTPUT_ROOT>/<ckpt>/<task>/summary.json   (steering condition success rates)
```

## Stage 1 — Build conceptors

`experiments/robocasa_steering/build_conceptors.py` walks one
`<ACTIVATIONS_ROOT>/<ckpt>` activation tree (output of
`collect_activations_robocasa.py`) and produces a single `.npz` of conceptor
matrices for every mixed-outcome task at the requested layers and alphas.

**Key file:** `experiments/robocasa_steering/build_conceptors.py`

**Default alpha grids** (match `pi05_robocasa` / `pi05_libero` global α grid):
- `--alphas 0.1 0.5 1.0 2.0 10.0`        (global C grid)
- `--per-step-alphas 0.1 0.5 1.0 2.0 10.0`  (per_step C grid; multi-alpha — *not* in pi05's build, custom to this repo)

**Output keys** (extends pi05_libero's schema with alpha-aware per_step keys):
```
{task}__L{layer}__{alpha}__C_{success|failure|contrastive}            # main per-alpha grid
{task}__L{layer}__{alpha}__per_step_{ds}__C_{success|failure|contrastive}   # alpha-aware per_step (NEW)
{task}__L{layer}__per_step_{ds}__C_{success|failure|contrastive}      # legacy alpha-less (aliased to first per_step alpha)
{task}__L{layer}__linear__V_{success|failure|contrastive}             # ActAdd direction
{task}__L{layer}__linear_per_step_{ds}__V_{success|failure|contrastive}
```

Tasks where every episode succeeded **or** every episode failed are
auto-dropped (need ≥1 success AND ≥1 failure, controlled by
`MIN_PER_CLASS = 1`).

**Single-checkpoint usage:**
```bash
.venv/bin/python experiments/robocasa_steering/build_conceptors.py \
    --activations-dir /mnt/bird_home/kim34/eval_sweep_results/<ckpt_stem> \
    --output-npz       experiments/robocasa_steering/conceptors/<ckpt_stem>/conceptors.npz
```

**Backward compat:** legacy `--per-step-alpha <float>` (singular) still works
and overrides the plural list to a single-alpha build, so older callers don't
break.

## Stage 1b — `add_per_step_alphas.py` (post-hoc per_step alpha extension)

If you have an **existing** `conceptors.npz` built at a single per_step alpha
(pre-multi-alpha builds, or from pi05_libero's `build_conceptors.py`), you can
extend it with additional per_step alphas **without re-reading any activation
data**. Eigendecomposition recovers the underlying data correlation eigenvalues
from `C_α_orig`, then reconstructs `C` at the new alphas.

**Key file:** `experiments/robocasa_steering/add_per_step_alphas.py`

**Math:**
```
C_orig = R · (R + α_orig⁻²·I)⁻¹       where R = X_centered^T X_centered / N
       = U · diag(e_orig)  · U^T       (eigendecomposition, R is symmetric PSD)
       e_orig_i = λ_i / (λ_i + α_orig⁻²)
       λ_i      = e_orig_i · α_orig² / (1 - e_orig_i)        (recovery)
       e_new_i  = λ_i / (λ_i + α_new⁻²)
       C_new    = U · diag(e_new) · U^T
       C_c_new  = C_s_new · (I − C_f_new)                    (contrastive recompute)
```

**Numerical equivalence to a fresh native build:** verified to fp32 round-trip
precision (max element-wise Δ ≈ 6e-8, Frobenius rel. err ≈ 1e-7) — see
discussion in conversation history; matches the noise floor of the fp32
storage format.

**Usage:**
```bash
.venv/bin/python experiments/robocasa_steering/add_per_step_alphas.py \
    --conceptor-npz experiments/robocasa_steering/conceptors/<ckpt>/conceptors.npz \
    --orig-alpha    1.0 \
    --new-alphas    0.1 0.5 2.0
```

Adds the new alpha-aware keys in place (no rewrite of original keys); keeps
the legacy alpha-less `__per_step_{ds}__` keys alongside.

**When to use:**
- You already ran `build_conceptors.py` at one per_step alpha and want more.
- Avoids re-reading the activation tree (which dominates wall-clock — NFS I/O
  bottleneck, ~2.5 h per ckpt vs <1 minute for the eigendecomp recovery).

## Stage 2 — Select parameters

Two selectors:

- **`experiments/robocasa_steering/select_parameters.py`** — pi05_libero's
  classic single-recipe-per-conceptor-file selector. Picks the best layer by
  mean quota across tasks, then alphas in the [0.85, 0.95] overlap band.
  Output: ONE `(best_layer, selected_alphas, selected_betas)` per file
  (averaged across tasks).
- **`experiments/robocasa_steering/select_parameters_per_task.py`** *(new)* —
  wraps `select_parameters.select_parameters` with a per-task npz subset view
  (`_NpzSubset` filtering by `{task}__` prefix), runs the selection
  independently for each task. Output: a JSON keyed by task with one recipe
  per task.

**Usage (per-task):**
```bash
.venv/bin/python experiments/robocasa_steering/select_parameters_per_task.py \
    --conceptor-npz experiments/robocasa_steering/conceptors/<ckpt>/conceptors.npz \
    --output-json   experiments/robocasa_steering/conceptors/<ckpt>/selected_params__per_task.json \
    --betas 0.1 0.3
```

## Stage 3 — Driver across all checkpoints + manifest

`experiments/robocasa_steering/build_and_select_all.py` *(new)* walks every
`<ACTIVATIONS_ROOT>/<ckpt>` subdir, runs `build_conceptors.py` then
`select_parameters_per_task.py` for each, and writes a single
`manifest.json` keyed by `(ckpt_stem, task)` so downstream steering can
dispatch from one file.

**Key file:** `experiments/robocasa_steering/build_and_select_all.py`

**Manifest schema:**
```json
{
  "activations_root": "...",
  "checkpoints": {
    "<ckpt_stem>": {
      "conceptor_npz": "...",                  // absolute path
      "selected_params_json": "...",
      "tasks": {
        "<task>": {
          "best_layer": int,
          "selected_alphas": [...],
          "selected_betas": [...],
          "overlap_band": [low, high]
        }
      },
      "tasks_skipped_no_success": [...]
    }
  }
}
```

**Usage:**
```bash
.venv/bin/python experiments/robocasa_steering/build_and_select_all.py \
    --activations-root /mnt/bird_home/kim34/eval_sweep_results \
    --output-dir       experiments/robocasa_steering/conceptors
```

For 5-way parallel builds (one per ckpt, useful when NFS I/O dominates):
```bash
for d in /mnt/bird_home/kim34/eval_sweep_results/*/; do
    STEM=$(basename "$d")
    mkdir -p experiments/robocasa_steering/conceptors/$STEM
    .venv/bin/python experiments/robocasa_steering/build_conceptors.py \
        --activations-dir "$d" \
        --output-npz       experiments/robocasa_steering/conceptors/$STEM/conceptors.npz \
        > /tmp/build_$STEM.log 2>&1 &
done
wait
.venv/bin/python experiments/robocasa_steering/build_and_select_all.py \
    --activations-root /mnt/bird_home/kim34/eval_sweep_results \
    --output-dir       experiments/robocasa_steering/conceptors --skip-build
```

## Stage 4 — Build recipes

Two flavours:

- **`build_recipes.py`** *(new)* — produces `recipes.json`: ONE `(layer, alpha,
  beta)` per `(ckpt, task)` cell, applying the rule "first overlapping alpha
  across epochs (per `common_params__per_task.json`); per-(task, epoch)
  selection for the named exception tasks (CoffeeSetupMug)."
- **`ablation_recipes.json`** — written ad-hoc from `manifest.json` directly,
  using full `selected_alphas` + `selected_betas` lists per `(ckpt, task)`.
  Used when you want every selected (α, β) combo explored in one sweep.

The sbatch reads either schema transparently (auto-detects `alpha` (single)
vs `alphas` (list)).

## Stage 5 — Steering sweep

`experiments/robocasa_steering/steering_sweep/submit_steering_sweep.sh`
fans out one SLURM job per ckpt; each job loops tasks from the recipe and
invokes `experiments/robocasa_steering/steering.py` per task.

**Files:**

| File | Role |
|---|---|
| `submit_steering_sweep.sh` | Top-level launcher. Globs `CHECKPOINTS_DIR/*.ckpt`, queues one sbatch per ckpt pinned to `dj-a40-1.grasp.maas`, with the right `CHECKPOINT`, `CONCEPTOR_NPZ`, `RECIPE_JSON`, `OUTPUT_ROOT` exports. |
| `steering_sweep.sbatch`    | Per-ckpt sbatch (1 GPU, 20 CPUs, 128 GB, 23 h, `dj-high` QoS). Loops tasks from the recipe; per task pulls `(layer, alphas, betas)` from `RECIPE_JSON` and calls `steering.py --strategies $STRATEGIES …`. |
| `build_recipes.py`         | Generates the per-(ckpt, task) recipe used by the sweep (single (α, β) per cell, `common_params`-driven). |
| `recipes.json`             | Default recipe (single value per cell). |
| `ablation_recipes.json`    | Per-(ckpt, task) full lists for ablation sweeps. |

**Sbatch env knobs:**

| Env var | Default | Meaning |
|---|---|---|
| `STRATEGIES` | `per_step` | Space-separated steering strategies passed to `steering.py --strategies`. Mix any of `linear global per_step positive_only random`. |
| `SKIP_BASELINE` | `0` | Set `1` to skip the no-steering baseline run (saves ~half the wall-clock per task). |
| `TASKS_OVERRIDE` | (unset) | Comma-separated list; restrict the task loop to this subset. E.g. `TASKS_OVERRIDE=CloseFridge`. |
| `PER_STEP_STATIC_DS` | (unset) | If set (int), `per_step` strategy uses ONE conceptor at this ds index applied uniformly across denoising steps (matches pi05_libero/robocasa "per_step_K" semantics). Default: time-varying per_step. |
| `ALPHAS_OVERRIDE` | (unset) | Space-separated; replaces recipe's per-task alphas list for this job. |
| `BETAS_OVERRIDE`  | (unset) | Space-separated; replaces recipe's per-task betas list for this job. |
| `NUM_ROLLOUTS` | `30` | Rollouts per condition. |
| `NUM_ENVS` | `15` | Parallel envs per chunk. |
| `SPLIT` | `pretrain` | RoboCasa split. |

**Per-(task, ckpt) output:**
```
$OUTPUT_ROOT/<ckpt_stem>/<task>/
├── summary.json                 # {"task": ..., "conditions": [{"condition": ..., "success_rate": ...}, ...]}
├── _runner_scratch/             # env_runner videos / scratch (large)
└── sweep_args.json              # snapshot of CLI args for reproducibility
```

`steering.py` is **resume-friendly** — it loads `summary.json` on entry and
skips conditions whose names are already present, so re-running the sweep
extends rather than re-runs.

**`steering.py` flags relevant to ablations:**

| Flag | Use |
|---|---|
| `--strategies linear global per_step positive_only random` | Pick which strategies to run. |
| `--layers L [L …]` | Layer indices to sweep at. |
| `--alphas α [α …]` | Alpha grid (used by `global`, `per_step`, `positive_only`). |
| `--betas β [β …]` | Beta grid (mix coefficient `(1-β)I + βC`). |
| `--linear_alphas α [α …]` | Step size for `linear` (ActAdd) strategy. |
| `--skip-baseline` | Skip the no-steering baseline condition. |
| `--per-step-static-ds K` | per_step strategy uses ONE conceptor at ds=K applied uniformly (pi05-style). Without this, per_step is *time-varying*: a different conceptor at each denoising step via `_PerStepLookup`. |
| `--num_rollouts N` | Episodes per condition. |
| `--num_envs M` | Parallel envs (chunks = ceil(N/M)). |

**Submit a sweep:**
```bash
# default per_step strategy across all 5 ckpts:
bash experiments/robocasa_steering/steering_sweep/submit_steering_sweep.sh

# global only, no baseline, just CloseFridge:
STRATEGIES=global SKIP_BASELINE=1 TASKS_OVERRIDE=CloseFridge \
    bash experiments/robocasa_steering/steering_sweep/submit_steering_sweep.sh

# pi05-style per_step at ds=0, one task:
STRATEGIES=per_step PER_STEP_STATIC_DS=0 SKIP_BASELINE=1 TASKS_OVERRIDE=CloseFridge \
    bash experiments/robocasa_steering/steering_sweep/submit_steering_sweep.sh

# full (α, β) ablation using ablation_recipes.json:
RECIPE_JSON=experiments/robocasa_steering/steering_sweep/ablation_recipes.json \
STRATEGIES=global SKIP_BASELINE=1 \
    bash experiments/robocasa_steering/steering_sweep/submit_steering_sweep.sh
```

## Strategy semantics

`steering.py` supports five strategies. The conceptor hook at layer L is
`h' = (1-β)·h + β·(C·h)`; the `linear` strategy uses ActAdd `h' = h + α·v`.

| Strategy | Conceptor source | Hook semantics |
|---|---|---|
| `linear` | `linear_V_contrastive` (mean of success − mean of failure) | `h' = h + α·v` (ActAdd; `--linear_alphas`) |
| `global` | `{task}__L{L}__{α}__C_contrastive` | One static `M = (1-β)I + βC`. |
| `per_step` (default, time-varying) | `{task}__L{L}__{α}__per_step_{ds}__C_contrastive` (alpha-aware lookup; falls back to legacy alpha-less keys) | `M(t)` swaps each denoising step via `_PerStepLookup`; `set_denoise_step(t)` is called inside the patched `conditional_sample`. **More principled for diffusion policy** (textbook per-step). |
| `per_step` + `--per-step-static-ds K` | One matrix at the requested ds | One static `M` applied uniformly. **Matches pi05_libero / pi05_robocasa "per_step_K"** semantics. |
| `positive_only` | `{task}__L{L}__{α}__C_success` (no NOT-failure) | One static `M = (1-β)I + βC_s`. |
| `random` | Random PSD eigenvalues | Sanity-control matched to `global`'s shape. |

## Output / aggregation

Per `(ckpt, task)`: `$OUTPUT_ROOT/<ckpt>/<task>/summary.json`, schema:
```json
{
  "task": "<task>",
  "conditions": [
    {"condition": "baseline",                   "success_rate": 0.5},
    {"condition": "global_L5_a2.0_b0.1",        "success_rate": 0.467},
    {"condition": "global_L5_a2.0_b0.3",        "success_rate": 0.533},
    {"condition": "per_step_L5_a2.0_b0.1",      "success_rate": 0.467},
    {"condition": "per_step_ds0_L5_a2.0_b0.1",  "success_rate": 0.433}
  ]
}
```

To compute a per-(task, epoch) comparison table across all summaries, use:
```bash
.venv/bin/python -c "
import json, pathlib
ROOT = pathlib.Path('/mnt/bird_home/kim34/steering_sweep_results')
for ck in sorted(ROOT.iterdir()):
    if not ck.is_dir(): continue
    for task_dir in sorted(ck.iterdir()):
        s = task_dir / 'summary.json'
        if not s.is_file(): continue
        d = json.load(open(s))
        for c in sorted(d['conditions'], key=lambda x: -x.get('success_rate', -1)):
            print(f'{ck.name}/{task_dir.name}: {c[\"condition\"]:<32s} SR={c[\"success_rate\"]:.3f}')
"
```

## Resource sizing rationale

`steering.py` does **not** collect activations during steering eval — disk
writes per step go away, so per-step rate is ~3× faster than activation
collection (~4.5 it/s vs ~1.6 s/it on dj-a40-1). Per-task wall-clock per
condition (30 rollouts in 2 chunks of 15 envs):

| Task horizon | max_steps (h × 1.5) | est. per-condition time |
|---:|---:|---:|
| 300 | 450 | ~3 min |
| 400 | 600 | ~4 min |
| 500 | 750 | ~5 min |
| 600 | 900 | ~6 min |

Total per ckpt ≈ Σ(condition counts × condition times). With 4 GPUs
concurrent (QoS `dj-high` cap), sweeps with the per-(task, epoch) selected
hyperparameters take ~1.5–2 h wall-clock for the 7-task suite.
