# Experiments

Each subdirectory holds a single environment's mechanistic-interpretability
tuning workflow. Read that subdir's `README.md` to reproduce its output.

For instructions on how to **use** steering in a normal eval run, see
`examples/{libero_env,robocasa_env,metaworld,droid}/README.md` — the `--steer`
flag on `main.py` / `eval_all.py` is the end-user entry point on all four
clients.

## Layout

```
experiments/
├── shared/
│   └── select_parameters.py   # generic overlap-band / quota narrower (used by DROID)
├── libero/
│   ├── compute_conceptors.py  # rebuild NPZ from a fresh activation tree
│   ├── find_best_configs.py   # sweep entrypoint (subprocess-per-condition)
│   ├── best_configs.json      # committed winning (task, layer, α, β, strategy) tuples
│   ├── steering_results/      # gitignored per-run sweep artifacts
│   └── README.md
├── robocasa/                  # same structure as libero/
│   └── ...
├── metaworld/                 # same structure as libero/
│   └── ...
└── droid/
    ├── compute_conceptors.py  # rebuild NPZ from a fresh activation tree
    ├── select_parameters.py   # diagnostic-based narrower (no sweep — real robot)
    ├── selected_params.json   # committed shortlist from select_parameters.py
    └── README.md
```

**Why DROID has no `find_best_configs.py`**: real-robot evaluation is manual
(operator labels success after each rollout), so an automated subprocess grid
search doesn't apply. `select_parameters.py` narrows the grid 10× and the
operator evaluates conditions by hand.

## Shared primitives

All sweep scripts import from `src/openpi/serving/steering.py` (runtime) and
`src/openpi/serving/conceptors.py` (offline computation):

- `SteeredPolicyWrapper`: dispatches on `obs["__steering__"]`
- `ConceptorSteeringHook` / `LinearSteeringHook`: pre-built forward hooks
- `load_conceptor_npz`, `get_conceptor_matrix`, `get_linear_direction`: NPZ helpers
- `DEFAULT_STEERING_{LAYER,ALPHA,BETA,STRATEGY}`: single source of truth for defaults
- `compute_all_conceptors`: offline pipeline for rebuilding the NPZ from fresh
  activations (canonical NPZs are shipped via HuggingFace; rebuild only when
  the checkpoint changes)

## Supported steering strategies

Six strategies sweep across the same (`layer`, `alpha`, `beta`) grid. See the
per-env README's **Running with Steering → Steering strategies** table for the
math. Short summary:

| Strategy | What it does |
|---|---|
| `global` | Contrastive conceptor `C_s ∧ NOT(C_f)` applied uniformly (the default) |
| `per_step` | A different contrastive conceptor at each of pi0.5's 10 denoising steps |
| `positive_only` | Success-only conceptor `C_s` (ablation) |
| `random_matched` | Random-eigenvector conceptor with matched spectrum (control) |
| `linear` | Additive `h + α·v` using unit mean-difference direction (ActAdd baseline) |

## Prereqs (common)

1. Downloaded conceptor NPZs at
   `conceptors/{libero,robocasa,metaworld,droid}_conceptors.npz`. See each
   env's README for the `hf download` command.
2. A PyTorch-converted checkpoint for the target config.
3. A free GPU (`nvidia-smi`, then `export CUDA_VISIBLE_DEVICES=<id>`).

Note: only pi0.5 is currently supported. TODO: extend to

- **pi0-fast**: autoregressive decoder, per-token activations instead of
  per-denoise-step — the `per_step` strategy and NPZ key schema need
  rethinking. Checkpoints live at
  [`brandonyang/pi0fast-metaworld-checkpoints`](https://huggingface.co/brandonyang/pi0fast-metaworld-checkpoints)
  (see `examples/metaworld/README.md`).
- **GR00T N1.5** (`groot_env/`): different DiT backbone with different hook
  attach points and activation dimensionality. Served from its own venv,
  not from `scripts/serve_policy.py`. See `groot_env/README.md`.
