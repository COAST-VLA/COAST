# Experiments

Each subdirectory holds a single environment's mechanistic-interpretability
sweep. Read that subdir's `README.md` to reproduce its `best_configs.json`.

For instructions on how to **use** steering in a normal eval run, see
`examples/libero_env/README.md` or `examples/robocasa_env/README.md` instead —
the `--steer` flag on `main.py` / `eval_all.py` is the end-user entry point.

## Layout

```
experiments/
├── libero/
│   ├── find_best_configs.py   # sweep entrypoint
│   ├── best_configs.json      # committed winning (task, layer, α, β, strategy) tuples
│   ├── steering_results/      # gitignored per-run sweep artifacts
│   └── README.md
└── robocasa/
    ├── find_best_configs.py
    ├── best_configs.json
    ├── steering_results/
    └── README.md
```

## Shared primitives

All sweep scripts import from `src/openpi/serving/steering.py` (runtime) and
`src/openpi/serving/conceptors.py` (offline computation):

- `SteeredPolicyWrapper`: dispatches on `obs["__steering__"]`
- `ConceptorSteeringHook`: pre-builds `M = (1-β)I + β·C`
- `load_conceptor_npz`, `get_conceptor_matrix`: NPZ helpers
- `DEFAULT_STEERING_{LAYER,ALPHA,BETA,STRATEGY}`: single source of truth for defaults
- `compute_all_conceptors`: offline pipeline for rebuilding the NPZ from fresh
  activations (canonical NPZs are shipped via HuggingFace; rebuild only when
  the checkpoint changes)

## Prereqs (common)

1. Downloaded conceptor NPZs at `conceptors/{libero,robocasa}_conceptors.npz`.
   See each env's README for the `hf download` command.
2. A PyTorch-converted checkpoint for the target config.
3. A free GPU (`nvidia-smi`, then `export CUDA_VISIBLE_DEVICES=<id>`).
