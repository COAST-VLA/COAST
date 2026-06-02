# COAST

COAST is a VLA evaluation, activation-collection, and steering repo built on [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi). It wires MetaWorld, LIBERO, and RoboCasa examples end-to-end, adds server-side activation collection, and includes a pluggable policy-server layer. RoboCasa can target either the pi0/pi0.5 server or the isolated NVIDIA GR00T server without changing the client. For GR00T setup steps, please follow the [groot_env/README.md](groot_env/README.md). The setup steps below are for the openpi-derived pi0 and pi0.5 models.

https://github.com/user-attachments/assets/343316f1-37d3-464c-ad97-2965f0ebe456

https://github.com/user-attachments/assets/84053a5e-bb96-4368-843a-466db44f5d3d

https://github.com/user-attachments/assets/9fa72915-379a-4107-9759-f9a6794db4b4

## Installation

```bash
git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The base install above is everything `examples/metaworld/` needs. `examples/libero_env/` and `examples/robocasa_env/` ship their own venvs because their dependencies conflict with the root COAST env — see the per-example READMEs for setup.

## Examples

| Environment | README | Notes |
|---|---|---|
| **MetaWorld** | [`examples/metaworld/README.md`](examples/metaworld/README.md) | 50 manipulation tasks (ML45 split). Shares the root COAST venv. Uses server-side activation collection for mechanistic interpretability, plus the ML45 evaluation results table. |
| **LIBERO** | [`examples/libero_env/README.md`](examples/libero_env/README.md) | WebSocket client/server LIBERO benchmark. Has its own Python 3.8 venv (`examples/libero_env/.venv`). |
| **Robocasa** | [`examples/robocasa_env/README.md`](examples/robocasa_env/README.md) | RoboCasa kitchen tasks. Has its own venv (`examples/robocasa_env/.venv`) because robosuite/robocasa conflict with the root COAST venv. Uses a client/server WebSocket eval pattern. |

## Servers

| Model | Entry point | Venv | Notes |
|---|---|---|---|
| **pi0 / pi0-FAST / pi0.5** | [`scripts/serve_policy.py`](scripts/serve_policy.py) | root (`uv sync`) | Primary COAST pi0/pi0.5 server. Serves any `pi05_*` / `pi0_*` training config. Supports `--collect_activations` for mech-interp. |
| **NVIDIA GR00T N1.5** | [`groot_env/serve.py`](groot_env/README.md) | `groot_env/.venv` (Python 3.10) | Serves `nvidia/GR00T-N1.5-3B` or any robocasa365 fine-tuned checkpoint. Has its own venv — N1.5 pins torch 2.5.1 which conflicts with the root COAST env. Uses the same WebSocket protocol as `serve_policy.py`; in this branch the GR00T adapter is wired for RoboCasa only. |

## Support

What each client + model combination supports today. This table excludes
baseline-only experiment branches. `✅` means the code path is implemented and
intended to run; `❌` means the training config, input transform, checkpoint, or
serve integration is not wired up in this branch. `E2E` in the notes means a
real server + simulator client rollout was run locally on this branch.

### Training

We run fine-tuning in-repo for MetaWorld and LIBERO, and release a series of **intermediate-step checkpoints** for both — a span of checkpoints across training is what lets downstream mech-interp work compare behavior as the policy learns, rather than just inspecting the fully-trained endpoint. For RoboCasa and DROID we skip in-repo training and evaluate against the upstream fully-trained checkpoints directly.

| Client | In-repo training | Dataset | Train configs | Checkpoints |
|---|---|---|---|---|
| **MetaWorld** | ✅ | [`brandonyang/metaworld_ml45`](https://huggingface.co/datasets/brandonyang/metaworld_ml45) | `pi05_metaworld`, `pi0_fast_metaworld` | intermediate (released on HF — see `examples/metaworld/README.md`) |
| **LIBERO** | ✅ | [`physical-intelligence/libero`](https://huggingface.co/datasets/physical-intelligence/libero) | `pi05_libero`, `pi0_fast_libero` | intermediate (released on HF — see `examples/libero_env/README.md`) |
| **RoboCasa** | ❌ | — | — | upstream [`robocasa/robocasa365_checkpoints`](https://huggingface.co/robocasa/robocasa365_checkpoints) |
| **DROID** | ❌ | — | — | upstream `gs://openpi-assets/checkpoints/pi05_droid`, `gs://openpi-assets/checkpoints/pi0_fast_droid` |

### Simulator Evaluation, Collection, Steering

MetaWorld, LIBERO, and RoboCasa route collection through a
`--collect_activations` policy server. Protocol, output directory layout,
per-schema file lists (`v1` / `fast_v1` / `groot_v1`), and verification are all
in [`docs/activation_collection.md`](docs/activation_collection.md).
The DROID real-robot client is documented separately in
[`examples/droid/README.md`](examples/droid/README.md) and is intentionally not
part of this simulator matrix.

Steering is served by `scripts/serve_policy.py --steer` for pi0/pi0.5/pi0-FAST
and by `groot_env/serve.py --steer` for RoboCasa GR00T N1.5. pi0/pi0.5
steering uses PyTorch hooks and requires `--pytorch`; pi0-FAST steering is
JAX-only and must run without `--pytorch`, using fast conceptors built from
`token_pre_logits` (`fast_v1` activations). GR00T steering uses PyTorch hooks on
the action DiT residual stream and conceptors built from `groot_v1`
`dit_hidden_states.npz` activations.

| Client | Model | Naive eval | Activation collection | Steering eval | Notes |
|---|---|---:|---:|---:|---|
| **MetaWorld** | pi0.5 | ✅ | ✅ `v1` | ✅ PyTorch hooks | `pi05_metaworld`; collection and steering are separate server modes. |
| **MetaWorld** | pi0-FAST | ✅ | ✅ `fast_v1` | ✅ JAX pre-logit | `pi0_fast_metaworld`; steering E2E: `reach-v3`, 3 full episodes, 2/3 success. |
| **MetaWorld** | GR00T-N1.5 | ❌ | ❌ | ❌ | No GR00T MetaWorld policy/server wiring. |
| **LIBERO** | pi0.5 | ✅ | ✅ `v1` | ✅ PyTorch hooks | `pi05_libero`; collection and steering are separate server modes. |
| **LIBERO** | pi0-FAST | ✅ | ✅ `fast_v1` | ✅ JAX pre-logit | `pi0_fast_libero`; steering E2E: `libero_10` task 0, 3 full episodes, 2/3 success; one nonfatal FAST detokenization warning observed. |
| **LIBERO** | GR00T-N1.5 | ❌ | ❌ | ❌ | No GR00T LIBERO policy/server wiring in this branch. |
| **RoboCasa** | pi0.5 | ✅ | ✅ `v1` | ✅ PyTorch hooks | `pi05_robocasa`; upstream RoboCasa checkpoint. |
| **RoboCasa** | pi0-FAST | ❌ | ❌ | ❌ | No `pi0_fast_robocasa` config/checkpoint/client path is wired. |
| **RoboCasa** | GR00T-N1.5 | ✅ | ✅ `groot_v1` | ✅ DiT hooks | Served from `groot_env/`; conceptors use `groot_v1` DiT residuals. |

## Repo Layout

- `src/openpi/` — model code (JAX primary, PyTorch in `models_pytorch/`), training configs, policies, and the WebSocket policy server.
- `scripts/` — shared training and serving entry points (`train.py`, `train_pytorch.py`, `serve_policy.py`, `compute_norm_stats.py`). Env-specific scripts live under each `examples/<env>/` directory.
- `examples/` — per-environment client quickstarts. See the Clients table above.
- `groot_env/` — isolated-venv NVIDIA GR00T N1.5 server. See [`groot_env/README.md`](groot_env/README.md) for setup.
- `third_party/Isaac-GR00T/` — submodule pinned to `n1.5-release`; installed editable from `groot_env/`.
- `tests/` — pytest suite. Run with `uv run pytest --strict-markers -m "not manual"` for CI-equivalent (env tests requiring GPU + EGL are marked `manual` and skipped).
- `docs/` — design notes and historical implementation docs.
- `CLAUDE.md` — guidance for Claude Code agents working in this repo.

## Upstream Base

COAST builds on Physical Intelligence's [`openpi`](https://github.com/Physical-Intelligence/openpi) codebase while adding simulator-specific evaluation, activation collection, and steering workflows.
