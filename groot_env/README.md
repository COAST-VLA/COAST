# GR00T N1.5 Server

An isolated-venv server that serves [NVIDIA Isaac GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release) models over openpi's WebSocket protocol, so any openpi env client (robocasa, libero, metaworld, droid, ...) can target GR00T without any client-side changes.

## Setup

From the repo root:

```bash
# 1. Pull the Isaac-GR00T submodule (first time only).
git submodule update --init --recursive

# 2. Create the groot_env venv. uv auto-installs Python 3.10 per pyproject.toml.
cd groot_env
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 3. flash-attn needs torch present at build time, so install it WITHOUT build
#    isolation (matches NVIDIA's official install guide).
uv pip install --no-build-isolation flash-attn==2.7.1.post4

# 4. Download a robocasa365 N1.5 checkpoint (~8 GB incl. optimizer state).
#    This one is the GR00T analog of pi05's pi05_pretrain_human300/multitask_learning/75000.
cd ..
uv run hf download robocasa/robocasa365_checkpoints \
    --include "gr00t_n1-5/multitask_learning/checkpoint-120000/*" \
    --local-dir checkpoints/groot_n15
```

Alternative checkpoints under `gr00t_n1-5/` on the same HF repo:

| Path | Recipe |
|---|---|
| `multitask_learning/checkpoint-120000` *(default)* | 365-task multitask pre-training. Reported atomic-seen mean: **43.0%**. Direct analog of pi05's `pi05_pretrain_human300/multitask_learning/75000`. |
| `foundation_model_learning/pretraining/checkpoint-80000` | Foundation-model-learning recipe pretraining (65 atomic tasks). |
| `foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000` | Same pretraining + post-training on the `atomic_seen` task set. Reported mean: **68.5%**. |
| `lifelong_learning/phase{1..4}/checkpoint-*` | Sequential / lifelong-learning experiments. |

## Serving

From `groot_env/`:

```bash
uv run python serve.py --port 8000
```

`serve.py` defaults to the `multitask_learning/checkpoint-120000` you downloaded above; override with `--model-path`. Full flags:

```bash
uv run python serve.py \
    --model-path ../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000 \
    --embodiment robocasa \
    --device cuda:0 \
    --port 8000 \
    --denoising-steps 4         # NVIDIA's default; matches their published numbers
```

Collection mode (saves per-step activations to disk):

```bash
uv run python serve.py \
    --collect-activations \
    --output-dir ../groot_n15-robocasa-activations-v1-15env
# …then run any client with --collect (see examples/robocasa_env/README.md).
```

The metadata the server reports on connection (`get_server_metadata()`):

```json
{
  "backend": "groot_n15",
  "model_path": "…/checkpoint-120000",
  "embodiment": "robocasa",
  "denoising_steps": 4,
  "collection_mode": "v1",       // only when --collect-activations
  "checkpoint_step": "checkpoint-120000"
}
```

---

## Running evaluations

Because the WebSocket protocol is identical to `scripts/serve_policy.py`, existing clients target GR00T with **no edits**:

```bash
# In a second terminal, with GR00T serving on :8000
cd examples/robocasa_env
MUJOCO_GL=egl uv run python main.py --env_name OpenDrawer --num_episodes 15

# Or the task-set driver:
MUJOCO_GL=egl uv run python eval_all.py --task_set atomic_seen --num_episodes 15 --num_workers 5
```

That's it. Videos + `results.json` go under `output/...` exactly as for pi05.

---

## Activation collection

`serve.py --collect-activations` wraps the policy in `activation_collector.CollectingPolicy`, which rejects plain inference requests and only accepts calls carrying `__collect__` or `__finalize_episode__` magic keys. The robocasa client's `--collect` flag already emits these (see `openpi_client.collection_session.CollectionSession`).

The on-disk layout mirrors pi0's `activations_v1` schema where possible, with GR00T-specific renames for the per-step `.npz` files:

```
<output_dir>/checkpoint-120000/<task_name>/episode_NNN_env_NNN/
├── metadata.json            # task_name, episode_id, episode_success, total_reward, total_inference_steps, prompt, …
├── rewards.npz              # per_step_reward, cumulative_reward, success_at_step
└── step_NNNN/
    ├── metadata.json        # step, inference_step, cumulative_reward, success_so_far
    ├── denoising.npz        # all_x_t (D+1,H,A) fp32, all_v_t (D,H,A) fp32
    ├── backbone_cond.npz    # backbone_features (S,C) fp16 — VL backbone output, GR00T analog of pi0's adarms_cond
    └── dit_hidden_states.npz  # all_dit_hidden_states (D,L+1,S,C) fp16 — DiT per-layer residuals, pi0 analog: suffix_residual
```

Why this differs from pi0's schema: GR00T uses cross-attention to a variable-length VL backbone sequence instead of pi0's pooled AdaRMS conditioning, and captures the full DiT residual stream (16 layers + input) instead of just pi0's 4 "suffix" layers. GR00T's `suffix_mlp_hidden` analog is not captured by default — adding it would roughly double the per-step disk and require forward hooks on each `BasicTransformerBlock.ff`.

### Verifying activations

```bash
# Fast schema + shape + finiteness check (parses every .npz).
uv run python verify_activations.py ../groot_n15-robocasa-activations-v1-15env

# pytest suite — same shape as tests/test_activations.py for pi0.
ACTIVATIONS_DIR=../groot_n15-robocasa-activations-v1-15env/checkpoint-120000/OpenDrawer \
    uv run pytest tests/test_groot_activations.py -v
```

`tests/test_groot_activations.py` covers directory layout, metadata fields, reward-array/length agreement, the flow-matching signature (x_t norm decreases across denoising steps), and cross-episode variation.

---

## Reference: robocasa atomic-seen (15 ep/task, multitask_learning/checkpoint-120000)

| Task | **GR00T N1.5 (here)** | pi05 (`examples/robocasa_env/figures/results_75000.json`) | GR00T published |
|---|:-:|:-:|:-:|
| PickPlaceCounterToStove | **60%** | 47% | 63.2% |
| PickPlaceCounterToCabinet | **53%** | 47% | 47.5% |
| OpenDrawer | **53%** | 60% | 81.1% |
| OpenStandMixerHead | **33%** | 67% | — |
| CloseFridge | **27%** | 13% | — |
| TurnOnElectricKettle | **27%** | 13% | — |
| CoffeeSetupMug | **20%** | 27% | 31.0% |
| **Mean (these 7)** | **39%** | 39% | — |

Robocasa's published N1.5 multitask atomic-seen average is **43.0%** over all 18 tasks.

---

## How this integrates (for reference)

- `serve.py` — thin wrapper that loads the checkpoint and starts a `WebsocketPolicyServer`. Mirrors `scripts/serve_policy.py` on the pi0 side.
- `groot_adapter.py` — `GR00TAdapterPolicy(BasePolicy)` that translates openpi's flat `observation/*`-keyed client dict to GR00T's nested `{video, state, language}` dict, runs `Gr00tPolicy.get_action`, and concatenates GR00T's per-action-key dict back to a single `(action_horizon, action_dim)` array under the `"actions"` key.
- `activation_collector.py` — the CollectingPolicy wrapper that serializes per-step intermediates. Shares its on-disk schema with `src/openpi/serving/activation_collector.py` where the semantics match.
- `websocket_policy_server.py` — copied verbatim from `src/openpi/serving/` so `groot_env/` can serve without pulling in JAX/flax.
- `tests/test_groot_activations.py` — GR00T N1.5 analog of `tests/test_activations.py` for the different schema.
