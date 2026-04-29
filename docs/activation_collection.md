# Activation Collection

The canonical reference for collecting intermediate activations from the VLA
policies in this repo (pi0 / pi0.5 / pi0-FAST / GR00T N1.5). Per-client READMEs
link here instead of duplicating the protocol.

## Overview

All clients use a **server-side** collection pattern: a "collection-mode"
policy server wraps the policy in a `CollectingPolicy` and writes files on
the **server's** filesystem; the client only sends `__collect__` /
`__finalize_episode__` metadata over WebSocket.

| Client | Per-inference batch | `__collect__` shape | Entry point |
|---|---|---|---|
| MetaWorld   | `num_envs` parallel envs share one inference call | **list** of N per-env entries | `examples/metaworld/main.py --collect`, `examples/metaworld/eval_all.py --collect` |
| LIBERO      | 1 env per inference                                | **dict** (single env)          | `examples/libero_env/main.py --collect`, `examples/libero_env/eval_all.py --collect` |
| RoboCasa    | 1 env per inference                                | **dict** (single env)          | `examples/robocasa_env/main.py --collect`, `examples/robocasa_env/eval_all.py --collect` |
| DROID       | 1 env per inference                                | **dict** (single env)          | DROID client                         |
| GR00T N1.5  | 1 env per inference                                | **dict** (single env)          | `groot_env/serve.py --collect_activations` |

The server (`scripts/serve_policy.py --collect_activations`,
`groot_env/serve.py --collect_activations`) auto-dispatches on the
`__collect__` shape: a `dict` is treated as a single-env call, a `list` as a
batched call where each entry maps to a slice of the batch dim of the
captured intermediates. The on-disk tree is identical either way.

Client-side helpers in `openpi_client.collection_session`:
- `CollectionSession` — single-env bookkeeping (LIBERO / RoboCasa / DROID).
- `BatchCollectionSession` — `num_envs`-env bookkeeping (MetaWorld). Owns N
  per-env states and emits `__collect__` / `__finalize_episode__` as a list.

The **on-disk schema is the same** regardless of client, so downstream
analysis tooling is client-agnostic. Only the per-step `.npz` file set
differs — those are schema-versioned (`v1` / `fast_v1` / `groot_v1`) and
picked automatically from the model type.

### Collection-mode identifiers

Reported on the server's `collection_mode` metadata field (and stamped into
per-step `metadata.json` for `fast_v1`):

| `collection_mode` | Model family | Per-step file set |
|---|---|---|
| `v1`      | pi0, pi0.5 (diffusion)        | `denoising.npz`, `adarms_cond.npz`, `suffix_residual.npz`, `suffix_mlp_hidden.npz`, `metadata.json` |
| `fast_v1` | pi0-FAST (autoregressive)     | `tokens.npz`, `hidden_states.npz`, `token_logprobs.npz`, `metadata.json` |
| `groot_v1`| GR00T N1.5 (DiT + VL backbone) | `denoising.npz`, `backbone_cond.npz`, `dit_hidden_states.npz`, `dit_mlp_hidden.npz`, `metadata.json` |

## Output directory layout

The server writes the same tree for every client. Configure the root with
`--output-dir` on the server CLI.

```
<output_root>/<checkpoint_step>/<task_name>/
├── episode_NNN_env_NNN/
│   ├── metadata.json    # task_name, episode_id, env_id, episode_success,
│   │                    # total_reward, steps_to_success, total_env_steps,
│   │                    # total_inference_steps, prompt, checkpoint_dir, config_name
│   ├── rewards.npz      # per_step_reward, cumulative_reward, success_at_step
│   └── step_NNNN/
│       ├── metadata.json    # step, inference_step, cumulative_reward, success_so_far, ...
│       └── *.npz            # activation tensors (schema-dependent — see below)
```

Where:
- `<checkpoint_step>` is `pathlib.Path(policy_dir).name` (e.g. `5000`,
  `pi05_libero`, `checkpoint-120000`).
- `<task_name>` is the client-supplied task identifier (validated — see
  [Task-name validation](#task-name-validation)).
- `episode_NNN_env_NNN` uses `episode_{episode_id:03d}_env_{env_id:03d}`.
- `step_NNNN` uses `step_{step:04d}` (the env step at which the inference
  call was issued; not every env step has a `step_*` dir because of
  action-chunking — only inference calls do).

## Schema reference

### `v1` — pi0 / pi0.5 (diffusion)

Written by `save_step_activations` in
`src/openpi/serving/activation_collector.py`.
`D = num_denoising_steps`, `L = num_suffix_layers`, `H = action_horizon`,
`A = action_dim`, `C = hidden_dim`, `F = ff_inner_dim`.

| File | Array keys | Shape | Dtype |
|---|---|---|---|
| `denoising.npz`         | `all_x_t`, `all_v_t`       | `(D, H, A)`       | fp32 |
| `adarms_cond.npz`       | `all_adarms_cond`          | `(D, C)`          | fp32 |
| `suffix_residual.npz`   | `all_suffix_residual`      | `(D, L, H, C)`    | fp32 |
| `suffix_mlp_hidden.npz` | `all_suffix_mlp_hidden`    | `(D, L, H, F)`    | fp32 |

### `fast_v1` — pi0-FAST (autoregressive)

Written by `save_step_activations_fast`. `T = num_tokens` (variable per step).

| File | Array keys | Shape | Dtype |
|---|---|---|---|
| `tokens.npz`         | `generated_tokens`  | `(T,)`       | int32   |
| `hidden_states.npz`  | `token_pre_logits`  | `(T-1, C)`   | fp16    |
| `token_logprobs.npz` | `token_logprobs`    | `(T,)`       | fp32    |

`metadata.json` additionally carries `num_tokens` and
`collection_version="fast_v1"`. `hidden_states.npz` is omitted when
`num_tokens == 1` (no forward pass produced a usable hidden state).

### `groot_v1` — GR00T N1.5 (DiT + VL backbone)

Written by `save_step_activations` in
`groot_env/groot_activation_collector.py`. `D = num_denoising_steps`
(default 4), `L = num_dit_layers`, `H = action_horizon`, `A = padded_action_dim`,
`S = state + future + action token count`, `C = dit_hidden_dim`,
`F = ff_inner_dim`.

| File | Array keys | Shape | Dtype |
|---|---|---|---|
| `denoising.npz`         | `all_x_t`, `all_v_t`       | `(D, H, A)`     | fp32 |
| `backbone_cond.npz`     | `backbone_features`        | `(S, C)`        | fp16 |
| `dit_hidden_states.npz` | `all_dit_hidden_states`    | `(D, L, S, C)`  | fp16 |
| `dit_mlp_hidden.npz`    | `all_dit_mlp_hidden`       | `(D, L, S, F)`  | fp16 |

**Note on `backbone_cond`:** GR00T cross-attends the DiT to a VL backbone
sequence that's computed **once per inference** (not per denoising step), so
its shape is `(S, C)`. Pi0's `adarms_cond` is a per-step pooled conditioning
vector of shape `(D, C)`. Everything else is semantically aligned — per-layer
residual streams and MLP expansions across denoising steps.

## Wire protocol

The collection-mode server (`scripts/serve_policy.py --collect_activations` or
`groot_env/serve.py --collect_activations`) is **collection-only**: every
WebSocket request must carry exactly one of `__collect__` or
`__finalize_episode__` on the obs dict, or the server raises `ValueError` and
closes the connection.

The transport is standard openpi WebSocket (`msgpack_numpy` serialization,
same `policy.infer(obs_dict)` shape). The magic keys are pulled off before
dispatch.

Two client-side helpers in `openpi_client.collection_session` cover both
batching modes:
- `CollectionSession` — one env per inference call. Used by LIBERO,
  RoboCasa, DROID. `__collect__` is a single dict.
- `BatchCollectionSession` — `num_envs` envs per inference call (vectorized
  rollouts). Used by MetaWorld. `__collect__` is a list of `num_envs` dicts;
  obs arrays are pre-batched with shape `(num_envs, ...)`.

Writing a custom client means speaking one of the two specs below.

### Server metadata (on connect)

The collection-mode server publishes its config in the WebSocket greeting
metadata so clients can discover the checkpoint identity. Read via
`client.get_server_metadata()`.

```python
{
    "policy_dir":      "/home/.../checkpoints/pi05_libero",
    "config_name":     "pi05_libero",
    "collection_mode": "v1",            # "fast_v1" for pi0-FAST; "groot_v1" for GR00T N1.5
    "model_type":      "pi05",          # "pi0" / "pi05" / "pi0_fast" / "groot_n15"
    "checkpoint_step": "pi05_libero",
    "output_root":     "/abs/path/to/activations",
}
```

The GR00T server additionally publishes `backend`, `model_path`, `embodiment`,
and `denoising_steps` (all pre-startup config).

### Per-step inference call: `__collect__`

Send a normal inference obs dict plus a `__collect__` field. The server runs
`policy.infer_with_intermediates(obs)` once, writes per-env activations +
`metadata.json`, and returns the actions like a normal infer call. Two
shapes are accepted:

#### Single-env (LIBERO / RoboCasa / DROID)

`__collect__` is a single dict; obs arrays are 1-D / unbatched (the server
adds the leading batch dim).

```python
{
    # --- normal inference fields (single example; server adds the batch dim) ---
    "observation/image":       <H, W, 3 uint8>,
    "observation/wrist_image": <H, W, 3 uint8>,
    "observation/state":       <state_dim float32>,
    "prompt":                  "<task description>",

    # --- collection metadata (required) ---
    "__collect__": {
        "task_name":                   "pick_up_the_alphabet_soup_...",
        "episode_id":                  0,
        "env_id":                      0,
        "step":                        47,
        "inference_step":              8,
        "prompt":                      "<task description>",
        "cumulative_reward":           0.0,
        "success_so_far":              false,
        "reward_since_last_inference": 0.0,
    },
}
```

Server response:

```python
{"actions": <action_horizon, action_dim float32>, "policy_timing": {...}}
```

#### Batched (MetaWorld)

`__collect__` is a list of `num_envs` dicts; obs arrays are pre-batched with
shape `(num_envs, ...)`. The server runs **one** forward pass and slices
`env_id` from the batch dim of each intermediate per entry — preserving the
throughput of vectorized rollouts.

```python
{
    "observation/image":       <num_envs, H, W, 3 uint8>,
    "observation/wrist_image": <num_envs, H, W, 3 uint8>,
    "observation/state":       <num_envs, state_dim float32>,
    "prompt":                  ["<prompt>"] * num_envs,

    "__collect__": [
        {  # entry per env_id, same fields as the single-env case
            "task_name":                   "reach-v3",
            "episode_id":                  0,
            "env_id":                      env_id,
            "step":                        47,
            "inference_step":              8,
            "prompt":                      "<task description>",
            "cumulative_reward":           0.0,
            "success_so_far":              false,
            "reward_since_last_inference": 0.0,
        }
        for env_id in range(num_envs)
    ],
}
```

Server response keeps the batch dim on `actions`:

```python
{"actions": <num_envs, action_horizon, action_dim float32>, "policy_timing": {...}}
```

The list length must equal the obs batch dim, otherwise the server raises
`ValueError("Batch size mismatch: ...")`.

### Per-episode finalization call: `__finalize_episode__`

After the rollout loop ends (success, max-steps, or exception), send a
**separate** call with only the `__finalize_episode__` field. The server
skips the model entirely, writes `episode_NNN_env_NNN/metadata.json` +
`rewards.npz`, and returns an ack (no `actions` key).

Like `__collect__`, the field accepts a single dict (single-env) or a list
of dicts (batched).

#### Single-env finalize

```python
{
    "__finalize_episode__": {
        "task_name":             "...",
        "episode_id":            0,
        "env_id":                0,
        "prompt":                "...",
        "episode_success":       true,
        "total_reward":          1.0,
        "steps_to_success":      152,         # index of first done; -1 if never
        "total_env_steps":       153,         # == len(per_step_reward)
        "total_inference_steps": 28,          # count of __collect__ calls
        "per_step_reward":       [0.0, 0.0, ..., 1.0],
        "per_step_success":      [false, ..., true],
    },
}
```

Server response:

```python
{"ack": True, "episode_dir": "<absolute path>"}
```

#### Batched finalize

```python
{
    "__finalize_episode__": [
        {  # entry per env_id, same fields as the single-env case
            "task_name":             "reach-v3",
            "episode_id":            0,
            "env_id":                env_id,
            "prompt":                "...",
            "episode_success":       ...,
            "total_reward":          ...,
            "steps_to_success":      ...,
            "total_env_steps":       ...,
            "total_inference_steps": ...,
            "per_step_reward":       [...],
            "per_step_success":      [...],
        }
        for env_id in range(num_envs)
    ],
}
```

Server response:

```python
{"ack": True, "episode_dirs": ["<env 0 path>", "<env 1 path>", ...]}
```

### State ownership

The collection server is **stateless between requests**. It only knows
`output_root`, `checkpoint_step`, `policy_dir`, and `config_name` (from
its startup args). Everything else — current task, inference_step counter,
per-step rewards — lives on the **client**. `CollectionSession` /
`BatchCollectionSession` are the reference implementations of that
bookkeeping.

Multiple clients can talk to one collection server simultaneously as long
as their `(task_name, episode_id, env_id, step)` tuples don't collide.
Vectorized clients (MetaWorld) avoid intra-batch collisions because every
list entry has a distinct `env_id`.

### Task-name validation

The server rejects `task_name` values containing path traversal (`..`),
absolute paths (`/tmp/foo`), nested paths (`a/b`), or backslashes — this
prevents writes outside `<output_root>`. See `_sanitize_task_name` in
`src/openpi/serving/activation_collector.py`.

### Error responses

- Neither magic key present → `ValueError("Collection-only server requires either __collect__ or __finalize_episode__ ...")`
- Both present → `ValueError("Request contains both __collect__ and __finalize_episode__; only one is allowed per call.")`
- Invalid `task_name` → `ValueError("Invalid task_name {!r}: ...")`
- Single-env path on a multi-example obs → `ValueError("Collection mode only supports single-example inputs ... Send one inference call per env.")`
- Batched path with a 1-D obs → `ValueError("Batched collection requires pre-batched obs ...")`
- Batched path with batch dim != list length → `ValueError("Batch size mismatch: ...")`
- Empty `__collect__` / `__finalize_episode__` list → `ValueError("... must not be empty.")`

In all cases the WebSocket layer sends the traceback back as a string frame
and closes the connection with `INTERNAL_ERROR`.
`WebsocketClientPolicy.infer` surfaces this as a `RuntimeError` with the
server traceback inlined.

## Running collection

Start a collection-mode policy server in one terminal, run the client with
`--collect` in another. Every client uses the same launch pattern; the only
difference is which `--policy.config` you point the server at and which
client venv you run from.

```bash
# Terminal 1 — pi0.5 diffusion (PyTorch required for activation collection):
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py --pytorch --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi05_libero \
    --policy.dir=/path/to/checkpoint

# Terminal 1 (alternative) — pi0-FAST autoregressive (JAX only):
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py --collect_activations \
    --output-dir ./activations \
    policy:checkpoint --policy.config=pi0_fast_libero \
    --policy.dir=/path/to/checkpoint

# Terminal 1 (alternative) — GR00T N1.5 (PyTorch-only, no --pytorch flag needed):
cd groot_env
export CUDA_VISIBLE_DEVICES=0
uv run python serve.py --port 8000 --collect_activations \
    --model-path /path/to/checkpoint \
    --output-dir ../activations
```

```bash
# Terminal 2 — client, any of (defaults shown; override with --env_name / --task_suite_name / --task_set / --tasks):
MUJOCO_GL=egl uv run examples/metaworld/eval_all.py --collect --split subset --num_envs 15
(cd examples/libero_env && MUJOCO_GL=egl uv run python eval_all.py --collect --num_workers 5)
(cd examples/robocasa_env && MUJOCO_GL=egl uv run python eval_all.py --collect --num_workers 5)
```

The metaworld client sends list-shaped `__collect__` payloads (one entry per
env in the vectorized batch), so a single `--collect_activations` server
serves all `num_envs` rollouts from one forward pass. LIBERO / RoboCasa
clients send dict-shaped payloads; their `eval_all.py` parallelizes by
spawning subprocess workers, all hitting the same shared server.

Notes:

- `--pytorch` is **required** for pi0 / pi0.5 collection (forward hooks are
  PyTorch-only) and **forbidden** for pi0-FAST (no PyTorch port of the
  autoregressive decode). `groot_env/serve.py` is PyTorch-only by
  construction; no `--pytorch` flag exists.
- A collection-mode server **rejects plain inference**. If you also want
  regular eval, run a separate non-collection server on a different port.
- The server's `--output-dir` is on the **server's** filesystem — client
  and server can be on different machines; the client never touches the
  activation files.

## Verification

Env-var-driven pytest suites validate a collected tree. Skipped in CI when
`ACTIVATIONS_DIR` is unset.

```bash
# pi0 / pi0.5 diffusion (v1 schema — denoising.npz / adarms_cond.npz /
# suffix_residual.npz / suffix_mlp_hidden.npz):
ACTIVATIONS_DIR=./activations/5000/reach-v3 \
    uv run pytest tests/test_activations.py -v

# pi0-FAST autoregressive (fast_v1 schema — tokens.npz / hidden_states.npz /
# token_logprobs.npz). Uses ACTIVATIONS_FAST_DIR, not ACTIVATIONS_DIR:
ACTIVATIONS_FAST_DIR=./activations/pi0fast-metaworld-activations-v1-ml45train-16env/2500/reach-v3 \
    uv run pytest tests/test_activations_fast.py -v

# GR00T N1.5 (groot_v1 schema):
cd groot_env
ACTIVATIONS_DIR=../activations/groot_n15-robocasa-activations-v1-15env/checkpoint-120000/OpenDrawer \
    uv run pytest tests/test_groot_activations.py -v
```

To sweep every task in a dataset, loop the same command over each
per-task subdirectory.

## Pre-collected datasets

Ready-made HuggingFace datasets you can download with `hf download` instead
of running your own collection:

| Backend | Client | Dataset | Notes |
|---|---|---|---|
| pi0.5 (`v1`)     | MetaWorld | [`brandonyang/pi05-metaworld-activations-v1-ml45train-16env`](https://huggingface.co/datasets/brandonyang/pi05-metaworld-activations-v1-ml45train-16env) | 16 envs × 45 ML45-train tasks |
| pi0-FAST (`fast_v1`) | MetaWorld | [`brandonyang/pi0fast-metaworld-activations-v1-ml45train-16env`](https://huggingface.co/datasets/brandonyang/pi0fast-metaworld-activations-v1-ml45train-16env) | 16 envs × 45 ML45-train tasks |
| pi0.5 (`v1`)     | LIBERO    | [`brandonyang/pi05-libero-activations-v1-2000-15env`](https://huggingface.co/datasets/brandonyang/pi05-libero-activations-v1-2000-15env) | 2000-step checkpoint, 10 tasks × 15 episodes |
| pi0-FAST (`fast_v1`) | LIBERO | [`brandonyang/pi0fast-libero-activations-v1-2000-15env`](https://huggingface.co/datasets/brandonyang/pi0fast-libero-activations-v1-2000-15env) | libero_10, 2000-step, 1.1 GB, mean success 0.65 |
| pi0.5 (`v1`)     | RoboCasa  | [`ksb21st/robocasa-activations-75000`](https://huggingface.co/datasets/ksb21st/robocasa-activations-75000) | 7 tasks × 15 episodes, `pi05_pretrain_human300/multitask_learning/75000` |
| GR00T N1.5 (`groot_v1`) | RoboCasa | [`brandonyang/groot_n15-robocasa-activations-v1-15env`](https://huggingface.co/datasets/brandonyang/groot_n15-robocasa-activations-v1-15env) | 7 tasks × 15 episodes, `gr00t_n1-5/multitask_learning/checkpoint-120000` |
