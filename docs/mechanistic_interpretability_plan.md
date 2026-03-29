# Mechanistic Interpretability Plan for Pi0.5

This document outlines the plan for performing mechanistic interpretability analysis on the pi0.5 model fine-tuned on MetaWorld.

## Relevant Literature

### Directly on Pi0/Pi0.5

1. **"Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models"** (Mar 2026, [arXiv:2603.19183](https://arxiv.org/html/2603.19183v1)) — Trains SAEs directly on pi0.5. Extracts residual stream from PaliGemma layers 0, 5, 11, 17 (d=2048) and Action Expert layers 0, 5, 11, 17 (d=1024). Uses TopK SAE with AuxK auxiliary loss. Key findings: 97% of features are memorized on LIBERO; PaliGemma middle layers (5, 11) are most interpretable; Action Expert features organize around manipulation phases.

2. **"Not All Features Are Created Equal"** (Mar 2026, [arXiv:2603.19233](https://arxiv.org/html/2603.19233v1)) — Cross-architecture study including pi0.5 with 424 SAEs across 394k+ rollouts. **Critical finding: per-token SAE processing is essential; mean-pooling destroys action fidelity (96% -> 8% on pi0.5).** Expert pathways encode motor programs; VLM pathways encode goal semantics.

3. **"Mechanistic Interpretability for Steering VLAs"** (Aug 2025, [arXiv:2509.00328](https://arxiv.org/abs/2509.00328)) — Projects FFN value vectors onto the embedding space, finding sparse semantic directions (speed, direction) causally linked to action selection. Tested on pi0.

### Diffusion / Flow Matching Interpretability

4. **TIDE: Temporal-Aware SAEs for Diffusion Transformers** (Mar 2025, [arXiv:2503.07050](https://arxiv.org/html/2503.07050v1)) — Activations at different timesteps are different distributions, requiring timestep-dependent modulation. Discovers coarse-to-fine generation with abrupt phase transitions.

5. **"Emergent World Representations in OpenVLA"** (Sep 2025, [arXiv:2509.24559](https://arxiv.org/abs/2509.24559)) — Linear probes on residual stream activations predict state transitions. Middle layers encode world dynamics best.

### Multimodal Interpretability

6. **"Survey on MI for Multi-Modal Foundation Models"** (Feb 2025, [arXiv:2502.17516](https://arxiv.org/html/2502.17516v1))
7. **"Circuit Tracing in VLMs"** (Feb 2026, [arXiv:2602.20330](https://arxiv.org/abs/2602.20330))
8. **"SAEs Learn Monosemantic Features in VLMs"** (NeurIPS 2025, [arXiv:2504.02821](https://arxiv.org/abs/2504.02821))

---

## What Makes Pi0.5 Different from Standard LLM Interp

| Aspect | Standard LLM | Pi0.5 | Implication |
|---|---|---|---|
| Output space | Discrete tokens | Continuous actions via denoising | Track activations across 10 denoising steps |
| Temporal structure | Autoregressive, one token at a time | All 32 action tokens generated simultaneously | Analyze spatial patterns across action horizon |
| Conditioning | Just context | Timestep via adaRMS | Gate values reveal temporal layer specialization |
| Experts | Single model | Two experts sharing attention | Cross-expert patching; KV cache is the "bridge" |
| Evaluation | Next-token accuracy | Task success in environment | Need rollouts to evaluate interventions |

---

## Activations to Collect

### Priority 1: Residual Stream

| Activation | Shape per example | Where | Why |
|---|---|---|---|
| PaliGemma residual stream (layers 0,5,11,17) | `(prefix_len, 2048)` x 4 | After each block, prefix pass | Task/visual semantics; best layers for SAEs per literature |
| Action Expert residual stream (layers 0,5,11,17) | `(32, 1024)` x 4 x 10 steps | After each block, per denoising step | Action formation; motor programs |
| Denoising trajectory `x_t` | `(32, 32)` x 10 steps | In `sample_actions` while loop | How actions refine from noise to plan |
| Velocity predictions `v_t` | `(32, 32)` x 10 steps | After `action_out_proj` | What the model "wants to do" at each step |

### Priority 2: adaRMS Modulation (Unique to Pi0.5)

| Activation | Shape | Where | Why |
|---|---|---|---|
| adaRMS scale, shift, gate | `(1, 1024)` x 3 x 18 layers x 2 (attn+FFN) x 10 steps | `RMSNorm` in `gemma.py:128` | Timestep conditioning mechanism. The gate in `_gated_residual(x + y * gate)` controls per-layer contribution per timestep. |

### Priority 3: MLP Hidden Activations

The Gemma FFN uses a gated MLP: `output = (GELU(x @ W_gate) * (x @ W_up)) @ W_down`. The "MLP hidden layer" is the post-gating activation `GELU(x @ W_gate) * (x @ W_up)`, which lives in the high-dimensional space (`mlp_dim`) before projection back down. This is where feature mixing and nonlinear computation happen.

| Activation | Shape | Where | Why |
|---|---|---|---|
| PaliGemma MLP hidden (layers 0,5,11,17) | `(prefix_len, 16384)` x 4 | `lora.FeedForward.__call__` line `activations = gate_value * ff1` in `lora.py:138`, prefix pass | High-dim feature space where VLM computes over visual/language representations. 16384-d is 8x the residual stream width — this is where the model has the most expressive capacity. |
| Action Expert MLP hidden (layers 0,5,11,17) | `(32, 4096)` x 4 x 10 steps | Same location, suffix pass per denoising step | High-dim feature space where action expert computes motor programs. 4096-d is 4x the residual stream width. Gating pattern reveals which features are active per timestep. |

**Storage note:** MLP hidden activations are large. PaliGemma: `prefix_len x 16384 x 4 bytes ≈ 60 MB` per layer. Action Expert: `32 x 4096 x 4 bytes ≈ 0.5 MB` per layer per step. Collecting 4 layers of the Action Expert across 10 steps adds ~20 MB per inference call (manageable). PaliGemma MLP hidden is computed once per inference call (not per denoising step), so 4 layers adds ~240 MB per inference call (expensive — consider collecting only layers 5 and 11, or subsampling tokens).

**Practical approach:** Start with Action Expert MLP hidden only (cheap). Add PaliGemma MLP hidden for a subset of tasks/steps if storage permits.

### Priority 4: Attention Patterns

| Activation | Shape | Where | Why |
|---|---|---|---|
| Action->Image attention weights | `(8 heads, 32, ~768)` x 18 layers x 10 steps | `Attention` in `gemma.py:226` | Which image patches drive which actions |
| Action->Language attention weights | `(8 heads, 32, ~200)` x 18 layers x 10 steps | Same | How task description influences actions |
| KV cache (prefix) | `(18, prefix_len, 1, 256)` x 2 | After prefix forward pass | The "world representation" -- computed once, reused across all denoising steps |

### Priority 5: FFN Gate Activations (pre-multiplication)

| Activation | Shape | Where | Why |
|---|---|---|---|
| FFN gate pre-activation (Action Expert) | `(32, 4096)` x 18 layers x 10 steps | `GELU(x @ W_gate)` before multiplication with `x @ W_up` in `lora.py:131` | Which gating units are open/closed — a sparser, more interpretable view than the full MLP hidden. Useful for identifying dead neurons and feature selectivity. |

---

## Activation Site Map

### Prefix (computed once per inference)

**SigLIP Vision Encoder** (27 blocks, width 1152):
- Input: `[b, 224, 224, 3]` per camera (3 cameras)
- Patch embeddings: `[b, 256, 1152]` per camera
- Per-block outputs (27 layers): attention output, FFN output, residual stream
- Final output: `[b, 256, 1152]` per camera -> projected to PaliGemma width

**PaliGemma Expert** (18 blocks, width 2048):
- Language embedding: `[b, max_token_len, 2048]`
- Concatenated prefix tokens: `[b, prefix_len, 2048]` (images + language)
- Per-block: pre-attn norm, Q/K/V, attention logits/weights, attention output, post-attn residual, pre-FFN norm, FFN gate/output, post-FFN residual
- KV cache stored: `(18, b, prefix_len, 1, 256)` x 2 (keys, values)

### Suffix (computed per denoising step, 10 steps)

**Action Expert** (18 blocks, width 1024):
- Action token projection: `[b, 32, 1024]` from `action_in_proj`
- Time embedding: `[b, 1024]` from `posemb_sincos` -> `time_mlp_in` -> swish -> `time_mlp_out` -> swish
- adaRMS conditioning: `[b, 1024]` passed to every RMSNorm in action expert
- Per-block: same intermediates as PaliGemma but with adaRMS scale/shift/gate
- Final velocity: `[b, 32, 32]` from `action_out_proj`

### Total Activation Count

- One-time prefix intermediates: ~750 tensors (SigLIP + prefix transformer + KV caches)
- Per-step suffix intermediates: ~370 tensors (time embedding + action projection + suffix transformer)
- Denoising steps: 10
- Total per inference: ~4,400 unique tensors

---

## Data Collection Strategy

### Overview

Collect activations during evaluation rollouts on all 45 ML45 train tasks (in-distribution).
The trained checkpoint is at `checkpoints/pi05_metaworld/pi05_metaworld_test/5000/`.

We use the **5000-step checkpoint** because it has a good mix of successes and failures (mean success rate 75.6%), which is ideal for comparing "good" vs "bad" activations. A fully converged checkpoint would have too few failures to analyze.

### Evaluation Results (5000-step checkpoint)

**Perfect tasks (100%):** button-press-topdown-v3, button-press-topdown-wall-v3, button-press-v3, button-press-wall-v3, coffee-button-v3, door-close-v3, door-open-v3, drawer-close-v3, drawer-open-v3, faucet-open-v3, handle-press-side-v3, handle-press-v3, plate-slide-v3, plate-slide-side-v3, push-wall-v3, sweep-v3, window-open-v3, window-close-v3

**High success (67-93%):** coffee-pull-v3, pick-place-v3, peg-unplug-side-v3, reach-wall-v3, reach-v3, shelf-place-v3, push-v3, faucet-close-v3, push-back-v3, coffee-push-v3, disassemble-v3, hammer-v3, lever-pull-v3, peg-insert-side-v3, stick-pull-v3, sweep-into-v3, plate-slide-back-v3, plate-slide-back-side-v3

**Low success (0-40%):** basketball-v3 (40%), handle-pull-side-v3 (40%), handle-pull-v3 (33%), pick-place-wall-v3 (33%), stick-push-v3 (33%), pick-out-of-hole-v3 (27%), assembly-v3 (20%), soccer-v3 (20%), dial-turn-v3 (0%)

This distribution enables direct comparison of activations between:
- Tasks the model has fully learned vs tasks it struggles with
- Successful vs failed rollouts within partially-learned tasks (67-93% range)

### Evaluation Setup

Uses **PyTorch inference** via `scripts/collect_activations.py` which loads the policy in-process (no WebSocket server). The JAX checkpoint is auto-converted to PyTorch format (via `--pytorch` flag or `ensure_pytorch_checkpoint()`).

**Why PyTorch:** The JAX implementation had multiple blockers for activation collection (NNX graph inflation, `nn.scan` hiding per-layer outputs, `sow()`/NNX-bridge compatibility). PyTorch's `register_forward_hook` natively supports per-layer extraction with no workarounds.

Key setup:
- **`SyncVectorEnv`** with `num_envs=2` (AsyncVectorEnv forks cause deadlocks)
- **`register_forward_hook`** on target layers — activations captured and moved to CPU via `.detach().cpu()` each denoising step
- **No `torch.compile`** for the intermediates path — hooks work on underlying modules in eager mode
- Hook targets for Action Expert: `model.paligemma_with_expert.gemma_expert.model.layers[i]` and `.layers[i].mlp`
- Hook targets for PaliGemma: `model.paligemma_with_expert.paligemma.model.language_model.layers[i]`

At each `policy.infer_with_intermediates()` call, we have:
- The observation (images, state, prompt)
- The environment step index
- Per-denoising-step intermediates (x_t, v_t, adaRMS conditioning, per-layer residuals, MLP outputs), already on CPU

### Labeling Activations: "Good" vs "Bad"

Activations are stored with outcome labels for success vs failure comparison:

**Per-episode labels (coarse) — in `episode_*/metadata.json`:**
- `task_name`: which ML45 task
- `episode_id`, `env_id`: identifiers
- `episode_success`: bool — whether the environment's `success` flag was set to True at any point
- `total_reward`: float — cumulative reward for the episode
- `steps_to_success`: int — environment step at which success was first triggered (-1 if failed)
- `total_env_steps`: int — total environment steps in the episode
- `total_inference_steps`: int — total policy inference calls in the episode
- `prompt`: the natural language task description
- `checkpoint_dir`, `config_name`: provenance

**Per-episode reward trajectory — in `episode_*/rewards.npz`:**
- `per_step_reward`: (total_env_steps,) — reward at each environment step
- `cumulative_reward`: (total_env_steps,) — running sum of rewards
- `success_at_step`: (total_env_steps,) — bool, whether success was achieved by that step

This enables plotting reward curves alongside activation trajectories for individual episodes.

**Per-inference-call labels (fine-grained) — in `step_*/metadata.json`:**
- `task_name`: which ML45 task (e.g., "reach-v3")
- `episode_id`: episode index (0 for Phase 1)
- `env_id`: which parallel environment (0 or 1 with num_envs=2)
- `step`: environment step at which this inference was called
- `inference_step`: sequential index of this inference call within the episode
- `prompt`: the natural language task description
- `cumulative_reward`: float — total reward accumulated up to this inference call
- `success_so_far`: bool — whether success has been achieved before this inference call
- `reward_since_last_inference`: float — reward accumulated since the previous inference call

This labeling enables several analyses:
- Compare activations from successful vs failed episodes of the **same task** (controls for task identity)
- Compare activations from the **same task at different timesteps** (early vs late in episode)
- Compare activations from tasks with **high vs low success rates** (easy vs hard tasks)
- Track how activations evolve **within a single episode** as the robot approaches success
- **Correlate activations with reward signal** — do activations change when reward starts increasing? Can early activations predict eventual success?

### Storage Format

```
activations/
  {checkpoint_step}/
    {task_name}/
      episode_{episode_id:03d}_env_{env_id:03d}/
        metadata.json              # episode-level labels (see above)
        rewards.npz                # per-step reward trajectory (see above)
        step_{step:04d}/
          denoising.npz            # all_x_t: (10, 32, 32), all_v_t: (10, 32, 32)
          adarms_cond.npz          # all_adarms_cond: (10, 1024) — timestep conditioning input
          suffix_residual.npz      # Action Expert post-layer residual at layers 0,5,11,17: (10, 4, 32, 1024)
          suffix_mlp_hidden.npz    # Action Expert MLP hidden (pre-down_proj, 4096-d) at layers 0,5,11,17: (10, 4, 32, 4096)
          prefix_residual.npz      # (optional) PaliGemma residual at layers 0,5,11,17: (4, ~968, 2048)
          metadata.json            # step-level labels with cumulative_reward, success_so_far (see above)
```

### Collection Scope

**Phase 1 (PyTorch — fresh collection):**
- All 45 ML45 train tasks, 2 envs per task, 1 episode each = 90 rollouts
- Collect: denoising trajectories (x_t, v_t) + adaRMS conditioning + Action Expert per-layer residuals + MLP hidden (4096-d, pre-down_proj) at layers 0,5,11,17
- All collected via `register_forward_hook` in a single pass — no separate "Step 4"
- MLP hidden captured by hooking `down_proj` and taking its **input** (= `act_fn(gate_proj(x)) * up_proj(x)`)
- Estimated: ~30 MB per inference call, ~15 inference calls per episode, ~45 tasks ≈ **~14 GB total** (without prefix)
- With prefix residual (PaliGemma): ~40 GB total

**Phase 2 (expanded):**
- Multiple episodes for tasks of interest (high/low success rates) with `num_envs=5+`
- Add PaliGemma per-layer residuals for select tasks (~30 MB per prefix pass)
- Add per-layer adaRMS scale/shift/gate and attention patterns
- Potentially collect from multiple checkpoint iterations (e.g., 1000, 5000, 10000 steps)

---

## Analysis Pipeline

### Phase 1: Collect & Visualize
- Collect all activations via PyTorch `register_forward_hook` (denoising trajectory, adaRMS conditioning, per-layer residuals, MLP outputs)
- Visualize denoising trajectories for successful vs failed episodes side-by-side
- Plot adaRMS conditioning norm/cosine-similarity across denoising steps
- Compare x_t variance decay curves between successful and failed episodes
- Visualize Action Expert residual stream PCA across layers and denoising steps

### Phase 2: Probe & Compare
- **Success vs failure analysis**: compare mean activations, PCA/t-SNE clustering of residual streams between successful and failed episodes of the same task
- Linear probes on residual stream for task identity, object position, gripper state
- Train a linear probe to **predict episode outcome (success/failure) from early-step activations** -- can the model's internal state predict whether it will succeed?
- Track cosine similarity between consecutive denoising steps (identifies phase transitions)
- Compare denoising dynamics: do successful episodes show different phase transition patterns?
- Activation patching: swap prefix KV cache between tasks, null out language tokens

### Phase 3: SAEs
- Train TopK SAEs on Action Expert layers 0, 5, 11, 17 (**per-token, NOT mean-pooled**)
- Train PaliGemma SAEs on middle layers (5, 11)
- Classify features as memorized vs. general
- Identify features that are differentially active in successful vs failed episodes
- Attempt feature steering

---

## Implementation Notes

- **Using PyTorch** (`src/openpi/models_pytorch/pi0_pytorch.py`) for all activation collection — `register_forward_hook` natively supports per-layer extraction
- JAX checkpoint auto-converts to PyTorch via `--pytorch` flag on `serve_policy.py` or `ensure_pytorch_checkpoint()` in `src/openpi/models_pytorch/convert.py`
- Hook targets for Action Expert layers: `model.paligemma_with_expert.gemma_expert.model.layers[i]` (18 layers, width 1024, mlp_dim 4096)
- Hook targets for PaliGemma layers: `model.paligemma_with_expert.paligemma.model.language_model.layers[i]` (18 layers, width 2048, mlp_dim 16384)
- `adaRMS` in the Action Expert: `GemmaRMSNorm(cond_dim=1024)` produces `(normed_output, gate)` — the gate controls per-layer contribution per timestep
- The KV cache is the "bridge" between the VLM world and the action world — computed once in the prefix pass, reused across all 10 denoising steps
- `torch.compile(mode="max-autotune")` is applied to `sample_actions` but activation collection bypasses this by calling model methods directly — hooks work on the underlying modules
- Per-token SAEs are essential for action tokens; consider mean-pooling only for image tokens to manage memory
- See `docs/activation_collection_implementation.md` for the full implementation plan with code examples
