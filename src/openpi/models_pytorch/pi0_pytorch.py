import inspect
import logging
import math
from pathlib import Path
import shutil
import sys

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

logger = logging.getLogger(__name__)


def _ensure_transformers_patched():
    """Auto-apply the transformers library patch if not already applied.

    The patch adds AdaRMS support, precision control, and KV cache flexibility
    to the transformers library's Gemma, PaliGemma, and SigLIP models.
    """
    from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

    # Check if patch is already applied (AdaRMS adds cond_dim parameter)
    if "cond_dim" in inspect.signature(GemmaRMSNorm.__init__).parameters:
        return

    import transformers

    logger.info("Auto-applying transformers library patch for AdaRMS/precision/KV-cache support...")
    patch_src = Path(__file__).parent / "transformers_replace"
    transformers_dir = Path(transformers.__file__).parent
    shutil.copytree(patch_src, transformers_dir, dirs_exist_ok=True)

    # Force re-import of patched modules so changes take effect
    for mod_name in list(sys.modules):
        if mod_name.startswith(
            ("transformers.models.gemma", "transformers.models.paligemma", "transformers.models.siglip")
        ):
            del sys.modules[mod_name]
    logger.info("Transformers patch applied successfully.")


_ensure_transformers_patched()

from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel  # noqa: E402


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Transformers patch is auto-applied at module import time by _ensure_transformers_patched()

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    @torch.no_grad()
    def sample_actions_with_intermediates(
        self,
        device,
        observation,
        *,
        noise=None,
        num_steps=10,
        collect_layers=(0, 5, 11, 17),
        steering_hooks=None,
    ) -> tuple[Tensor, dict]:
        """Like sample_actions() but collects per-step intermediates via hooks.

        Does NOT use torch.compile — runs in eager mode for hook compatibility.

        Args:
            steering_hooks: optional list of (layer_idx, hook_callable) pairs.
                When provided, steering hooks are registered BEFORE capture hooks
                so that captures record the post-steering activations. Each
                hook_callable should have an optional set_denoise_step(t) method.

        Returns (final_actions, intermediates_dict).
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # Prefix pass — compute KV cache (same as sample_actions)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Register hooks on Action Expert layers.
        # Steering hooks (if any) are registered FIRST so they fire before
        # capture hooks — capture hooks then record the post-steering output.
        hooks = []
        step_activations = {}
        expert_layers = self.paligemma_with_expert.gemma_expert.model.layers

        if steering_hooks is not None:
            for layer_idx, hook_fn in steering_hooks:
                hooks.append(expert_layers[layer_idx].register_forward_hook(hook_fn))

        def make_output_hook(name):
            """Capture module output (for layer residual streams)."""

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    step_activations[name] = output[0].detach().cpu()
                else:
                    step_activations[name] = output.detach().cpu()

            return hook_fn

        def make_input_hook(name):
            """Capture module input (for MLP hidden = input to down_proj)."""

            def hook_fn(module, input, output):
                step_activations[name] = input[0].detach().cpu()

            return hook_fn

        for i in collect_layers:
            hooks.append(expert_layers[i].register_forward_hook(make_output_hook(f"expert_residual_{i}")))
            hooks.append(
                expert_layers[i].mlp.down_proj.register_forward_hook(make_input_hook(f"expert_mlp_hidden_{i}"))
            )

        try:
            all_x_t, all_v_t, all_adarms_cond = [], [], []
            all_suffix_residual, all_suffix_mlp_hidden = [], []

            dt = -1.0 / num_steps
            dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)

            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            step_counter = 0

            while time >= -dt_tensor / 2:
                expanded_time = time.expand(bsize)
                step_activations.clear()

                # Notify steering hooks of current denoising step
                if steering_hooks is not None:
                    for _, hook_fn in steering_hooks:
                        if hasattr(hook_fn, "set_denoise_step"):
                            hook_fn.set_denoise_step(step_counter)

                # Inline denoise_step to capture adarms_cond
                suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                    state, x_t, expanded_time
                )

                suffix_len = suffix_pad_masks.shape[1]
                batch_size = prefix_pad_masks.shape[0]
                prefix_len = prefix_pad_masks.shape[1]

                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
                suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
                full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

                full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
                self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

                outputs_embeds, _ = self.paligemma_with_expert.forward(
                    attention_mask=full_att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=[None, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                )

                suffix_out = outputs_embeds[1]
                suffix_out = suffix_out[:, -self.config.action_horizon :]
                suffix_out = suffix_out.to(dtype=torch.float32)
                v_t = self.action_out_proj(suffix_out)

                # Capture intermediates
                all_x_t.append(x_t.detach().cpu())
                all_v_t.append(v_t.detach().cpu())
                all_adarms_cond.append(adarms_cond.detach().cpu())

                all_suffix_residual.append(
                    torch.stack([step_activations[f"expert_residual_{i}"] for i in collect_layers])
                )
                all_suffix_mlp_hidden.append(
                    torch.stack([step_activations[f"expert_mlp_hidden_{i}"] for i in collect_layers])
                )

                # Euler step
                x_t = x_t + dt_tensor * v_t
                time += dt_tensor
                step_counter += 1
        finally:
            for h in hooks:
                h.remove()

        intermediates = {
            "all_x_t": torch.stack(all_x_t).float().numpy(),
            "all_v_t": torch.stack(all_v_t).float().numpy(),
            "all_adarms_cond": torch.stack(all_adarms_cond).float().numpy(),
            "all_suffix_residual": torch.stack(all_suffix_residual).float().numpy(),
            "all_suffix_mlp_hidden": torch.stack(all_suffix_mlp_hidden).float().numpy(),
        }
        return x_t, intermediates

    @torch.no_grad()
    def sample_actions_with_intermediates_v2(
        self,
        device,
        observation,
        *,
        noise=None,
        num_steps=10,
        collect_denoise_steps=(0, 4, 9),
        residual_layers=(5, 11),
        mlp_layers=(11,),
        attention_layers=(5, 11),
    ) -> tuple[Tensor, dict]:
        """V2: Selective intermediate collection with attention weights and adaRMS gates.

        Only collects intermediates at specified denoising steps (but runs all steps).
        Captures attention weights and per-layer adaRMS gates in addition to residuals/MLP.

        Returns (final_actions, intermediates_dict).
        intermediates_dict also contains 'adarms_cond_global' (conditioning from step 0).
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # Prefix pass — compute KV cache (same as sample_actions)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # --- Register hooks ---
        hooks = []
        step_activations = {}
        num_expert_layers = len(self.paligemma_with_expert.gemma_expert.model.layers)
        expert_layers = self.paligemma_with_expert.gemma_expert.model.layers

        # Hook factory: residual stream (output of decoder layer)
        def make_residual_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    step_activations[name] = output[0].detach().cpu()
                else:
                    step_activations[name] = output.detach().cpu()

            return hook_fn

        # Hook factory: MLP hidden (input to down_proj)
        def make_mlp_input_hook(name):
            def hook_fn(module, input, output):
                step_activations[name] = input[0].detach().cpu()

            return hook_fn

        # Hook factory: attention weights via output_attentions=True injection
        def make_attn_pre_hook():
            """Forward pre-hook that injects output_attentions=True into kwargs."""

            def hook_fn(module, args, kwargs):
                kwargs["output_attentions"] = True
                return args, kwargs

            return hook_fn

        def make_attn_output_hook(name):
            """Forward hook that captures attention weights (output[1])."""

            def hook_fn(module, input, output):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    step_activations[name] = output[1].detach().cpu()

            return hook_fn

        # Hook factory: adaRMS gate capture
        def make_adarms_gate_hook(name):
            """Capture the gate from GemmaRMSNorm output (normalized_output, gate)."""

            def hook_fn(module, input, output):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    step_activations[name] = output[1].detach().cpu()

            return hook_fn

        # Register residual hooks
        hooks.extend(
            expert_layers[i].register_forward_hook(make_residual_hook(f"expert_residual_{i}")) for i in residual_layers
        )

        # Register MLP hidden hooks
        hooks.extend(
            expert_layers[i].mlp.down_proj.register_forward_hook(make_mlp_input_hook(f"expert_mlp_hidden_{i}"))
            for i in mlp_layers
        )

        # Register attention hooks (pre-hook for injection, post-hook for capture)
        for i in attention_layers:
            hooks.append(expert_layers[i].register_forward_pre_hook(make_attn_pre_hook(), with_kwargs=True))
            hooks.append(expert_layers[i].register_forward_hook(make_attn_output_hook(f"expert_attn_{i}")))

        # Register adaRMS gate hooks for ALL layers (both input_layernorm and post_attention_layernorm)
        for i in range(num_expert_layers):
            hooks.append(
                expert_layers[i].input_layernorm.register_forward_hook(make_adarms_gate_hook(f"adarms_gate_attn_{i}"))
            )
            hooks.append(
                expert_layers[i].post_attention_layernorm.register_forward_hook(
                    make_adarms_gate_hook(f"adarms_gate_mlp_{i}")
                )
            )

        try:
            all_x_t, all_v_t = [], []
            all_suffix_residual, all_suffix_mlp_hidden = [], []
            all_attention_weights, all_adarms_gates = [], []
            adarms_cond_global = None

            dt = -1.0 / num_steps
            dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)

            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            step_counter = 0

            while time >= -dt_tensor / 2:
                expanded_time = time.expand(bsize)
                step_activations.clear()

                # Inline denoise_step to capture adarms_cond
                suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                    state, x_t, expanded_time
                )

                # Capture conditioning vector from step 0 (deterministic)
                if step_counter == 0:
                    adarms_cond_global = adarms_cond.detach().cpu()

                suffix_len = suffix_pad_masks.shape[1]
                batch_size = prefix_pad_masks.shape[0]
                prefix_len = prefix_pad_masks.shape[1]

                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
                suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
                full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

                full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
                self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

                outputs_embeds, _ = self.paligemma_with_expert.forward(
                    attention_mask=full_att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=[None, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                )

                suffix_out = outputs_embeds[1]
                suffix_out = suffix_out[:, -self.config.action_horizon :]
                suffix_out = suffix_out.to(dtype=torch.float32)
                v_t = self.action_out_proj(suffix_out)

                # Only collect intermediates at specified denoising steps
                if step_counter in collect_denoise_steps:
                    all_x_t.append(x_t.detach().cpu())
                    all_v_t.append(v_t.detach().cpu())

                    all_suffix_residual.append(
                        torch.stack([step_activations[f"expert_residual_{i}"] for i in residual_layers])
                    )
                    all_suffix_mlp_hidden.append(
                        torch.stack([step_activations[f"expert_mlp_hidden_{i}"] for i in mlp_layers])
                    )
                    all_attention_weights.append(
                        torch.stack([step_activations[f"expert_attn_{i}"] for i in attention_layers])
                    )

                    # adaRMS gates: (num_expert_layers, 2, batch, seq, dim)
                    # 2 = [input_layernorm (attn), post_attention_layernorm (mlp)]
                    layer_gates = []
                    for i in range(num_expert_layers):
                        attn_gate = step_activations.get(f"adarms_gate_attn_{i}")
                        mlp_gate = step_activations.get(f"adarms_gate_mlp_{i}")
                        # If gate is None (non-adaptive layer), use zeros
                        if attn_gate is None:
                            attn_gate = torch.zeros(bsize, 1, self.config.model.transformer.emb_dim)
                        if mlp_gate is None:
                            mlp_gate = torch.zeros(bsize, 1, self.config.model.transformer.emb_dim)
                        layer_gates.append(torch.stack([attn_gate, mlp_gate]))  # (2, batch, seq, dim)
                    all_adarms_gates.append(torch.stack(layer_gates))  # (num_layers, 2, batch, seq, dim)

                # Euler step
                x_t = x_t + dt_tensor * v_t
                time += dt_tensor
                step_counter += 1
        finally:
            for h in hooks:
                h.remove()

        intermediates = {
            "all_x_t": torch.stack(all_x_t).float().numpy(),
            "all_v_t": torch.stack(all_v_t).float().numpy(),
            "all_suffix_residual": torch.stack(all_suffix_residual).float().numpy(),
            "all_suffix_mlp_hidden": torch.stack(all_suffix_mlp_hidden).float().numpy(),
            "all_attention_weights": torch.stack(all_attention_weights).float().numpy(),
            "all_adarms_gates": torch.stack(all_adarms_gates).float().numpy(),
            "adarms_cond_global": adarms_cond_global.float().numpy(),
        }
        return x_t, intermediates

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    def sample_actions_with_steering(
        self,
        device,
        observation,
        *,
        noise=None,
        num_steps=10,
        steering_hooks=None,
    ) -> tuple[Tensor, dict]:
        """Like sample_actions() but applies steering hooks at specified layers during denoising.

        Runs in eager mode (no torch.compile) for hook compatibility.
        steering_hooks: list of (layer_idx, hook_callable) pairs. Each hook_callable
                must implement __call__(module, input, output) -> modified_output and
                have a set_denoise_step(t) method.

        Returns (final_actions, diagnostics_dict).
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # Prefix pass — compute KV cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Register steering hooks on action expert layers
        hook_handles = []
        expert_layers = self.paligemma_with_expert.gemma_expert.model.layers
        if steering_hooks is not None:
            for layer_idx, hook_fn in steering_hooks:
                handle = expert_layers[layer_idx].register_forward_hook(hook_fn)
                hook_handles.append(handle)

        try:
            dt = -1.0 / num_steps
            dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)

            x_t = noise
            time = torch.tensor(1.0, dtype=torch.float32, device=device)
            step_counter = 0

            while time >= -dt_tensor / 2:
                expanded_time = time.expand(bsize)

                # Notify hooks of current denoising step
                if steering_hooks is not None:
                    for _, hook_fn in steering_hooks:
                        if hasattr(hook_fn, "set_denoise_step"):
                            hook_fn.set_denoise_step(step_counter)

                # Inline denoise_step
                suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                    state, x_t, expanded_time
                )

                suffix_len = suffix_pad_masks.shape[1]
                batch_size = prefix_pad_masks.shape[0]
                prefix_len = prefix_pad_masks.shape[1]

                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
                suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
                full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

                full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
                self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

                outputs_embeds, _ = self.paligemma_with_expert.forward(
                    attention_mask=full_att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=[None, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                )

                suffix_out = outputs_embeds[1]
                suffix_out = suffix_out[:, -self.config.action_horizon :]
                suffix_out = suffix_out.to(dtype=torch.float32)
                v_t = self.action_out_proj(suffix_out)

                # Euler step
                x_t = x_t + dt_tensor * v_t
                time += dt_tensor
                step_counter += 1
        finally:
            for h in hook_handles:
                h.remove()

        return x_t, {}
