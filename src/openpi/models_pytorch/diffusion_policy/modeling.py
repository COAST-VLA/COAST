"""openpi wrapper around the vendored ``DiffusionTransformerHybridImagePolicy``.

The wrapper exposes two methods that openpi's training / serving loops call:

- ``forward(observation, actions) -> loss_tensor``
- ``sample_actions(device, observation) -> actions``

Responsibilities beyond those:
- Translate openpi's ``Observation`` (canonical ``images`` dict + 1-D ``state``) into the
  shape_meta-keyed batch dict that the vendored policy expects.
- Own a ``LinearNormalizer`` so loads of the robocasa-benchmark .ckpt (which stores one
  per-key normalizer) succeed bit-exactly, and so fresh training runs can fit a normalizer
  from their first data batch.
- Handle ``.ckpt`` checkpoint loading (the upstream serialization format) in addition to the
  plain state_dict path openpi uses elsewhere.

The translation between openpi's flat state tensor and the per-key lowdim obs entries of the
shape_meta is a simple concat/split driven by the ``lowdims`` tuple in the config. The
ordering of that tuple is load-bearing — it defines the byte layout of ``observation.state``
produced by the data pipeline.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from openpi.models_pytorch.diffusion_policy.config import DiffusionPolicyConfig
from openpi.models_pytorch.diffusion_policy.vendored import (
    robomimic_extensions,  # noqa: F401 — registration side-effect
)
from openpi.models_pytorch.diffusion_policy.vendored.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)


def _make_noise_scheduler(config: DiffusionPolicyConfig):
    """Build the DDPMScheduler that matches the vendored policy's expectations."""
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        variance_type=config.variance_type,
        clip_sample=config.clip_sample,
        prediction_type=config.prediction_type,
    )


class DiffusionPolicy(nn.Module):
    """Transformer-Hybrid Diffusion Policy, packaged for openpi.

    This is a thin wrapper over the vendored ``DiffusionTransformerHybridImagePolicy``. The
    vendored module contains the actual parameters; we mediate between openpi's observation
    format and the vendored module's shape_meta-keyed obs dict.
    """

    def __init__(self, config: DiffusionPolicyConfig):
        super().__init__()
        self.config = config

        # Build the underlying policy. Kwargs mirror the vendored config exactly.
        self._policy = DiffusionTransformerHybridImagePolicy(
            shape_meta=config.shape_meta_dict(),
            noise_scheduler=_make_noise_scheduler(config),
            horizon=config.horizon,
            n_action_steps=config.n_action_steps,
            n_obs_steps=config.n_obs_steps,
            num_inference_steps=config.num_inference_steps,
            vision_backbone=config.vision_backbone,
            crop_shape=tuple(config.crop_shape),
            obs_encoder_group_norm=config.obs_encoder_group_norm,
            eval_fixed_crop=config.eval_fixed_crop,
            n_layer=config.n_layer,
            n_cond_layers=config.n_cond_layers,
            n_head=config.n_head,
            n_emb=config.n_emb,
            p_drop_emb=config.p_drop_emb,
            p_drop_attn=config.p_drop_attn,
            causal_attn=config.causal_attn,
            time_as_cond=config.time_as_cond,
            obs_as_cond=config.obs_as_cond,
        )

    # ---- weight loading ----------------------------------------------------------------------

    def load_weights(self, weight_path: str) -> None:
        """Load weights from either a robocasa-benchmark .ckpt or an openpi safetensors file.

        The .ckpt format is the one the upstream training loop saves: a torch.save'd dict with
        keys ``cfg``, ``state_dicts.{model, ema_model, optimizer}``, and ``pickles``. We load
        ``state_dicts.model`` (the EMA shadow is available at ``state_dicts.ema_model`` if you
        ever want it — the pretrain_human300 checkpoint has EMA enabled, so both exist).
        """
        weight_path = str(weight_path)
        if weight_path.endswith(".ckpt"):
            payload = torch.load(weight_path, map_location="cpu", weights_only=False)
            state_dict = payload["state_dicts"]["model"]
            res = self._policy.load_state_dict(state_dict, strict=True)
            # strict=True above would have raised on mismatches; this branch is defensive.
            if res.missing_keys or res.unexpected_keys:
                raise RuntimeError(
                    f"unexpected key mismatch loading {weight_path}: "
                    f"missing={res.missing_keys[:5]}, unexpected={res.unexpected_keys[:5]}"
                )
            return
        if weight_path.endswith(".safetensors") or Path(weight_path).is_dir():
            # openpi's own checkpoint format: either a dir with model.safetensors, or the file
            # directly.
            import safetensors.torch

            if Path(weight_path).is_dir():
                weight_path = str(Path(weight_path) / "model.safetensors")
            safetensors.torch.load_model(self, weight_path)
            return
        raise ValueError(f"Unrecognized weight_path format: {weight_path}")

    # ---- obs translation ---------------------------------------------------------------------

    def _openpi_obs_to_batch(
        self,
        observation,
        lang_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Turn openpi ``Observation`` (+ optional ``lang_emb``) into the per-key dict the vendored policy expects.

        Output convention: every value is ``(B, n_obs_steps, ...)``. We duplicate a single
        openpi time-step across n_obs_steps, since openpi's data pipeline is single-step.

        ``lang_emb`` is threaded as a kwarg rather than sitting on ``Observation`` because
        openpi's Observation dataclass types ``tokenized_prompt`` as an integer tensor — stuffing
        a float language embedding through that field trips its runtime typechecker. Keeping
        lang_emb separate also lets envs without language conditioning (metaworld, libero) pass
        it as None.
        """
        obs_dict: dict[str, torch.Tensor] = {}
        n_obs_steps = self.config.n_obs_steps

        for img_spec in self.config.images:
            img = observation.images[img_spec.key]
            img = _to_nchw_01(img)
            # Resize to the declared camera H/W. The policy has its own CropRandomizer for train
            # and a fixed center-crop at eval — we do NOT crop here.
            if img.shape[-2:] != (img_spec.height, img_spec.width):
                img = nn.functional.interpolate(
                    img, size=(img_spec.height, img_spec.width), mode="bilinear", align_corners=False
                )
            obs_dict[img_spec.key] = img.unsqueeze(1).expand(-1, n_obs_steps, -1, -1, -1).contiguous()

        state = observation.state
        offset = 0
        for lowdim in self.config.lowdims:
            slice_ = state[..., offset : offset + lowdim.dim]
            obs_dict[lowdim.key] = slice_.unsqueeze(1).expand(-1, n_obs_steps, -1).contiguous()
            offset += lowdim.dim
        if offset != state.shape[-1]:
            raise ValueError(f"state dim mismatch: lowdims sum to {offset}, observation.state has {state.shape[-1]}")

        if self.config.lang_emb_dim is not None:
            if lang_emb is None:
                raise RuntimeError("config.lang_emb_dim is set but no lang_emb was passed to forward/sample_actions")
            if lang_emb.ndim == 2:
                lang_emb = lang_emb.unsqueeze(1).expand(-1, n_obs_steps, -1).contiguous()
            obs_dict["lang_emb"] = lang_emb.to(dtype=next(iter(obs_dict.values())).dtype)

        return obs_dict

    # ---- training: forward ------------------------------------------------------------------

    def forward(
        self,
        observation,
        actions: torch.Tensor,
        *,
        lang_emb: torch.Tensor | None = None,
        **_ignored,
    ) -> torch.Tensor:
        """Return per-element MSE loss shaped ``(B, horizon, action_dim)``.

        Shape matches ``PI0Pytorch.forward`` so openpi's training loop (which does
        ``loss.mean()``) works unchanged. Internally we call the vendored ``compute_loss``, which
        normalizes inputs via the in-model ``LinearNormalizer`` — callers pass the raw
        un-normalized actions.
        """
        obs_dict = self._openpi_obs_to_batch(observation, lang_emb=lang_emb)
        batch = {"obs": obs_dict, "action": actions.to(torch.float32)}
        # compute_loss returns a scalar mean; expand to (B, H, A) so openpi's .mean() stays sound.
        scalar_loss = self._policy.compute_loss(batch)
        return scalar_loss.expand(actions.shape[0], self.config.horizon, self.config.action_dim).clone()

    # ---- inference: sample_actions -----------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        *,
        lang_emb: torch.Tensor | None = None,
        **_ignored,
    ) -> torch.Tensor:
        """Return sampled actions, shape ``(B, horizon, action_dim)``.

        Returns the full ``action_pred`` (horizon length) rather than the already-sliced
        ``action[:, n_action_steps]`` — openpi downstream consumers (e.g., the metaworld client)
        slice the first ``replan_steps`` entries of the returned chunk themselves.
        """
        obs_dict = self._openpi_obs_to_batch(observation, lang_emb=lang_emb)
        result = self._policy.predict_action(obs_dict)
        return result["action_pred"]

    # ---- inference: sample_actions_from_dict -------------------------------------------------

    @torch.no_grad()
    def sample_actions_from_dict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Inference entry point that bypasses openpi's typed Observation.

        ``obs_dict`` must already be shape_meta-keyed with per-key tensors of shape
        ``(B, n_obs_steps, ...)`` — i.e., exactly what ``DiffusionTransformerHybridImagePolicy``
        expects. Useful for the robocasa inference path where the client has direct access to
        language embeddings and robocasa-specific keys.

        Returns ``action_pred`` (B, horizon, action_dim). The vendored policy also computes
        ``action`` (B, n_action_steps, action_dim), but downstream usually wants to slice the
        full chunk itself.
        """
        result = self._policy.predict_action(obs_dict)
        return result["action_pred"]

    # ---- normalizer management ---------------------------------------------------------------

    @torch.no_grad()
    def fit_normalizer(
        self,
        observations: list,
        actions_list: list[torch.Tensor],
        lang_embs: list[torch.Tensor] | None = None,
    ) -> None:
        """Fit the in-model ``LinearNormalizer`` from a list of openpi observations + actions.

        Must be called exactly once before training starts (and before the first ``forward`` call
        that would otherwise trip the normalizer's "stats are infinity" assertion). Samples
        should come from the real training data — a handful of batches is enough to get a
        usable min/max per key for the ``limits``-mode normalizer this policy uses.

        Skipped at inference time when the checkpoint we're loading already carries a
        populated normalizer.
        """
        from openpi.models_pytorch.diffusion_policy.vendored.normalizer import LinearNormalizer

        data: dict[str, torch.Tensor] = {}
        # Actions
        data["action"] = torch.cat([a.to(torch.float32) for a in actions_list], dim=0)

        # Images: pool all observations per key, convert to NCHW [0, 1], resize to declared shape.
        for img_spec in self.config.images:
            pieces = []
            for obs in observations:
                img = obs.images[img_spec.key]
                img = _to_nchw_01(img)
                if img.shape[-2:] != (img_spec.height, img_spec.width):
                    img = nn.functional.interpolate(
                        img, size=(img_spec.height, img_spec.width), mode="bilinear", align_corners=False
                    )
                pieces.append(img)
            data[img_spec.key] = torch.cat(pieces, dim=0)

        # Lowdim: slice state per declared lowdim spec, in order.
        offset = 0
        for lowdim in self.config.lowdims:
            slices = [obs.state[..., offset : offset + lowdim.dim] for obs in observations]
            data[lowdim.key] = torch.cat(slices, dim=0)
            offset += lowdim.dim

        # lang_emb (optional)
        if self.config.lang_emb_dim is not None:
            if not lang_embs:
                raise RuntimeError("lang_emb_dim is set but no lang_embs were provided to fit_normalizer")
            data["lang_emb"] = torch.cat(lang_embs, dim=0)

        normalizer = LinearNormalizer()
        normalizer.fit(data)
        self._policy.set_normalizer(normalizer)


def _to_nchw_01(image: torch.Tensor) -> torch.Tensor:
    """openpi image tensor -> (B, C, H, W) float32 in [0, 1].

    Input contract: either ``uint8`` in ``[0, 255]`` or ``float`` in ``[-1, 1]`` (the
    canonical form after ``openpi.models.model.Observation.from_dict``). NHWC and NCHW both
    accepted — auto-detected by the channel dim.
    """
    if image.dtype == torch.uint8:
        if image.ndim == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        return image.to(torch.float32) / 255.0
    if image.ndim == 4 and image.shape[-1] == 3 and image.shape[1] != 3:
        image = image.permute(0, 3, 1, 2)
    return (image / 2.0 + 0.5).clamp(0.0, 1.0)
