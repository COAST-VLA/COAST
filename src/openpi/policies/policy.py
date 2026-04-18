from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def collate_transformed_singles(singles: list[dict]) -> dict:
    # TODO(branyang02): This is hardcoded, but it should be fine??

    # singles: list[dict] where each dict has keys:
    # state (array), tokenized_prompt (array), tokenized_prompt_mask (array),
    # image (dict[str, array]), image_mask (dict[str, array])
    # pi0-fast also adds: token_ar_mask, token_loss_mask
    out = {}

    # Stack flat array fields (include optional pi0-fast fields if present)
    flat_keys = ["state", "tokenized_prompt", "tokenized_prompt_mask"]
    flat_keys.extend(k for k in ["token_ar_mask", "token_loss_mask"] if k in singles[0])
    for k in flat_keys:
        out[k] = jnp.stack([jnp.asarray(ex[k]) for ex in singles], axis=0)

    # Stack nested dict fields
    for k in ["image", "image_mask"]:
        keys = singles[0][k].keys()
        out[k] = {kk: jnp.stack([jnp.asarray(ex[k][kk]) for ex in singles], axis=0) for kk in keys}

    return out


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            if hasattr(model, "sample_actions_with_intermediates"):
                self._sample_actions_with_intermediates = nnx_utils.module_jit(
                    model.sample_actions_with_intermediates
                )
            self._rng = rng or jax.random.key(0)

    @override
    def infer_batched(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        # 1) unbatch -> list of single-example dicts
        eval_batch_size = int(inputs["observation/state"].shape[0])
        singles = []
        for i in range(eval_batch_size):
            ex = {}
            for k, v in inputs.items():
                if k == "prompt":
                    ex[k] = v[i]  # str
                else:
                    ex[k] = v[i]  # array leaf with leading batch dim
            singles.append(ex)
        # 2) run single-example transform per item
        singles = [self._input_transform(ex) for ex in singles]
        # 3) collate back -> batch dict
        inputs = collate_transformed_singles(singles)

        if self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
            sample_rng_or_pytorch_device = self._pytorch_device
        else:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time

        if self._is_pytorch_model:
            outputs = jax.tree.map(
                lambda x: np.asarray(x.detach().cpu()) if isinstance(x, torch.Tensor) else np.asarray(x),
                outputs,
            )
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x), outputs)

        # pi0-fast returns raw tokens of shape (batch, max_decoding_steps). ExtractFASTActions
        # decodes one sample at a time, so apply the output transform per sample and re-stack.
        if outputs["actions"].ndim == 2:
            per_sample_outputs = [
                self._output_transform({"state": outputs["state"][i], "actions": outputs["actions"][i]})
                for i in range(eval_batch_size)
            ]
            outputs = {
                k: np.stack([o[k] for o in per_sample_outputs], axis=0)
                for k in per_sample_outputs[0]
            }
        else:
            outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        if obs["observation/state"].ndim == 2:
            return self.infer_batched(obs, noise=noise)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_with_intermediates(self, obs: dict) -> tuple[dict, dict]:
        """Like infer_batched() but also returns intermediate activations.

        Supports both PyTorch pi0/pi0.5 models (via forward hooks) and JAX pi0-fast
        models (via unrolled autoregressive decoding).
        """
        is_fast = hasattr(self._model, "sample_actions_with_intermediates") and not self._is_pytorch_model

        if not self._is_pytorch_model and not is_fast:
            raise NotImplementedError(
                "infer_with_intermediates requires either a PyTorch model or a JAX pi0-fast model"
            )

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        # 1) unbatch -> list of single-example dicts
        eval_batch_size = int(inputs["observation/state"].shape[0])
        singles = []
        for i in range(eval_batch_size):
            ex = {}
            for k, v in inputs.items():
                if k == "prompt":
                    ex[k] = v[i]
                else:
                    ex[k] = v[i]
            singles.append(ex)
        # 2) run single-example transform per item
        singles = [self._input_transform(ex) for ex in singles]
        # 3) collate back -> batch dict
        inputs = collate_transformed_singles(singles)

        if is_fast:
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            observation = _model.Observation.from_dict(inputs)
            self._rng, sample_rng = jax.random.split(self._rng)
            sample_kwargs = dict(self._sample_kwargs)
            start_time = time.monotonic()
            actions, intermediates = self._sample_actions_with_intermediates(
                sample_rng, observation, **sample_kwargs
            )
            model_time = time.monotonic() - start_time
            # JIT returns fixed-size buffers on the leading axis
            # (max_decoding_steps). Slice down to num_tokens so downstream
            # consumers see tokens/logprobs of length num_tokens and
            # pre_logits of length num_tokens-1 (the EOS-trigger iteration's
            # pre_logits is not meaningful — pre_logits[k] is the hidden
            # state that produced token[k+1], so once all envs have EOS'd we
            # don't need another pre_logit).
            intermediates = jax.tree.map(lambda x: np.asarray(x), intermediates)
            num_tokens = int(intermediates["num_tokens"])
            intermediates["generated_tokens"] = intermediates["generated_tokens"][:num_tokens]
            intermediates["token_logprobs"] = intermediates["token_logprobs"][:num_tokens]
            intermediates["token_pre_logits"] = intermediates["token_pre_logits"][: max(num_tokens - 1, 0)]
            intermediates["num_tokens"] = num_tokens
            # ExtractFASTActions operates per-sample (decodes token sequence to
            # continuous actions), so apply output transforms per sample then re-stack.
            per_sample_outputs = []
            for i in range(eval_batch_size):
                single_out = {
                    "state": np.asarray(inputs["state"][i]),
                    "actions": np.asarray(actions[i]),
                }
                single_out = self._output_transform(single_out)
                per_sample_outputs.append(single_out)
            outputs = {k: np.stack([o[k] for o in per_sample_outputs], axis=0) for k in per_sample_outputs[0]}
        else:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
            observation = _model.Observation.from_dict(inputs)
            start_time = time.monotonic()
            actions, intermediates = self._model.sample_actions_with_intermediates(self._pytorch_device, observation)
            model_time = time.monotonic() - start_time
            outputs = {
                "state": inputs["state"],
                "actions": actions,
            }
            outputs = jax.tree.map(
                lambda x: np.asarray(x.detach().cpu()) if isinstance(x, torch.Tensor) else np.asarray(x),
                outputs,
            )
            outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs, intermediates

    def infer_with_intermediates_v2(self, obs: dict) -> tuple[dict, dict]:
        """Like infer_batched() but returns v2 intermediates (selective steps, attention, adaRMS). PyTorch only."""
        if not self._is_pytorch_model:
            raise NotImplementedError("infer_with_intermediates_v2 is only supported for PyTorch models")

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        # 1) unbatch -> list of single-example dicts
        eval_batch_size = int(inputs["observation/state"].shape[0])
        singles = []
        for i in range(eval_batch_size):
            ex = {}
            for k, v in inputs.items():
                if k == "prompt":
                    ex[k] = v[i]
                else:
                    ex[k] = v[i]
            singles.append(ex)
        # 2) run single-example transform per item
        singles = [self._input_transform(ex) for ex in singles]
        # 3) collate back -> batch dict
        inputs = collate_transformed_singles(singles)

        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions, intermediates = self._model.sample_actions_with_intermediates_v2(self._pytorch_device, observation)
        model_time = time.monotonic() - start_time

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(
            lambda x: np.asarray(x.detach().cpu()) if isinstance(x, torch.Tensor) else np.asarray(x),
            outputs,
        )
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs, intermediates

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
