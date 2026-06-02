"""Adapter that wraps NVIDIA GR00T N1.5 as an openpi BasePolicy.

GR00T N1.5 has a flat observation API:
    {"video.<cam>": np.ndarray (T, H, W, 3) uint8,
     "state.<key>": np.ndarray (T, D) float32,
     "annotation.<lang_key>": list[str]}

Returns:
    {"action.<key>": np.ndarray (T, D) float32}

The openpi robocasa client (`examples/robocasa_env/main.py`) sends a different
flat dict tailored to pi05:
    {"observation/image": (H,W,3) uint8,        # agentview_left
     "observation/wrist_image": (H,W,3) uint8,  # eye_in_hand
     "observation/state": (16,) float32,        # concat'd proprioception
     "prompt": str}

This adapter translates between the two. To preserve pi05 compatibility we don't
change the client, even though GR00T N1.5's robocasa head was trained with TWO
side cameras (left + right). We synthesize the missing right side view by
duplicating the left view; success rate may suffer slightly versus the native
3-camera input but the geometry is close enough that the policy still works.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy

# GR00T N1.5 robocasa modality keys (verified from
# checkpoints/groot_n15/.../experiment_cfg/metadata.json):
#   video: robot0_agentview_left, robot0_agentview_right, robot0_eye_in_hand
#   state: end_effector_position_relative(3) + end_effector_rotation_relative(4 quat)
#          + gripper_qpos(2) + base_position(3) + base_rotation(4 quat) = 16
#   action: end_effector_position(3) + end_effector_rotation(3 axis-angle)
#           + gripper_close(1) + base_motion(4) + control_mode(1) = 12
ROBOCASA_VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
ROBOCASA_STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
)
ROBOCASA_ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)
ROBOCASA_LANGUAGE_KEY = "annotation.human.action.task_description"


def build_robocasa_state_dict(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Split the openpi robocasa client's 16-D state vector into GR00T N1.5 state keys.

    The openpi robocasa client (`build_state` in main.py) concatenates:
        ee_pos(3) + ee_rot_quat(4) + base_pos(3) + base_rot_quat(4) + gripper_qpos(2) = 16
    which matches the env's `PandaOmronKeyConverter.map_obs` ordering.
    """
    state = np.asarray(obs["observation/state"], dtype=np.float32)
    if state.shape[-1] != 16:
        raise ValueError(
            f"Expected 16-D robocasa state "
            f"(ee_pos3+ee_rot_quat4+base_pos3+base_rot_quat4+gripper_qpos2), "
            f"got shape {state.shape}."
        )
    return {
        "state.end_effector_position_relative": state[0:3],
        "state.end_effector_rotation_relative": state[3:7],
        "state.base_position": state[7:10],
        "state.base_rotation": state[10:14],
        "state.gripper_qpos": state[14:16],
    }


def _resize_to_256(img: np.ndarray) -> np.ndarray:
    """Resize an (H, W, 3) uint8 image to (256, 256, 3), matching the resolution
    declared in the N1.5 robocasa checkpoint's metadata.json. The openpi client
    defaults to resizing to 224x224 (pi05's expected input), so we upsize here
    rather than changing the client and breaking pi05.
    """
    import cv2

    if img.shape[0] == 256 and img.shape[1] == 256:
        return img
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)


def build_robocasa_videos(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Map openpi client camera keys to GR00T N1.5 video keys.

    The N1.5 robocasa head was trained with THREE distinct cameras (two side
    views + wrist). The openpi robocasa client emits three keys:
        observation/image       -> agentview_left  -> video.robot0_agentview_left
        observation/image2      -> agentview_right -> video.robot0_agentview_right
        observation/wrist_image -> eye_in_hand     -> video.robot0_eye_in_hand
    `observation/image2` is a backward-compatible addition — pi0's RobocasaInputs
    doesn't read it, so adding this key to the client doesn't affect pi05
    serving. If a client omits it (older builds), we fall back to duplicating
    `observation/image` into the right-view slot so the model still has a
    three-channel input; the stereo signal is degraded but correctness is
    preserved. Images are resized to 256x256 to match the N1.5 checkpoint's
    declared metadata resolution.
    """
    img = _resize_to_256(np.asarray(obs["observation/image"], dtype=np.uint8))
    wrist = _resize_to_256(np.asarray(obs["observation/wrist_image"], dtype=np.uint8))
    if "observation/image2" in obs:
        img2 = _resize_to_256(np.asarray(obs["observation/image2"], dtype=np.uint8))
    else:
        # Fallback: no right-view available -> reuse left-view. Degrades
        # accuracy but keeps the shape contract intact.
        img2 = img
    return {
        "video.robot0_agentview_left": img,
        "video.robot0_agentview_right": img2,
        "video.robot0_eye_in_hand": wrist,
    }


# gr00t is always installed in this venv (the whole package exists to serve
# N1.5), so importing `BaseDataConfig` at module load is fine. The subclass
# below catches any upstream API drift (renamed attribute, new required
# method, etc.) as a clear error at construction time.
from gr00t.experiment.data_config import BaseDataConfig


class RobocasaPandaOmronDataConfig(BaseDataConfig):
    """N1.5 DataConfig for robocasa Panda Omron. Derived from
    `gr00t.experiment.data_config.SinglePandaGripperDataConfig` but with the
    video keys aligned to the openpi robocasa env's wrapper outputs
    (`robot0_agentview_left/right`, `robot0_eye_in_hand`) and the
    state/action key ordering verified against the checkpoint's
    `experiment_cfg/metadata.json`.
    """

    video_keys = list(ROBOCASA_VIDEO_KEYS)
    state_keys = list(ROBOCASA_STATE_KEYS)
    action_keys = list(ROBOCASA_ACTION_KEYS)
    language_keys = [ROBOCASA_LANGUAGE_KEY]
    observation_indices = [0]
    action_indices = list(range(16))

    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "min_max",
        "action.end_effector_rotation": "min_max",
        "action.gripper_close": "binary",
        "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }

    # `BaseDataConfig.modality_config` is already defined exactly as we need
    # it (builds ModalityConfig objects from the six class attributes above),
    # so we inherit it unchanged.

    def transform(self):
        return _build_robocasa_transform(self)


def _build_robocasa_transform(self):
    """Module-level helper because the transform body is long; the inner
    subclass above just delegates here."""
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.data.transform.concat import ConcatTransform
    from gr00t.data.transform.state_action import (
        StateActionToTensor,
        StateActionTransform,
    )
    from gr00t.data.transform.video import (
        VideoColorJitter,
        VideoCrop,
        VideoResize,
        VideoToNumpy,
        VideoToTensor,
    )
    from gr00t.model.transforms import GR00TTransform

    transforms = [
        VideoToTensor(apply_to=self.video_keys),
        VideoCrop(apply_to=self.video_keys, scale=0.95),
        VideoResize(
            apply_to=self.video_keys, height=224, width=224, interpolation="linear"
        ),
        VideoColorJitter(
            apply_to=self.video_keys,
            brightness=0.3,
            contrast=0.4,
            saturation=0.5,
            hue=0.08,
        ),
        VideoToNumpy(apply_to=self.video_keys),
        StateActionToTensor(apply_to=self.state_keys),
        StateActionTransform(
            apply_to=self.state_keys,
            normalization_modes=self.state_normalization_modes,
            target_rotations=self.state_target_rotations,
        ),
        StateActionToTensor(apply_to=self.action_keys),
        StateActionTransform(
            apply_to=self.action_keys,
            normalization_modes=self.action_normalization_modes,
        ),
        ConcatTransform(
            video_concat_order=self.video_keys,
            state_concat_order=self.state_keys,
            action_concat_order=self.action_keys,
        ),
        GR00TTransform(
            state_horizon=len(self.observation_indices),
            action_horizon=len(self.action_indices),
            max_state_dim=64,
            max_action_dim=32,
        ),
    ]
    return ComposedModalityTransform(transforms=transforms)


class GR00TAdapterPolicy(_base_policy.BasePolicy):
    """Wrap a `gr00t.model.policy.Gr00tPolicy` (N1.5) as an openpi `BasePolicy`.

    Translates the openpi robocasa client's flat observation dict to GR00T
    N1.5's flat observation dict, runs inference, and concatenates the
    per-action-key outputs into a single (T, action_dim) array under the
    "actions" key so the client can use `result["actions"]` unchanged.
    """

    def __init__(
        self,
        gr00t_policy,
        *,
        video_builder: Callable[[dict[str, Any]], dict[str, np.ndarray]],
        state_builder: Callable[[dict[str, Any]], dict[str, np.ndarray]],
        action_keys: list[str],
        language_key: str = "prompt",
        groot_language_key: str = ROBOCASA_LANGUAGE_KEY,
    ) -> None:
        self._policy = gr00t_policy
        self._video_builder = video_builder
        self._state_builder = state_builder
        self._action_keys = list(action_keys)
        self._language_key = language_key
        self._groot_language_key = groot_language_key

    def _squeeze_leading_batch(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Strip a leading size-1 batch dim if the collector already added one.

        `activation_collector.CollectingPolicy._batch_single_example` adds a
        leading batch dim to every ndarray and wraps `prompt` in a list. This
        was written for pi0's batched `infer_with_intermediates`; our adapter
        batches internally and expects single-example inputs. Detect + unwrap.
        """
        state = obs.get("observation/state")
        if state is None:
            return obs
        state_arr = np.asarray(state)
        # Single-example state is 1-D (shape (16,)); batched is 2-D ((1, 16)).
        if state_arr.ndim < 2:
            return obs
        unwrapped: dict[str, Any] = {}
        for key, value in obs.items():
            if key == "prompt":
                # _batch_single_example wrapped a str in a list; take first.
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    unwrapped[key] = value[0]
                else:
                    unwrapped[key] = value
            elif (
                isinstance(value, np.ndarray)
                and value.ndim >= 1
                and value.shape[0] == 1
            ):
                unwrapped[key] = value[0]
            else:
                unwrapped[key] = value
        return unwrapped

    def _build_groot_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Translate the openpi client's flat obs dict into GR00T N1.5's flat
        {video.X, state.Y, annotation.Z} format."""
        obs = self._squeeze_leading_batch(obs)
        groot_obs: dict[str, Any] = {}
        # Video: each (H, W, 3) uint8 -> (1, H, W, 3) uint8 (T=1).
        for k, v in self._video_builder(obs).items():
            arr = v if v.dtype == np.uint8 else v.astype(np.uint8)
            groot_obs[k] = arr[None]  # add T dim
        # State: each (D,) float32 -> (1, D) float32 (T=1).
        for k, v in self._state_builder(obs).items():
            arr = np.asarray(v, dtype=np.float32)
            groot_obs[k] = arr[None] if arr.ndim == 1 else arr
        # Language: a single string -> list[str] of length 1 (T=1).
        prompt = obs.get(self._language_key, "")
        if isinstance(prompt, (bytes, np.bytes_)):
            prompt = prompt.decode("utf-8")
        groot_obs[self._groot_language_key] = [str(prompt)]
        return groot_obs

    def _action_dict_to_array(self, action_dict: dict[str, Any]) -> np.ndarray:
        pieces = [
            np.asarray(action_dict[k], dtype=np.float32) for k in self._action_keys
        ]
        return np.concatenate(pieces, axis=-1)  # (T, D_total)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        groot_obs = self._build_groot_obs(obs)
        action_dict = self._policy.get_action(groot_obs)
        return {"actions": self._action_dict_to_array(action_dict)}

    def infer_with_intermediates(
        self, obs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Run inference and return the per-step activations used for mech-interp.

        Intermediates shape convention (matches pi0's `sample_actions_with_intermediates`
        schema one-for-one where the architecture allows):
          - all_x_t: (num_denoising_steps, B, action_horizon, action_dim), fp32
                     Denoising input trajectory. Entry k is x_t at the START of
                     step k (k=0 is the sampled noise). Matches pi0's shape and
                     semantic — the post-final-step action is NOT included here,
                     it's returned as part of `actions`.
          - all_v_t: (num_denoising_steps, B, action_horizon, action_dim), fp32
                     Predicted velocity at each denoising step.
          - backbone_features: (B, seq_len, hidden_dim), fp16
                               VL backbone output the DiT cross-attends to. This
                               is GR00T's analog of pi0's `adarms_cond`, but note
                               the architectures differ: pi0 has a POOLED
                               per-step conditioning vector (hence shape
                               (num_steps, hidden)), while GR00T computes a
                               variable-length VL sequence ONCE and feeds it to
                               the DiT via cross-attention (hence (seq, hidden)).
          - all_dit_hidden_states: (num_denoising_steps, num_dit_layers, B, seq_len_sa, hidden_dim), fp16
                                   DiT residual stream per layer per denoising
                                   step. Captured via forward hooks on each
                                   `BasicTransformerBlock` output (matches pi0's
                                   `suffix_residual` capture pattern on each
                                   `expert_layers[i]`). Only layer OUTPUTS are
                                   captured, not the DiT input.
          - all_dit_mlp_hidden: (num_denoising_steps, num_dit_layers, B, seq_len_sa, ff_inner_dim), fp16
                                Per-layer MLP expanded activation. Captured via
                                input hooks on each DiT block's `ff.net[2]` (the
                                `inner_dim -> dim` contraction Linear); matches
                                pi0's `suffix_mlp_hidden` which hooks inputs of
                                `expert_layers[i].mlp.down_proj`.
        Batch dim B is always 1 here (openpi server sends one obs at a time).
        """
        groot_obs = self._build_groot_obs(obs)
        action_dict, intermediates = _get_action_with_intermediates(
            self._policy, groot_obs
        )
        return {"actions": self._action_dict_to_array(action_dict)}, intermediates

    def infer_with_steering(
        self, obs: dict[str, Any], *, steering_hooks: list[tuple[int, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run inference while applying hooks to selected GR00T DiT layers.

        ``steering_hooks`` is a list of ``(layer_idx, hook)`` pairs. The hook is
        registered on ``head.model.transformer_blocks[layer_idx]`` and receives
        the block output tensor ``(B, seq_len_sa, hidden_dim)``. If the hook has
        ``set_denoise_step(t)``, it is called before every DiT forward so
        per-step conceptors can select the active matrix.
        """
        groot_obs = self._build_groot_obs(obs)
        action_dict, steering_metadata = _get_action_with_steering(
            self._policy, groot_obs, steering_hooks=steering_hooks
        )
        return {
            "actions": self._action_dict_to_array(action_dict),
            "steering_metadata": steering_metadata,
        }, steering_metadata

    def reset(self) -> None:
        # N1.5 Gr00tPolicy is stateless between calls; nothing to reset.
        pass


def _get_action_with_intermediates(gr00t_policy, groot_obs):
    """Run Gr00tPolicy inference while capturing denoising intermediates.

    Structured to match pi0's `sample_actions_with_intermediates` pattern
    (`src/openpi/models_pytorch/pi0_pytorch.py`), which is a HYBRID:
      - The denoising loop body is re-implemented inline so we can capture
        loop-level values (x_t, v_t, the backbone conditioning) directly at the
        point they're computed.
      - Per-layer internal activations (residual stream, MLP hidden) are
        captured via PyTorch forward hooks on the DiT transformer blocks, then
        picked up after each step. This is how pi0 captures
        `expert_residual_{i}` (output hook on each expert layer) and
        `expert_mlp_hidden_{i}` (input hook on each expert layer's
        `mlp.down_proj`). We do the exact same thing on the DiT side: output
        hook on each `transformer_blocks[i]` for the residual, input hook on
        each `transformer_blocks[i].ff.net[2]` (the `inner_dim -> dim`
        contraction Linear) for the pre-contraction MLP activation.

    Schema lines up with pi0 one-for-one where architecturally possible
    (see `GR00TAdapterPolicy.infer_with_intermediates` for the full spec).

    Parameters
    ----------
    gr00t_policy: gr00t.model.policy.Gr00tPolicy
        The N1.5 policy wrapping a `GR00T_N1_5` model.
    groot_obs: dict
        Observation in GR00T's flat format (video.X, state.X, annotation.X).

    Returns
    -------
    (action_dict, intermediates)
        action_dict: {"action.X": np.ndarray (T, D), ...} — exactly what
                     `Gr00tPolicy.get_action` would return.
        intermediates: dict of numpy arrays: all_x_t, all_v_t,
                       backbone_features, all_dit_hidden_states,
                       all_dit_mlp_hidden.
    """
    import torch

    obs_copy = {}
    for k, v in groot_obs.items():
        if not isinstance(v, np.ndarray) and not isinstance(v, list):
            obs_copy[k] = np.array(v)
        else:
            obs_copy[k] = v

    # Detect batch dim, add if missing. _check_state_is_batched treats any key
    # containing "state" with ndim < 3 as unbatched.
    from gr00t.model.policy import squeeze_dict_values, unsqueeze_dict_values

    is_batch = gr00t_policy._check_state_is_batched(obs_copy)
    if not is_batch:
        obs_copy = unsqueeze_dict_values(obs_copy)

    # Apply normalization/video transforms exactly as Gr00tPolicy.get_action does.
    normalized_input = gr00t_policy.apply_transforms(obs_copy)

    model = gr00t_policy.model
    head = model.action_head

    # --- Forward hooks for per-layer intermediates (pi0 pattern) -----------
    # step_activations is cleared at the start of every denoising step; the
    # per-layer captures for that step are collected into the aggregate lists
    # right after the DiT forward returns.
    step_activations: dict[str, torch.Tensor] = {}
    num_layers = len(head.model.transformer_blocks)

    def _make_output_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                step_activations[name] = output[0].detach()
            else:
                step_activations[name] = output.detach()

        return hook

    def _make_input_hook(name):
        def hook(module, input, output):
            # input is the positional args tuple; first element is the tensor
            # about to be contracted from ff_inner_dim back to hidden_dim.
            step_activations[name] = input[0].detach()

        return hook

    hooks = []
    for i, block in enumerate(head.model.transformer_blocks):
        # Residual stream after block i (analog of pi0's expert_residual_{i}).
        hooks.append(block.register_forward_hook(_make_output_hook(f"residual_{i}")))
        # MLP inner activation (pre-contraction). `ff.net[2]` is the
        # (ff_inner_dim -> hidden_dim) Linear, structurally analogous to pi0's
        # `mlp.down_proj`. Hooking its INPUT gives us the post-GELU expanded
        # activation, matching pi0's `expert_mlp_hidden_{i}` semantic.
        hooks.append(
            block.ff.net[2].register_forward_hook(_make_input_hook(f"mlp_hidden_{i}"))
        )

    # Replicate GR00T_N1_5.get_action's input prep.
    backbone_inputs, action_inputs = model.prepare_input(normalized_input)
    autocast_device_type = torch.device(gr00t_policy.device).type

    try:
        # autocast only on CUDA / MPS; on CPU we skip the context (bfloat16 on
        # CPU is supported in recent torch but N1.5's 3B model isn't viable on
        # CPU anyway; this keeps the code path honest).
        if autocast_device_type in ("cuda", "mps"):
            ac_ctx = torch.autocast(
                device_type=autocast_device_type, dtype=torch.bfloat16
            )
        else:
            ac_ctx = contextlib.nullcontext()
        with torch.inference_mode(), ac_ctx:
            backbone_outputs = model.backbone(backbone_inputs)
            processed_backbone = head.process_backbone_output(backbone_outputs)
            vl_embs = processed_backbone.backbone_features  # (B, S, C)
            embodiment_id = action_inputs.embodiment_id
            state_features = head.state_encoder(action_inputs.state, embodiment_id)

            batch_size = vl_embs.shape[0]
            device = vl_embs.device
            actions = torch.randn(
                size=(
                    batch_size,
                    head.config.action_horizon,
                    head.config.action_dim,
                ),
                dtype=vl_embs.dtype,
                device=device,
            )

            num_steps = head.num_inference_timesteps
            dt = 1.0 / num_steps

            # Inline per-step aggregates (matches pi0's list-accumulator style).
            all_x_t_list = []
            all_v_t_list = []
            all_dit_residual_per_step = []  # each entry: (num_layers, B, S, D) tensor
            all_dit_mlp_hidden_per_step = []  # each entry: (num_layers, B, S, ff_inner) tensor

            for t in range(num_steps):
                step_activations.clear()

                # x_t at the START of this step (pi0: captured BEFORE the model
                # forward, at the same point in the loop).
                all_x_t_list.append(actions.detach().float().cpu())

                t_cont = t / float(num_steps)
                t_discretized = int(t_cont * head.num_timestep_buckets)
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = head.action_encoder(
                    actions, timesteps_tensor, embodiment_id
                )
                if head.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1], dtype=torch.long, device=device
                    )
                    pos_embs = head.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs
                future_tokens = head.future_tokens.weight.unsqueeze(0).expand(
                    vl_embs.shape[0], -1, -1
                )
                sa_embs = torch.cat(
                    (state_features, future_tokens, action_features), dim=1
                )

                # Plain DiT forward — hooks fire during this call. We no
                # longer ask for `return_all_hidden_states=True` because the
                # output hook on each transformer_block gives us the same
                # per-layer residual stream.
                model_output = head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )

                pred = head.action_decoder(model_output, embodiment_id)
                pred_velocity = pred[:, -head.action_horizon :]
                all_v_t_list.append(pred_velocity.detach().float().cpu())

                # Gather hook captures for this step. Stacking along layer
                # axis gives (num_layers, B, S, D) / (num_layers, B, S, ff_inner).
                residuals = torch.stack(
                    [step_activations[f"residual_{i}"] for i in range(num_layers)],
                    dim=0,
                )
                mlp_hidden = torch.stack(
                    [step_activations[f"mlp_hidden_{i}"] for i in range(num_layers)],
                    dim=0,
                )
                all_dit_residual_per_step.append(residuals)
                all_dit_mlp_hidden_per_step.append(mlp_hidden)

                actions = actions + dt * pred_velocity

            # Shapes match pi0:
            #   all_x_t: (num_steps, B, H, D)            (NOT num_steps+1)
            #   all_v_t: (num_steps, B, H, D)
            all_x_t = torch.stack(all_x_t_list, dim=0).numpy()
            all_v_t = torch.stack(all_v_t_list, dim=0).numpy()
            # all_dit_hidden_states: (num_steps, num_layers, B, S, D)   — no leading input
            dit_hidden_stacked = torch.stack(all_dit_residual_per_step, dim=0)
            all_dit_hidden_states = dit_hidden_stacked.to(torch.float16).cpu().numpy()
            # all_dit_mlp_hidden: (num_steps, num_layers, B, S, ff_inner)
            dit_mlp_stacked = torch.stack(all_dit_mlp_hidden_per_step, dim=0)
            all_dit_mlp_hidden = dit_mlp_stacked.to(torch.float16).cpu().numpy()
            backbone_features = vl_embs.to(torch.float16).cpu().numpy()

            final_action_tensor = actions.float().cpu()
    finally:
        for h in hooks:
            h.remove()

    # Go through the same unnormalize path as Gr00tPolicy so the action dict is
    # in physical units. Replicates _get_unnormalized_action.
    unnormalized_action = gr00t_policy.unapply_transforms(
        {"action": final_action_tensor}
    )

    if not is_batch:
        unnormalized_action = squeeze_dict_values(unnormalized_action)

    intermediates = {
        "all_x_t": all_x_t.astype(np.float32),
        "all_v_t": all_v_t.astype(np.float32),
        "backbone_features": backbone_features,
        "all_dit_hidden_states": all_dit_hidden_states,
        "all_dit_mlp_hidden": all_dit_mlp_hidden,
    }
    return unnormalized_action, intermediates


def _get_action_with_steering(gr00t_policy, groot_obs, *, steering_hooks):
    """Run Gr00tPolicy inference while applying DiT residual steering hooks."""
    import torch

    obs_copy = {}
    for k, v in groot_obs.items():
        if not isinstance(v, np.ndarray) and not isinstance(v, list):
            obs_copy[k] = np.array(v)
        else:
            obs_copy[k] = v

    from gr00t.model.policy import squeeze_dict_values, unsqueeze_dict_values

    is_batch = gr00t_policy._check_state_is_batched(obs_copy)
    if not is_batch:
        obs_copy = unsqueeze_dict_values(obs_copy)

    normalized_input = gr00t_policy.apply_transforms(obs_copy)

    model = gr00t_policy.model
    head = model.action_head
    num_layers = len(head.model.transformer_blocks)

    hooks = []
    for layer_idx, hook in steering_hooks:
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(
                f"GR00T steering layer {layer_idx} out of range "
                f"(num_layers={num_layers})"
            )
        hooks.append(
            head.model.transformer_blocks[layer_idx].register_forward_hook(hook)
        )

    backbone_inputs, action_inputs = model.prepare_input(normalized_input)
    autocast_device_type = torch.device(gr00t_policy.device).type

    try:
        if autocast_device_type in ("cuda", "mps"):
            ac_ctx = torch.autocast(
                device_type=autocast_device_type, dtype=torch.bfloat16
            )
        else:
            ac_ctx = contextlib.nullcontext()
        with torch.inference_mode(), ac_ctx:
            backbone_outputs = model.backbone(backbone_inputs)
            processed_backbone = head.process_backbone_output(backbone_outputs)
            vl_embs = processed_backbone.backbone_features
            embodiment_id = action_inputs.embodiment_id
            state_features = head.state_encoder(action_inputs.state, embodiment_id)

            batch_size = vl_embs.shape[0]
            device = vl_embs.device
            actions = torch.randn(
                size=(
                    batch_size,
                    head.config.action_horizon,
                    head.config.action_dim,
                ),
                dtype=vl_embs.dtype,
                device=device,
            )

            num_steps = head.num_inference_timesteps
            dt = 1.0 / num_steps

            for t in range(num_steps):
                for _, hook in steering_hooks:
                    if hasattr(hook, "set_denoise_step"):
                        hook.set_denoise_step(t)

                t_cont = t / float(num_steps)
                t_discretized = int(t_cont * head.num_timestep_buckets)
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = head.action_encoder(
                    actions, timesteps_tensor, embodiment_id
                )
                if head.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1], dtype=torch.long, device=device
                    )
                    pos_embs = head.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs
                future_tokens = head.future_tokens.weight.unsqueeze(0).expand(
                    vl_embs.shape[0], -1, -1
                )
                sa_embs = torch.cat(
                    (state_features, future_tokens, action_features), dim=1
                )

                model_output = head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )

                pred = head.action_decoder(model_output, embodiment_id)
                pred_velocity = pred[:, -head.action_horizon :]
                actions = actions + dt * pred_velocity

            final_action_tensor = actions.float().cpu()
    finally:
        for h in hooks:
            h.remove()

    unnormalized_action = gr00t_policy.unapply_transforms(
        {"action": final_action_tensor}
    )

    if not is_batch:
        unnormalized_action = squeeze_dict_values(unnormalized_action)

    steering_metadata = {
        "steering_hooks": [
            {
                "layer": int(layer_idx),
                "hook": repr(hook),
                "intervention_norms": [
                    float(v) for v in getattr(hook, "intervention_norms", [])
                ],
            }
            for layer_idx, hook in steering_hooks
        ]
    }
    return unnormalized_action, steering_metadata


def make_robocasa_policy(
    model_path: str, *, device: str = "cuda:0", denoising_steps: int = 4
):
    """Construct a Gr00tPolicy + GR00TAdapterPolicy for N1.5 robocasa.

    `model_path` is a HuggingFace id (`nvidia/...`) or a local checkpoint dir
    that contains `config.json`, `model-*.safetensors`, and
    `experiment_cfg/metadata.json` (with embodiment `new_embodiment`).
    """
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.policy import Gr00tPolicy

    data_config = RobocasaPandaOmronDataConfig()
    gr00t_policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=denoising_steps,
        device=device,
    )

    return GR00TAdapterPolicy(
        gr00t_policy,
        video_builder=build_robocasa_videos,
        state_builder=build_robocasa_state_dict,
        action_keys=list(ROBOCASA_ACTION_KEYS),
    )
