"""GR00T N1.5 activation-collection wrapper for the WebSocket policy server.

Independent of the pi0-side ``src/openpi/serving/activation_collector.py`` —
lives here because GR00T has its own venv (torch 2.5.1, conflicts with the root
COAST env). The on-disk schema (``groot_v1``) is distinct from pi0's ``v1``
(denoising is the same file, but conditioning + per-layer tensors have
GR00T-specific names; see ``save_step_activations`` below).

The ``__collect__`` / ``__finalize_episode__`` wire protocol, task-name
sanitization, per-episode ``metadata.json`` + ``rewards.npz`` layout, and
``_intermediates_lock`` concurrency guard all mirror the pi0-side
``CollectingPolicy`` so the same robocasa client can `--collect` against either
backend without changes.
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)


def save_step_activations(
    step_dir: pathlib.Path,
    intermediates: dict,
    env_id: int,
    step_metadata: dict,
) -> None:
    """Save per-env, per-step activation data (GR00T N1.5 schema, v1 format).

    Slices the env_id-th example out of the batch dimension of each intermediate
    array, then writes one .npz per activation kind plus metadata.json. The
    schema mirrors pi0's naming convention where it makes sense so downstream
    mech-interp tooling can reuse the same file layout:
      - denoising.npz {all_x_t, all_v_t}                - per denoising-step x_t and velocity
      - backbone_cond.npz {backbone_features}           - VL backbone output the DiT
                                                          cross-attends to (GR00T analog
                                                          of pi0's adarms_cond; shape
                                                          differs because of the
                                                          architectural difference —
                                                          see adapter docstring).
      - dit_hidden_states.npz {all_dit_hidden_states}   - DiT per-layer residual stream,
                                                          pi0 analog: suffix_residual.
      - dit_mlp_hidden.npz {all_dit_mlp_hidden}         - DiT per-layer MLP expanded
                                                          activation (input to the
                                                          ff_inner->hidden contraction);
                                                          pi0 analog: suffix_mlp_hidden.
    """
    step_dir.mkdir(parents=True, exist_ok=True)

    # GR00T intermediates shapes (produced by GR00TAdapterPolicy.infer_with_intermediates):
    #   all_x_t: (num_denoising_steps, batch, action_horizon, action_dim), fp32
    #   all_v_t: (num_denoising_steps, batch, action_horizon, action_dim), fp32
    #   backbone_features: (batch, seq_len, hidden_dim), fp16
    #   all_dit_hidden_states: (num_denoising_steps, num_dit_layers, batch, seq_len, hidden_dim), fp16
    #   all_dit_mlp_hidden: (num_denoising_steps, num_dit_layers, batch, seq_len, ff_inner_dim), fp16
    all_x_t = intermediates["all_x_t"][:, env_id]
    all_v_t = intermediates["all_v_t"][:, env_id]
    backbone_features = intermediates["backbone_features"][env_id]
    all_dit_hidden_states = intermediates["all_dit_hidden_states"][:, :, env_id]
    all_dit_mlp_hidden = intermediates["all_dit_mlp_hidden"][:, :, env_id]

    np.savez(step_dir / "denoising.npz", all_x_t=all_x_t, all_v_t=all_v_t)
    np.savez(step_dir / "backbone_cond.npz", backbone_features=backbone_features)
    np.savez(
        step_dir / "dit_hidden_states.npz", all_dit_hidden_states=all_dit_hidden_states
    )
    np.savez(step_dir / "dit_mlp_hidden.npz", all_dit_mlp_hidden=all_dit_mlp_hidden)

    with open(step_dir / "metadata.json", "w") as f:
        json.dump(step_metadata, f, indent=2)


def save_episode_files(
    episode_dir: pathlib.Path,
    episode_metadata: dict,
    per_step_reward: list[float],
    per_step_success: list[bool],
) -> None:
    """Write episode-level metadata.json and rewards.npz."""
    episode_dir.mkdir(parents=True, exist_ok=True)

    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(episode_metadata, f, indent=2)

    rewards_arr = np.array(per_step_reward, dtype=np.float32)
    cumulative_arr = np.cumsum(rewards_arr).astype(np.float32)
    success_arr = np.array(per_step_success, dtype=bool)
    np.savez(
        episode_dir / "rewards.npz",
        per_step_reward=rewards_arr,
        cumulative_reward=cumulative_arr,
        success_at_step=success_arr,
    )


_COLLECT_KEY = "__collect__"
_FINALIZE_KEY = "__finalize_episode__"


class CollectingPolicy(_base_policy.BasePolicy):
    """Wraps a Policy and saves intermediates to disk on demand.

    This is a *collection-only* wrapper: every infer() call must include either
    __collect__ (per-step inference + activation save) or __finalize_episode__
    (per-episode metadata + rewards write, no inference). Plain inference
    requests are rejected so a collection server cannot accidentally serve
    eval traffic.

    The wrapper holds no per-episode state. The client is responsible for
    tracking cumulative reward, per-step rewards, and the inference_step
    counter, and sending them in the magic-key payloads.
    """

    def __init__(
        self,
        policy: Any,
        output_root: pathlib.Path,
        checkpoint_step: str,
        policy_dir: str,
        config_name: str,
    ) -> None:
        self._policy = policy
        self._output_root = pathlib.Path(output_root)
        self._checkpoint_step = checkpoint_step
        self._policy_dir = policy_dir
        self._config_name = config_name
        # Serializes calls into the model's hook-based intermediate collection
        # path. See _handle_collect_infer for the rationale.
        self._intermediates_lock = threading.Lock()

    @property
    def metadata(self) -> dict:
        underlying = getattr(self._policy, "metadata", {}) or {}
        return {
            **underlying,
            "policy_dir": str(self._policy_dir),
            "config_name": self._config_name,
            # collection_mode is the on-disk schema identifier. "groot_v1" is
            # GR00T's DiT-plus-backbone layout — distinct from pi0's "v1"
            # (suffix_residual/adarms_cond) and pi0-fast's "fast_v1" (tokens/
            # hidden_states/token_logprobs). Downstream tools should key off
            # this field, not the backend, to pick the right loader.
            "collection_mode": "groot_v1",
            # Parallels the pi0-side model_type field ("pi0" / "pi05" / "pi0_fast").
            # Not the same as `backend` above — backend is a free-form label;
            # model_type is the uniform identifier shared across all collection
            # servers in this repo.
            "model_type": "groot_n15",
            "checkpoint_step": self._checkpoint_step,
            "output_root": str(self._output_root),
        }

    def infer(self, obs: dict) -> dict:
        finalize_meta = obs.get(_FINALIZE_KEY)
        collect_meta = obs.get(_COLLECT_KEY)

        if finalize_meta is not None and collect_meta is not None:
            raise ValueError(
                f"Request contains both {_COLLECT_KEY} and {_FINALIZE_KEY}; only one is allowed per call."
            )

        if finalize_meta is not None:
            return self._handle_finalize(finalize_meta)

        if collect_meta is None:
            raise ValueError(
                f"Collection-only server requires either {_COLLECT_KEY} or {_FINALIZE_KEY} to be set on every request."
            )

        return self._handle_collect_infer(obs, collect_meta)

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    # ---------------------------------------------------------------- helpers

    def _sanitize_task_name(self, task_name: Any) -> str:
        task_name_str = str(task_name)
        task_path = pathlib.PurePosixPath(task_name_str)
        if task_path.is_absolute():
            raise ValueError(
                f"Invalid task_name {task_name_str!r}: absolute paths are not allowed."
            )
        if any(part in {"", ".", ".."} for part in task_path.parts):
            raise ValueError(
                f"Invalid task_name {task_name_str!r}: path traversal segments are not allowed."
            )
        if len(task_path.parts) != 1:
            raise ValueError(
                f"Invalid task_name {task_name_str!r}: nested paths are not allowed."
            )
        if "\\" in task_name_str:
            raise ValueError(
                f"Invalid task_name {task_name_str!r}: path separators are not allowed."
            )
        return task_name_str

    def _episode_dir(self, meta: dict) -> pathlib.Path:
        task_name = self._sanitize_task_name(meta["task_name"])
        return (
            self._output_root
            / self._checkpoint_step
            / task_name
            / "episode_{:03d}_env_{:03d}".format(
                int(meta["episode_id"]), int(meta["env_id"])
            )
        )

    def _step_dir(self, meta: dict) -> pathlib.Path:
        return self._episode_dir(meta) / "step_{:04d}".format(int(meta["step"]))

    def _handle_collect_infer(self, obs: dict, collect_meta: dict) -> dict:
        # Drop the magic keys before passing the obs to the underlying policy.
        # We mutate a shallow copy so the caller's dict is untouched.
        clean_obs = {
            k: v for k, v in obs.items() if k not in (_COLLECT_KEY, _FINALIZE_KEY)
        }

        batched_obs = self._batch_single_example(clean_obs)

        # Collection mode is single-example only: the __collect__ payload carries
        # one env_id, so we cannot label per-element activations from a multi-env
        # batch. Reject batched obs loudly here rather than silently slicing
        # batch index 0 and writing it under the metadata's env_id (which would
        # corrupt the on-disk dataset). Future support for batched collection
        # would require extending the protocol so __collect__ carries a list
        # of per-element metadata dicts.
        # Find a proprioceptive observation key to verify single-example batch.
        probe_key = "observation/state"
        if probe_key not in batched_obs:
            for key in batched_obs:
                if key.startswith("observation/") and "image" not in key:
                    probe_key = key
                    break
        probe_arr = np.asarray(batched_obs[probe_key])
        if probe_arr.ndim != 2 or probe_arr.shape[0] != 1:
            raise ValueError(
                f"Collection mode only supports single-example inputs "
                f"({probe_key} shape (1, N)), got shape "
                f"{tuple(probe_arr.shape)}. Send one inference call per env."
            )

        # Serialize calls into infer_with_intermediates: the underlying
        # `_get_action_with_intermediates` (GR00T analog of pi0's V1
        # `sample_actions_with_intermediates`) registers forward hooks on
        # shared module instances, so two in-flight calls would pollute each
        # other's capture dicts. The current single-threaded asyncio server
        # already serializes calls implicitly, but this lock makes the
        # invariant explicit and defends against future executor-based
        # optimizations.
        with self._intermediates_lock:
            result, intermediates = self._policy.infer_with_intermediates(batched_obs)

        step_dir = self._step_dir(collect_meta)
        save_step_activations(
            step_dir=step_dir,
            intermediates=intermediates,
            env_id=0,  # batch_size is enforced to 1 above
            step_metadata=dict(collect_meta),
        )

        # Strip the batch dim from actions so the client receives the same
        # shape it would from a non-collection server (action_horizon, action_dim).
        actions = np.asarray(result["actions"])
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]

        response: dict = {"actions": actions}
        if "policy_timing" in result:
            response["policy_timing"] = result["policy_timing"]
        return response

    def _handle_finalize(self, finalize_meta: dict) -> dict:
        episode_dir = self._episode_dir(finalize_meta)

        per_step_reward = list(finalize_meta.get("per_step_reward", []))
        per_step_success = list(finalize_meta.get("per_step_success", []))

        episode_metadata = {
            "task_name": finalize_meta["task_name"],
            "episode_id": int(finalize_meta["episode_id"]),
            "env_id": int(finalize_meta["env_id"]),
            "episode_success": bool(finalize_meta.get("episode_success", False)),
            "total_reward": float(finalize_meta.get("total_reward", 0.0)),
            "steps_to_success": int(finalize_meta.get("steps_to_success", -1)),
            "total_env_steps": int(
                finalize_meta.get("total_env_steps", len(per_step_reward))
            ),
            "total_inference_steps": int(finalize_meta.get("total_inference_steps", 0)),
            "prompt": str(finalize_meta.get("prompt", "")),
            "checkpoint_dir": str(self._policy_dir),
            "config_name": self._config_name,
        }

        save_episode_files(
            episode_dir=episode_dir,
            episode_metadata=episode_metadata,
            per_step_reward=per_step_reward,
            per_step_success=per_step_success,
        )
        logger.info(
            "Finalized episode %s/%s episode_%03d (success=%s, env_steps=%d) -> %s",
            finalize_meta["task_name"],
            self._checkpoint_step,
            int(finalize_meta["episode_id"]),
            episode_metadata["episode_success"],
            episode_metadata["total_env_steps"],
            episode_dir,
        )
        return {"ack": True, "episode_dir": str(episode_dir)}

    @staticmethod
    def _batch_single_example(obs: dict) -> dict:
        """Add a leading batch dim to a single-example obs dict.

        Policy.infer_with_intermediates expects observation/state of shape
        (batch, state_dim). The libero client sends a 1-D state; metaworld
        sends pre-batched (num_envs, state_dim). Pass through unchanged when
        already batched.
        """
        # Find any observation array to check if already batched. Prefer
        # "observation/state" (metaworld/libero/robocasa) but fall back to the
        # first non-image observation key (droid sends observation/joint_position
        # + observation/gripper_position instead of observation/state).
        probe_key = None
        if "observation/state" in obs:
            probe_key = "observation/state"
        else:
            for key in obs:
                if key.startswith("observation/") and "image" not in key:
                    probe_key = key
                    break
        if probe_key is None:
            return obs
        probe_arr = np.asarray(obs[probe_key])
        if probe_arr.ndim >= 2:
            return obs

        batched: dict = {}
        for key, value in obs.items():
            if key == "prompt":
                batched[key] = [value] if isinstance(value, str) else value
            elif isinstance(value, np.ndarray):
                batched[key] = value[np.newaxis, ...]
            else:
                arr = np.asarray(value)
                batched[key] = arr[np.newaxis, ...]
        return batched
