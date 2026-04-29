"""Per-episode bookkeeping helper for the openpi activation-collection protocol.

This is an env-agnostic client-side helper. Drop it into any rollout loop that
talks to an openpi policy server started with `--collect_activations`. It
tracks per-episode state (cumulative reward, per-step rewards/successes,
inference_step counter) and shapes the metadata payloads that get attached to
each WebSocket request as `__collect__` / `__finalize_episode__` magic keys.
The server's CollectingPolicy (in openpi.serving.activation_collector) pops
those keys, runs `infer_with_intermediates`, and writes activations to its
own filesystem.

This helper has zero env-specific imports — it only depends on the duck-typed
`policy.infer(dict) -> dict` interface (i.e. WebsocketClientPolicy or anything
compatible). The libero, robocasa, and droid example clients can all use it
unchanged; future real-robot clients can too.

Usage from inside an eval loop:

    from openpi_client.collection_session import CollectionSession

    session = CollectionSession(client)
    session.start_episode(task_name, task_id, episode_id, prompt)
    for step in range(...):
        if not action_plan:
            element["__collect__"] = session.make_collect_metadata(step)
            action_chunk = client.infer(element)["actions"]
            ...
        obs, reward, done, info = env.step(action.tolist())
        session.record_step(step, float(reward), bool(done))
        if done:
            break
    session.finalize_episode()
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


class CollectionSession:
    """Tracks per-episode state and dispatches activation-collection payloads."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy
        self._task_name: str = ""
        self._task_id: int = -1
        self._episode_id: int = -1
        self._env_id: int = 0
        self._prompt: str = ""

        self._inference_step: int = 0
        self._cumulative_reward: float = 0.0
        self._reward_at_last_inference: float = 0.0
        self._per_step_reward: List[float] = []
        self._per_step_success: List[bool] = []
        self._success: bool = False
        self._steps_to_success: int = -1

    def start_episode(
        self,
        task_name: str,
        task_id: int,
        episode_id: int,
        prompt: str,
        env_id: int = 0,
    ) -> None:
        """Reset all per-episode state and store identifiers for the next rollout."""
        self._task_name = task_name
        self._task_id = task_id
        self._episode_id = episode_id
        self._env_id = env_id
        self._prompt = prompt

        self._inference_step = 0
        self._cumulative_reward = 0.0
        self._reward_at_last_inference = 0.0
        self._per_step_reward = []
        self._per_step_success = []
        self._success = False
        self._steps_to_success = -1

    def make_collect_metadata(self, step: int) -> Dict[str, Any]:
        """Build the dict that goes into obs['__collect__'] for the next infer call.

        Bumps the inference_step counter so subsequent calls get sequential ids.
        """
        meta = {
            "task_name": self._task_name,
            "episode_id": int(self._episode_id),
            "env_id": int(self._env_id),
            "step": int(step),
            "inference_step": int(self._inference_step),
            "prompt": self._prompt,
            "cumulative_reward": float(self._cumulative_reward),
            "success_so_far": bool(self._success),
            "reward_since_last_inference": float(self._cumulative_reward - self._reward_at_last_inference),
        }
        self._inference_step += 1
        self._reward_at_last_inference = self._cumulative_reward
        return meta

    def record_step(self, step: int, reward: float, done: bool) -> None:
        """Update per-step bookkeeping after an env.step call."""
        self._per_step_reward.append(float(reward))
        self._per_step_success.append(bool(done))
        self._cumulative_reward += float(reward)
        if done and self._steps_to_success == -1:
            self._steps_to_success = int(step)
        if done:
            self._success = True

    def set_episode_result(self, success: bool, total_reward: float = 0.0) -> None:
        """Override episode outcome directly (e.g. from a post-hoc human label).

        Unlike record_step(done=True), this does not append a fake env step to
        per_step_reward / per_step_success. Use this when the success signal comes
        from outside the env loop (human labeling, external classifier, etc.).
        """
        self._success = success
        if success and self._steps_to_success == -1:
            self._steps_to_success = len(self._per_step_reward) - 1
        elif not success:
            self._steps_to_success = -1
        if total_reward != 0.0:
            self._cumulative_reward = total_reward

    def finalize_episode(self) -> Dict[str, Any]:
        """Send the __finalize_episode__ payload so the server writes
        episode-level metadata.json + rewards.npz. Returns the server's ack.
        """
        payload = {
            "__finalize_episode__": {
                "task_name": self._task_name,
                "episode_id": int(self._episode_id),
                "env_id": int(self._env_id),
                "prompt": self._prompt,
                "episode_success": bool(self._success),
                "total_reward": float(self._cumulative_reward),
                "steps_to_success": int(self._steps_to_success),
                "total_env_steps": int(len(self._per_step_reward)),
                "total_inference_steps": int(self._inference_step),
                "per_step_reward": [float(r) for r in self._per_step_reward],
                "per_step_success": [bool(s) for s in self._per_step_success],
            }
        }
        return self._policy.infer(payload)


class BatchCollectionSession:
    """Per-episode bookkeeping for vectorized rollouts (one inference call per
    batch of N envs).

    Used by the metaworld client, where ``num_envs`` parallel envs share one
    inference call. Owns N per-env states and shapes the protocol payloads as
    list-of-dicts so the server iterates over the batch dim of the captured
    intermediates.

    Same protocol as :class:`CollectionSession` (one ``infer`` per inference,
    one final ``infer`` per episode), but the magic-key payloads carry a list
    of N entries instead of a single dict. The server's ``CollectingPolicy``
    transparently dispatches to the batched handlers when it sees a list.

    Usage from a vectorized rollout loop:

        from openpi_client.collection_session import BatchCollectionSession

        session = BatchCollectionSession(client, num_envs=num_envs)
        session.start_episode(task_name, episode_id, prompt)
        for step in range(...):
            if not action_plan:
                element["__collect__"] = session.make_collect_metadata(step)
                action_chunk = client.infer(element)["actions"]  # (B, ah, ad)
                ...
            obs, reward, done, info = env.step(action)
            session.record_step(step, reward_array, done_array)
        session.finalize_episode()
    """

    def __init__(self, policy: Any, num_envs: int) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self._policy = policy
        self._num_envs = int(num_envs)

        self._task_name: str = ""
        self._episode_id: int = -1
        self._prompt: str = ""

        self._inference_step: int = 0
        self._cumulative_reward: np.ndarray = np.zeros(self._num_envs, dtype=np.float64)
        self._reward_at_last_inference: np.ndarray = np.zeros(self._num_envs, dtype=np.float64)
        self._per_step_reward: List[List[float]] = [[] for _ in range(self._num_envs)]
        self._per_step_success: List[List[bool]] = [[] for _ in range(self._num_envs)]
        self._success: np.ndarray = np.zeros(self._num_envs, dtype=bool)
        self._steps_to_success: np.ndarray = np.full(self._num_envs, -1, dtype=int)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def start_episode(self, task_name: str, episode_id: int, prompt: str) -> None:
        """Reset all per-env state for a new episode."""
        self._task_name = task_name
        self._episode_id = int(episode_id)
        self._prompt = prompt

        self._inference_step = 0
        self._cumulative_reward = np.zeros(self._num_envs, dtype=np.float64)
        self._reward_at_last_inference = np.zeros(self._num_envs, dtype=np.float64)
        self._per_step_reward = [[] for _ in range(self._num_envs)]
        self._per_step_success = [[] for _ in range(self._num_envs)]
        self._success = np.zeros(self._num_envs, dtype=bool)
        self._steps_to_success = np.full(self._num_envs, -1, dtype=int)

    def make_collect_metadata(self, step: int) -> List[Dict[str, Any]]:
        """Build the list that goes into obs['__collect__'] for the next infer call.

        Returns one dict per env. Bumps the inference_step counter by 1 (shared
        across envs since they advance in lockstep within the vectorized rollout).
        """
        meta = [
            {
                "task_name": self._task_name,
                "episode_id": int(self._episode_id),
                "env_id": int(env_id),
                "step": int(step),
                "inference_step": int(self._inference_step),
                "prompt": self._prompt,
                "cumulative_reward": float(self._cumulative_reward[env_id]),
                "success_so_far": bool(self._success[env_id]),
                "reward_since_last_inference": float(
                    self._cumulative_reward[env_id] - self._reward_at_last_inference[env_id]
                ),
            }
            for env_id in range(self._num_envs)
        ]
        self._inference_step += 1
        self._reward_at_last_inference = self._cumulative_reward.copy()
        return meta

    def record_step(self, step: int, reward: Sequence[float], done: Sequence[bool]) -> None:
        """Update per-env bookkeeping after a vectorized env.step call.

        ``reward`` and ``done`` must have length ``num_envs``.
        """
        reward_arr = np.asarray(reward, dtype=np.float64)
        done_arr = np.asarray(done, dtype=bool)
        if reward_arr.shape != (self._num_envs,):
            raise ValueError(f"reward must have shape ({self._num_envs},), got {reward_arr.shape}")
        if done_arr.shape != (self._num_envs,):
            raise ValueError(f"done must have shape ({self._num_envs},), got {done_arr.shape}")
        self._cumulative_reward += reward_arr
        for env_id in range(self._num_envs):
            self._per_step_reward[env_id].append(float(reward_arr[env_id]))
            self._per_step_success[env_id].append(bool(done_arr[env_id]))
            if done_arr[env_id] and self._steps_to_success[env_id] == -1:
                self._steps_to_success[env_id] = int(step)
            if done_arr[env_id]:
                self._success[env_id] = True

    def finalize_episode(self) -> Dict[str, Any]:
        """Send a list-shaped __finalize_episode__ payload covering all envs.

        Server returns ``{"ack": True, "episode_dirs": [...]}``.
        """
        payload = {
            "__finalize_episode__": [
                {
                    "task_name": self._task_name,
                    "episode_id": int(self._episode_id),
                    "env_id": int(env_id),
                    "prompt": self._prompt,
                    "episode_success": bool(self._success[env_id]),
                    "total_reward": float(self._cumulative_reward[env_id]),
                    "steps_to_success": int(self._steps_to_success[env_id]),
                    "total_env_steps": int(len(self._per_step_reward[env_id])),
                    "total_inference_steps": int(self._inference_step),
                    "per_step_reward": [float(r) for r in self._per_step_reward[env_id]],
                    "per_step_success": [bool(s) for s in self._per_step_success[env_id]],
                }
                for env_id in range(self._num_envs)
            ]
        }
        return self._policy.infer(payload)
