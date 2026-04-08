"""Per-episode bookkeeping for activation collection from a libero rollout.

This is the only LIBERO-side component that knows about the activation
collection protocol. It tracks per-episode state (cumulative reward, per-step
rewards/successes, inference_step counter) and shapes the metadata payloads
that get attached to each WebSocket request as `__collect__` /
`__finalize_episode__` magic keys. The server's CollectingPolicy uses those
keys to dispatch and write activations to its own filesystem.

Usage from inside an eval loop:

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

from typing import Any, Dict, List


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
            "reward_since_last_inference": float(
                self._cumulative_reward - self._reward_at_last_inference
            ),
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
