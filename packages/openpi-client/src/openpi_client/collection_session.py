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

    def set_episode_outcome(self, success: bool, total_reward: float) -> None:
        """Manually set the terminal episode outcome.

        Use this for clients where success is determined out-of-band, after
        the rollout loop ends — for example human-graded real-robot rollouts
        where `env.step()` returns no per-step reward or done flag. Droid is
        the canonical case: the user enters a success score at a prompt
        after the rollout ends, and there is no in-loop signal to feed into
        `record_step`.

        This sets `episode_success`, `total_reward`, and `steps_to_success`
        on the session so the next `finalize_episode()` call writes them
        correctly. It also adjusts the last entry of `per_step_reward` so
        the rewards.npz cumulative sum equals `total_reward`, and flips
        the last entry of `per_step_success` to True if `success` is True.
        Both adjustments keep the on-disk schema consistent with what
        `tests/test_activations.py::TestEpisodeMetadata` asserts (in
        particular `test_rewards_cumulative_matches_total`).

        Call this AFTER the rollout loop and BEFORE `finalize_episode()`.
        Idempotent — recomputes the per-step adjustment from the current
        sum each call, so calling it twice with different values just
        ends up reflecting the latest call.

        If no `record_step` calls have happened (the per-step arrays are
        empty), this just sets the episode-level scalars; the rewards.npz
        will have empty arrays and there is no per-step entry to attribute
        the reward to.
        """
        self._success = bool(success)
        if not self._per_step_reward:
            self._cumulative_reward = float(total_reward)
            self._steps_to_success = -1
            return

        last_idx = len(self._per_step_reward) - 1
        delta = float(total_reward) - sum(self._per_step_reward)
        self._per_step_reward[last_idx] = float(self._per_step_reward[last_idx]) + delta
        self._cumulative_reward = float(total_reward)
        if success:
            self._per_step_success[last_idx] = True
            self._steps_to_success = last_idx
        else:
            self._steps_to_success = -1

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
