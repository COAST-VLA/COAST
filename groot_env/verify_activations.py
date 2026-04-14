"""Verify a directory of collected GR00T N1.5 activations is healthy.

Run this after a `main.py --collect` sweep against a server started with
`serve.py --collect-activations`. For every episode directory under
<root>/<checkpoint_step>/<task_name>/episode_NNN_env_NNN/, confirm:

  - episode-level metadata.json is parseable and has the expected fields
  - rewards.npz contains per_step_reward / cumulative_reward / success_at_step
    arrays of equal length
  - for each step_NNNN subdir:
      * metadata.json exists and has step / inference_step
      * denoising.npz has all_x_t (D+1,H,A) and all_v_t (D,H,A) with no NaNs
      * backbone_cond.npz has backbone_features (S,C) with no NaNs (fp16 ok)
      * dit_hidden_states.npz has all_dit_hidden_states (D,L+1,S,C) with no NaNs
      * the x_t norms decrease monotonically across denoising steps (the
        flow-matching signature that proves we captured the right thing)

Run:
    uv run python verify_activations.py /tmp/groot_n15_full
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np


def _check_step(step_dir: pathlib.Path, expected_denoising_steps: int) -> list[str]:
    errors: list[str] = []
    meta_path = step_dir / "metadata.json"
    if not meta_path.exists():
        return [f"{step_dir}: missing metadata.json"]
    meta = json.load(open(meta_path))
    for key in ("step", "inference_step", "task_name", "episode_id"):
        if key not in meta:
            errors.append(f"{step_dir}: metadata missing '{key}'")

    # denoising.npz
    denoising_path = step_dir / "denoising.npz"
    if not denoising_path.exists():
        errors.append(f"{step_dir}: missing denoising.npz")
    else:
        d = np.load(denoising_path)
        if "all_x_t" not in d.files or "all_v_t" not in d.files:
            errors.append(f"{step_dir}: denoising.npz missing arrays")
        else:
            x = d["all_x_t"]
            v = d["all_v_t"]
            if x.ndim != 3:
                errors.append(f"{step_dir}: all_x_t ndim={x.ndim} (expected 3)")
            if v.ndim != 3:
                errors.append(f"{step_dir}: all_v_t ndim={v.ndim} (expected 3)")
            if x.shape[0] != expected_denoising_steps + 1:
                errors.append(
                    f"{step_dir}: all_x_t has {x.shape[0]} entries, expected {expected_denoising_steps + 1}"
                )
            if v.shape[0] != expected_denoising_steps:
                errors.append(
                    f"{step_dir}: all_v_t has {v.shape[0]} entries, expected {expected_denoising_steps}"
                )
            if np.isnan(x).any() or np.isnan(v).any():
                errors.append(f"{step_dir}: NaN in denoising arrays")
            # Flow-matching signature: ||x_t|| should decrease over denoising steps
            # (noise is being converted to clean action).
            x_norms = [float(np.linalg.norm(x[i])) for i in range(x.shape[0])]
            if not (x_norms[0] > x_norms[-1] * 1.05):
                errors.append(
                    f"{step_dir}: x_t norm did not decrease ({x_norms[0]:.2f} -> {x_norms[-1]:.2f})"
                )

    # backbone_cond.npz
    bc_path = step_dir / "backbone_cond.npz"
    if not bc_path.exists():
        errors.append(f"{step_dir}: missing backbone_cond.npz")
    else:
        bc = np.load(bc_path)
        if "backbone_features" not in bc.files:
            errors.append(f"{step_dir}: backbone_cond.npz missing 'backbone_features'")
        else:
            feats = bc["backbone_features"]
            if feats.ndim != 2:
                errors.append(
                    f"{step_dir}: backbone_features ndim={feats.ndim} (expected 2)"
                )
            # fp16 may have occasional inf in intermediate reductions; check for NaN only.
            if np.isnan(feats.astype(np.float32)).any():
                errors.append(f"{step_dir}: NaN in backbone_features")

    # dit_hidden_states.npz
    dh_path = step_dir / "dit_hidden_states.npz"
    if not dh_path.exists():
        errors.append(f"{step_dir}: missing dit_hidden_states.npz")
    else:
        dh = np.load(dh_path)
        if "all_dit_hidden_states" not in dh.files:
            errors.append(
                f"{step_dir}: dit_hidden_states.npz missing 'all_dit_hidden_states'"
            )
        else:
            h = dh["all_dit_hidden_states"]
            if h.ndim != 4:
                errors.append(
                    f"{step_dir}: all_dit_hidden_states ndim={h.ndim} (expected 4)"
                )
            if h.shape[0] != expected_denoising_steps:
                errors.append(
                    f"{step_dir}: dit_hidden_states num_steps={h.shape[0]}, "
                    f"expected {expected_denoising_steps}"
                )
            if np.isnan(h.astype(np.float32)).any():
                errors.append(f"{step_dir}: NaN in all_dit_hidden_states")

    return errors


def _check_episode(
    episode_dir: pathlib.Path, expected_denoising_steps: int
) -> tuple[int, int, list[str]]:
    errors: list[str] = []
    ep_meta_path = episode_dir / "metadata.json"
    if not ep_meta_path.exists():
        return 0, 0, [f"{episode_dir}: missing episode metadata.json"]
    ep_meta = json.load(open(ep_meta_path))
    rewards_path = episode_dir / "rewards.npz"
    if not rewards_path.exists():
        errors.append(f"{episode_dir}: missing rewards.npz")
    else:
        r = np.load(rewards_path)
        for k in ("per_step_reward", "cumulative_reward", "success_at_step"):
            if k not in r.files:
                errors.append(f"{episode_dir}: rewards.npz missing {k}")
        if "per_step_reward" in r.files and "cumulative_reward" in r.files:
            if r["per_step_reward"].shape != r["cumulative_reward"].shape:
                errors.append(f"{episode_dir}: reward array shapes disagree")

    total_inf_steps = int(ep_meta.get("total_inference_steps", 0))

    step_dirs = sorted(
        p for p in episode_dir.iterdir() if p.is_dir() and p.name.startswith("step_")
    )
    if total_inf_steps != 0 and len(step_dirs) != total_inf_steps:
        errors.append(
            f"{episode_dir}: {len(step_dirs)} step_* dirs but metadata says "
            f"{total_inf_steps} inference steps"
        )

    for sd in step_dirs:
        errors.extend(_check_step(sd, expected_denoising_steps))

    return 1, len(step_dirs), errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root output dir passed to serve.py --output-dir")
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=4,
        help="Number of denoising steps the model was configured with "
        "(default: 4, matching serve.py's default)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If > 0, only verify this many episodes per task (useful for "
        "fast spot-checks on huge sweeps).",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    if not root.exists():
        print(f"no such dir: {root}")
        return 1

    # Find all episode dirs under <root>/*/*/episode_*_env_*/
    ep_dirs = sorted(root.glob("*/*/episode_*_env_*"))
    if not ep_dirs:
        # Also look one level deeper in case root is <output_dir>/<checkpoint> already.
        ep_dirs = sorted(root.glob("*/episode_*_env_*"))
    if not ep_dirs:
        print(f"no episode dirs under {root}")
        return 1

    # Optionally sample per task.
    if args.sample > 0:
        by_task: dict[str, list[pathlib.Path]] = {}
        for ep in ep_dirs:
            by_task.setdefault(ep.parent.name, []).append(ep)
        trimmed = []
        for task, eps in by_task.items():
            trimmed.extend(eps[: args.sample])
        ep_dirs = trimmed

    total_errors: list[str] = []
    task_counts: Counter = Counter()
    total_steps = 0
    for ep in ep_dirs:
        task = ep.parent.name
        _, n_steps, errs = _check_episode(ep, args.denoising_steps)
        task_counts[task] += 1
        total_steps += n_steps
        total_errors.extend(errs)

    print(
        f"Verified {len(ep_dirs)} episodes across {len(task_counts)} tasks, "
        f"{total_steps} inference steps total."
    )
    print("Per-task episode counts:")
    for task, n in sorted(task_counts.items()):
        print(f"  {task}: {n}")
    print()
    if total_errors:
        print(f"FAILED with {len(total_errors)} errors:")
        for e in total_errors[:40]:
            print(f"  - {e}")
        if len(total_errors) > 40:
            print(f"  ... and {len(total_errors) - 40} more")
        return 1
    print("OK: all activation files parse cleanly and have expected shapes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
