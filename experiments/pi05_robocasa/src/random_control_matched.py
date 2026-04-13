"""Spectrally-matched random control.

For each task, loads the best-global conceptor C, keeps its eigenvalues,
replaces the eigenvectors with a random orthogonal basis, and evaluates the
resulting "scrambled" conceptor at the same beta. This is a tighter baseline
than the prior unmatched random control: it preserves the *shape* of the
spectrum (how much each singular direction is attenuated) while randomizing
*which* directions are selected.

Hyperparameters are fixed per task to the best-global config identified in the
existing sweep — no grid search.
"""
import dataclasses
import json
import logging
import pathlib

import numpy as np
import torch
import tyro

from conceptor_steering import (  # type: ignore
    ConceptorSteeringHook,
    REPO_ROOT,
    SteeredPolicyWrapper,
    get_global_contrastive,
    load_npz,
    run_condition,
    start_server_background,
)

logger = logging.getLogger(__name__)

# Best-global (layer, alpha, beta) from the sweep.
BEST_GLOBAL = {
    "CloseFridge":               (11, 0.5, 0.3),
    "CoffeeSetupMug":            (11, 0.1, 0.1),
    "OpenDrawer":                (11, 0.5, 0.1),
    "OpenStandMixerHead":        (11, 0.5, 0.1),
    "PickPlaceCounterToCabinet": (5,  0.1, 0.1),
    "PickPlaceCounterToStove":   (11, 0.1, 0.1),
    "TurnOnElectricKettle":      (11, 0.5, 0.1),
}


def spectrum_matched_random(C: np.ndarray, seed: int) -> np.ndarray:
    """Return C' with the same eigenvalues as C but random orthogonal basis."""
    C_sym = 0.5 * (C + C.T)  # enforce symmetry for eigh
    eigvals, _ = np.linalg.eigh(C_sym)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal(C.shape))
    return (Q @ np.diag(eigvals) @ Q.T).astype(np.float32)


@dataclasses.dataclass
class Args:
    task: str
    config: str = "pi05_robocasa"
    checkpoint_dir: str = (
        "/vast/projects/ungar/stellar/miaom/openpi-metaworld/"
        "checkpoints/pi05_pretrain_human300/multitask_learning/75000"
    )
    num_episodes: int = 15
    port: int = 8600
    seed: int = 42
    output_dir: str = "experiments/pi05_robocasa/steering_results"


def main(args: Args):
    if args.task not in BEST_GLOBAL:
        raise ValueError(f"No best-global config for {args.task}")
    layer, alpha, beta = BEST_GLOBAL[args.task]

    task_output_dir = pathlib.Path(args.output_dir) / args.task
    task_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = task_output_dir / "summary.json"

    # Load model + conceptors
    from openpi.models_pytorch.convert import ensure_pytorch_checkpoint
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    ensure_pytorch_checkpoint(args.checkpoint_dir, args.config)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    device = str(policy._pytorch_device)  # noqa: SLF001

    npz = load_npz()
    C_global = get_global_contrastive(npz, args.task, layer, alpha)
    C_rand = spectrum_matched_random(C_global, seed=args.seed)

    cond_name = f"random_matched_L{layer}_a{alpha}_b{beta}"
    logger.info(f"{args.task}: matched random at {cond_name}")
    logger.info(f"  eigenvalue range: [{np.linalg.eigvalsh(0.5*(C_global+C_global.T)).min():.4f}, "
                f"{np.linalg.eigvalsh(0.5*(C_global+C_global.T)).max():.4f}]")

    wrapper = SteeredPolicyWrapper(policy, steering_hooks=None)
    start_server_background(wrapper, args.port)
    hook = ConceptorSteeringHook(C_rand, beta=beta, device=device)
    wrapper.update_hooks([(layer, hook)])
    r = run_condition(args.task, args.port, args.num_episodes, cond_name, task_output_dir)

    # Merge-on-save into summary.json
    merged = {}
    if summary_path.exists():
        try:
            existing = json.load(open(summary_path))
            for e in existing.get("conditions", []):
                if e is not None and "condition" in e:
                    merged[e["condition"]] = e
        except (json.JSONDecodeError, OSError):
            pass
    merged[cond_name] = r
    sorted_results = sorted(merged.values(),
                            key=lambda x: x.get("success_rate", 0), reverse=True)
    tmp = summary_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"task": args.task, "conditions": sorted_results}, f, indent=2)
    tmp.replace(summary_path)

    logger.info(f"Done. {cond_name}: SR={r['success_rate']:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main(tyro.cli(Args))
