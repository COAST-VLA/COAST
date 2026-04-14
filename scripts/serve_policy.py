"""
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/

"""

import dataclasses
import enum
import logging
import os
import pathlib
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.serving.activation_collector import CollectingPolicy
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    ROBOCASA = "robocasa"


# Default conceptor NPZ paths keyed by --env when --steer is set.
# Override with --conceptor_npz to point at a different file.
DEFAULT_CONCEPTOR_NPZ: dict[EnvMode, str] = {
    EnvMode.LIBERO: "conceptors/libero_conceptors.npz",
    EnvMode.ROBOCASA: "conceptors/robocasa_conceptors.npz",
}


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Use PyTorch backend for inference. Auto-converts the JAX checkpoint if needed.
    pytorch: bool = False

    # Apply torch.compile(sample_actions, mode="max-autotune") at model load. Off by
    # default for safety: compile trades a 30-60s first-call warmup for ~2x steady-state
    # speedup on baseline inference, and is incompatible with some forward-hook patterns
    # (activation collection, steering) in non-trivial call paths. Opt in when you are
    # running a long baseline-only eval and want the throughput.
    torch_compile: bool = False

    # Enable activation-collection mode. The server wraps the policy in CollectingPolicy
    # and rejects any client request that doesn't include the __collect__ or __finalize_episode__
    # magic key. Requires --pytorch (infer_with_intermediates is PyTorch-only).
    collect_activations: bool = False
    # Server-side root directory for collected activations. Activations are written to
    # <output_dir>/<checkpoint_step>/<task_name>/episode_NNN_env_NNN/step_NNNN/. Only
    # used when --collect_activations is set.
    output_dir: str = "activations"

    # Enable conceptor steering. When set, the server wraps the policy in
    # SteeredPolicyWrapper and dispatches on obs["__steering__"] (see
    # src/openpi/serving/steering.py). Implies --pytorch
    # (sample_actions_with_steering is PyTorch-only).
    steer: bool = False
    # Override the default conceptor NPZ path. If None, looks up a default based on --env
    # (DEFAULT_CONCEPTOR_NPZ). Only used when --steer is set.
    conceptor_npz: str | None = None

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            if args.pytorch:
                from openpi.models_pytorch.convert import ensure_pytorch_checkpoint

                ensure_pytorch_checkpoint(args.policy.dir, args.policy.config)
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    if args.collect_activations:
        if not args.pytorch:
            raise ValueError("--collect_activations requires --pytorch (infer_with_intermediates is PyTorch-only).")
        if not isinstance(args.policy, Checkpoint):
            raise ValueError("--collect_activations requires --policy=checkpoint (default policies are not supported).")

    if args.steer:
        if not args.pytorch:
            raise ValueError("--steer requires --pytorch (sample_actions_with_steering is PyTorch-only).")
        if args.collect_activations:
            raise ValueError("--steer and --collect_activations are mutually exclusive.")

    # torch.compile is off by default. The model's __init__ checks TORCH_COMPILE_DISABLE
    # and skips the compile wrap when set. Must be set before create_policy() below.
    if not args.torch_compile:
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

    policy = create_policy(args)

    if args.collect_activations:
        assert isinstance(args.policy, Checkpoint)  # narrowed above
        checkpoint_step = pathlib.Path(args.policy.dir).name
        output_root = pathlib.Path(args.output_dir).resolve()
        logging.info(
            "Activation collection enabled (checkpoint_step=%s, output_root=%s)",
            checkpoint_step,
            output_root,
        )
        policy = CollectingPolicy(
            policy=policy,
            output_root=output_root,
            checkpoint_step=checkpoint_step,
            policy_dir=args.policy.dir,
            config_name=args.policy.config,
        )

    if args.steer:
        from openpi.serving.steering import SteeredPolicyWrapper

        npz_path = args.conceptor_npz or DEFAULT_CONCEPTOR_NPZ.get(args.env)
        if npz_path is None:
            raise ValueError(
                f"--steer with --env={args.env.value} has no default conceptor NPZ. "
                f"Pass --conceptor_npz explicitly. Supported defaults: {list(DEFAULT_CONCEPTOR_NPZ)}"
            )
        device = str(policy._pytorch_device)  # noqa: SLF001
        logging.info("Steering enabled: loading conceptor NPZ from %s (device=%s)", npz_path, device)
        policy = SteeredPolicyWrapper(policy, conceptor_npz_path=npz_path, device=device)

    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
