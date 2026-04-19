"""
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_metaworld \
    --policy.dir=checkpoints/pi05_metaworld/pi05_metaworld_test/5000/

"""

import dataclasses
import enum
import logging
import pathlib
import socket

import tyro

from openpi.models import model as _model
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

    # Enable activation-collection mode. The server wraps the policy in CollectingPolicy
    # and rejects any client request that doesn't include the __collect__ or __finalize_episode__
    # magic key. Requires --pytorch (infer_with_intermediates is PyTorch-only).
    collect_activations: bool = False
    # Server-side root directory for collected activations. Activations are written to
    # <output_dir>/<checkpoint_step>/<task_name>/episode_NNN_env_NNN/step_NNNN/. Only
    # used when --collect_activations is set.
    output_dir: str = "activations"

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


def create_default_policy(
    env: EnvMode, *, default_prompt: str | None = None, use_pytorch: bool = False
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            use_pytorch=use_pytorch,
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
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                use_pytorch=args.pytorch,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt, use_pytorch=args.pytorch)


def main(args: Args) -> None:
    if args.collect_activations:
        if not isinstance(args.policy, Checkpoint):
            raise ValueError("--collect_activations requires --policy=checkpoint (default policies are not supported).")
        # pi0 / pi0.5 capture intermediates through PyTorch forward hooks; pi0-fast
        # has no PyTorch port of the autoregressive decode, so it captures through
        # JAX. Pick the backend here based on the configured model_type.
        model_type = _config.get_config(args.policy.config).model.model_type
        if model_type == _model.ModelType.PI0_FAST and args.pytorch:
            raise ValueError(
                "--pytorch cannot be combined with a pi0-fast model — there is no PyTorch port "
                "of the autoregressive decode. Drop --pytorch and let the server load the JAX checkpoint."
            )
        if model_type != _model.ModelType.PI0_FAST and not args.pytorch:
            raise ValueError(
                f"--collect_activations requires --pytorch for {model_type.value} "
                "(infer_with_intermediates uses PyTorch forward hooks for diffusion models)."
            )

    policy = create_policy(args)

    if args.collect_activations:
        assert isinstance(args.policy, Checkpoint)  # narrowed above
        checkpoint_step = pathlib.Path(args.policy.dir).name
        output_root = pathlib.Path(args.output_dir).resolve()
        model_type = _config.get_config(args.policy.config).model.model_type
        logging.info(
            "Activation collection enabled (checkpoint_step=%s, output_root=%s, model_type=%s)",
            checkpoint_step,
            output_root,
            model_type.value,
        )
        policy = CollectingPolicy(
            policy=policy,
            output_root=output_root,
            checkpoint_step=checkpoint_step,
            policy_dir=args.policy.dir,
            config_name=args.policy.config,
            model_type=model_type,
        )

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
