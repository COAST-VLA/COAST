"""Serve an NVIDIA GR00T N1.5 checkpoint via openpi's WebSocket protocol.

GR00T N1.5 has deep dependency conflicts with the openpi root venv (torch 2.5.1
vs 2.7.1, wandb 0.18.0 vs >=0.19.1, etc.), so it lives in its own venv here at
the repo root, peer to `examples/` (which holds CLIENTS) and `scripts/` (which
holds the pi0 server). Every openpi client in `examples/*_env/` can target this
server unchanged — the wire protocol matches `scripts/serve_policy.py`.

    cd groot_env
    uv sync
    uv run python serve.py \\
        --model-path ../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000 \\
        --port 8000

Then from any env client:

    cd examples/robocasa_env
    MUJOCO_GL=egl uv run python main.py --env_name CloseBlenderLid

No client-side changes are needed. `groot_adapter.py` translates the openpi
robocasa client's flat observation dict into GR00T N1.5's flat
{video.X, state.Y, annotation.Z} format, then concatenates GR00T's
per-action-key outputs back into a single (action_horizon, action_dim) array.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import socket

import tyro

import groot_activation_collector
import groot_adapter
import websocket_policy_server


@dataclasses.dataclass
class Args:
    # HuggingFace model id or local checkpoint directory. Defaults to the
    # downloaded N1.5 robocasa multitask checkpoint.
    model_path: str = (
        "../checkpoints/groot_n15/gr00t_n1-5/multitask_learning/checkpoint-120000"
    )
    # Embodiment preset. Currently only "robocasa" is implemented; libero /
    # metaworld will follow the same pattern (their own video/state builders).
    embodiment: str = "robocasa"
    # CUDA device. The N1.5 3B checkpoint takes ~7GB in bfloat16.
    device: str = "cuda:0"
    # WebSocket port. Must match the --port argument on the client.
    port: int = 8000
    # Number of denoising steps for the action diffusion head. NVIDIA's
    # inference_service.py defaults to 4, which is what the published numbers
    # were measured at.
    denoising_steps: int = 4
    # Enable activation-collection mode. The server wraps the policy in
    # `groot_activation_collector.CollectingPolicy` and rejects any request that
    # doesn't carry the __collect__ or __finalize_episode__ magic keys. Mirrors
    # `scripts/serve_policy.py --collect_activations` on the pi0 side.
    collect_activations: bool = False
    # Server-side root directory for saved activations. Activations land at
    # <output_dir>/<checkpoint_step>/<task_name>/episode_NNN_env_NNN/step_NNNN/.
    # Only used when --collect_activations is set. Default name matches the
    # `pi05-*-activations-v1-*` convention in the repo root .gitignore.
    output_dir: str = "../groot_n15-robocasa-activations-v1-15env"


def _build_policy(args: Args):
    if args.embodiment == "robocasa":
        return groot_adapter.make_robocasa_policy(
            model_path=args.model_path,
            device=args.device,
            denoising_steps=args.denoising_steps,
        )
    raise ValueError(
        f"Unknown embodiment {args.embodiment!r}. Currently only 'robocasa' is supported."
    )


def main(args: Args) -> None:
    logging.info(
        "Loading GR00T N1.5: model=%s, embodiment=%s, device=%s, denoising_steps=%d",
        args.model_path,
        args.embodiment,
        args.device,
        args.denoising_steps,
    )
    policy = _build_policy(args)

    metadata = {
        "backend": "groot_n15",
        "model_path": args.model_path,
        "embodiment": args.embodiment,
        "denoising_steps": args.denoising_steps,
    }

    if args.collect_activations:
        # Label activations by the checkpoint's final directory component (e.g.
        # "checkpoint-120000"), mirroring pi0's convention.
        checkpoint_step = pathlib.Path(args.model_path).name
        output_root = pathlib.Path(args.output_dir).resolve()
        logging.info(
            "Activation collection enabled (checkpoint_step=%s, output_root=%s)",
            checkpoint_step,
            output_root,
        )
        policy = groot_activation_collector.CollectingPolicy(
            policy=policy,
            output_root=output_root,
            checkpoint_step=checkpoint_step,
            policy_dir=args.model_path,
            config_name=f"groot_n15_{args.embodiment}",
        )
        metadata.update(policy.metadata)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
