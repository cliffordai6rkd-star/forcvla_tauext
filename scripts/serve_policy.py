import dataclasses
import enum
import logging
import socket
import time

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
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
    # Compile local LeRobot v3 ForceVLA policies before listening for clients.
    warmup: bool = True
    # Maximum size of a packed observation (including all images), in MiB.
    max_message_size_mib: int = 32

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="gs://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="gs://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="gs://openpi-assets/checkpoints/pi0_fast_libero",
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
            train_config = _config.get_config(args.policy.config)
            policy = _policy_config.create_trained_policy(
                train_config, args.policy.dir, default_prompt=args.default_prompt
            )
            if args.warmup and isinstance(train_config.data, _config.LeRobotV3ForceDataConfig):
                image_keys = ["image", "right_wrist_image"]
                if train_config.data.left_wrist_image_key is not None:
                    image_keys.append("wrist_image")
                observation = {
                    "state": np.zeros(7, dtype=np.float32),
                    "tau_ext": np.zeros(7, dtype=np.float32),
                    **{key: np.zeros((224, 224, 3), dtype=np.uint8) for key in image_keys},
                }
                prompt = args.default_prompt if args.default_prompt is not None else train_config.data.prompt
                if prompt is not None:
                    observation["prompt"] = prompt
                logging.info("Warming policy before opening the port; first JAX compilation may take a minute...")
                started = time.monotonic()
                policy.warmup(observation)
                logging.info("Policy warmup finished in %.2f seconds; ready to listen", time.monotonic() - started)
            return policy
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    if args.max_message_size_mib <= 0:
        raise ValueError("max_message_size_mib must be positive")
    policy = create_policy(args)
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
        max_message_size=args.max_message_size_mib * 1024 * 1024,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
