import dataclasses
import enum
import logging
import os
import socket

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import rtc_policy as _rtc_policy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and v.strip() != "" else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None and v.strip() != "" else default


# ====== G1 desk action layout ======
# Keep gripper outputs continuous on the server; the G1 client applies thresholds.
LEFT_GRIPPER_INDEX = 7
RIGHT_GRIPPER_INDEX = 15
# ==================================


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # ===== RTC (Real-Time Chunking) =====
    # Client-prefix guidance: the client sends back the still-to-be-executed
    # portion of the previous chunk (remaining_action_chunk + prefix_len) and
    # the server guides the new chunk toward it. CLI flags override; otherwise
    # fall back to RTC_* env vars (used by the run_serve_*_rtc.sh script).
    rtc: bool = dataclasses.field(default_factory=lambda: _env_bool("RTC_ENABLE", False))
    rtc_guidance_weight: float = dataclasses.field(default_factory=lambda: _env_float("RTC_GUIDANCE_WEIGHT", 5.0))
    rtc_decay_steps: int = dataclasses.field(default_factory=lambda: _env_int("RTC_DECAY_STEPS", 4))
    rtc_schedule: str = dataclasses.field(default_factory=lambda: os.environ.get("RTC_SCHEDULE", "exp"))


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


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None):
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args):
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


# ====== Core: leave gripper outputs continuous ======
def _pass_through_gripper(arr: np.ndarray) -> np.ndarray:
    return np.array(arr, copy=True)


class RawGripperPolicy:
    def __init__(self, base_policy):
        self._base_policy = base_policy

    @property
    def metadata(self):
        return self._base_policy.metadata

    def infer(self, obs):
        action = self._base_policy.infer(obs)

        if not isinstance(action, dict):
            return action

        # 优先处理 actions
        if "actions" in action and isinstance(action["actions"], np.ndarray):
            action["actions"] = _pass_through_gripper(action["actions"])
            return action

        # 兼容其他字段
        for k in ["action", "robot_actions"]:
            if k in action and isinstance(action[k], np.ndarray):
                action[k] = _pass_through_gripper(action[k])
                return action

        return action


def main(args: Args) -> None:
    policy = create_policy(args)

    logging.info(
        "Server gripper binarization disabled; passing raw gripper outputs to client (left=%d, right=%d)",
        LEFT_GRIPPER_INDEX,
        RIGHT_GRIPPER_INDEX,
    )

    if args.rtc:
        if not isinstance(policy, _policy.Policy):
            raise ValueError("RTC requires a JAX Policy (checkpoint policy); got %r" % type(policy))
        logging.info(
            "RTC enabled (client-prefix guidance): guidance_weight=%.3f decay_steps=%d schedule=%s",
            args.rtc_guidance_weight,
            args.rtc_decay_steps,
            args.rtc_schedule,
        )
        policy = _rtc_policy.RTCPolicy(
            policy,
            guidance_weight=args.rtc_guidance_weight,
            decay_steps=args.rtc_decay_steps,
            schedule=args.rtc_schedule,
        )
    else:
        logging.info("RTC disabled; using vanilla flow-matching sampler.")

    policy_metadata = policy.metadata

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

## 002246 head 3
## 20260320 head 10
##3636 head 20
##4635 head 30

##005009 40
##5326 50_1
##5618 50_2


##014041  50
## 15829
