"""关节策略的数据合同、参数配置与纯数组变换。

机器人、当前 OpenPI checkpoint、RTC 队列及 rollout 全部统一使用 16 维顺序：
``[left_arm7, left_hand, right_arm7, right_hand]``，夹爪索引固定为 7 和 15。
请求状态、模型响应和动作块仅做严格校验与复制，不再进行任何旧式索引重排。
``JointArgs`` 同时集中声明本地端口转发所需的 WS 超时与重试参数，并在硬件初始化前校验。
本模块仅依赖 NumPy 和 ``inference_common`` 轻量工具，不加载硬件、WebSocket、CLI 或 prompt。
同步 joint 入口、``policy_joint_rtc`` 与 ``rollout_schemas`` 调用这里的公开接口。
下发前必须通过 16 维/有限值检查，并限制双臂单步增量、二值化两个夹爪命令。
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from inference_common import resize_image, validate_websocket_args


JOINT_DIM = 16

ROBOT_LEFT_GRIPPER = 7
ROBOT_RIGHT_GRIPPER = 15
MODEL_LEFT_GRIPPER = 7
MODEL_RIGHT_GRIPPER = 15

# Explicit aliases for callers that use ``openpi`` rather than ``model`` in
# their terminology.
OPENPI_LEFT_GRIPPER = MODEL_LEFT_GRIPPER
OPENPI_RIGHT_GRIPPER = MODEL_RIGHT_GRIPPER


def checked_joint_vector(value: Any, *, name: str) -> np.ndarray:
    """Return a finite float32 copy of one strict 16-D joint vector."""

    array = np.asarray(value, dtype=np.float32)
    if array.shape != (JOINT_DIM,):
        raise ValueError(
            f"Expected {name} shape ({JOINT_DIM},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf: {array}")
    return array.copy()


def _checked_joint_chunk(value: Any, *, name: str) -> np.ndarray:
    """Return a finite float32 copy of a strict ``H x 16`` joint chunk."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != JOINT_DIM:
        raise ValueError(
            f"Expected {name} shape (H, {JOINT_DIM}), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array.copy()


def robot_joint_to_openpi_joint(robot_joint: Any) -> np.ndarray:
    """Copy robot order ``[L7, Lh, R7, Rh]`` to identical model order."""

    return checked_joint_vector(robot_joint, name="robot joint")


def openpi_joint_to_robot_joint(openpi_joint: Any) -> np.ndarray:
    """Copy model order ``[L7, Lh, R7, Rh]`` to identical robot order."""

    return checked_joint_vector(openpi_joint, name="OpenPI joint")


def robot_chunk_to_openpi_chunk(robot_chunk: Any) -> np.ndarray:
    """Copy an ``H x 16`` robot-order chunk to identical model order."""

    return _checked_joint_chunk(robot_chunk, name="robot joint chunk")


def openpi_chunk_to_robot_chunk(openpi_chunk: Any) -> np.ndarray:
    """Copy an ``H x 16`` model-order chunk to identical robot order."""

    return _checked_joint_chunk(openpi_chunk, name="OpenPI joint chunk")


def prepare_policy_request(
    obs: dict[str, Any],
    prompt: str,
    *,
    resize_images: bool,
    image_size: int,
) -> dict[str, Any]:
    """Build the request fields expected by the deployed joint policy."""

    required = (
        "image/head",
        "image/left_wrist",
        "image/right_wrist",
        "state/joint",
    )
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(
            f"RobotEnv observation is missing {missing}; "
            f"available={sorted(obs.keys())}"
        )

    return {
        "observation/image": resize_image(
            obs["image/head"], enabled=resize_images, image_size=image_size
        ),
        "observation/wrist_image": resize_image(
            obs["image/left_wrist"], enabled=resize_images, image_size=image_size
        ),
        "observation/second_image": resize_image(
            obs["image/right_wrist"], enabled=resize_images, image_size=image_size
        ),
        "observation/state": robot_joint_to_openpi_joint(obs["state/joint"]),
        "prompt": str(prompt),
    }


def extract_openpi_actions(
    response: dict[str, Any],
    *,
    max_horizon: int,
) -> np.ndarray:
    """Extract a finite model-order action chunk with horizon ``1..max``."""

    if max_horizon <= 0:
        raise ValueError(f"max_horizon must be positive, got {max_horizon}")
    if "actions" not in response:
        raise KeyError(
            f"Policy response has no 'actions'; keys={sorted(response.keys())}"
        )

    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError(f"Expected action batch size 1, got {actions.shape}")
        actions = actions[0]
    elif actions.ndim == 1:
        actions = actions[None, :]

    actions = _checked_joint_chunk(actions, name="OpenPI action chunk")
    if not 1 <= len(actions) <= max_horizon:
        raise ValueError(
            f"Expected 1..{max_horizon} action steps, got {len(actions)}"
        )
    return actions


def extract_robot_actions(
    response: dict[str, Any],
    *,
    max_horizon: int,
) -> np.ndarray:
    """Extract a policy response and convert it to robot joint order."""

    return openpi_chunk_to_robot_chunk(
        extract_openpi_actions(response, max_horizon=max_horizon)
    )


def select_action_window(
    actions: Any,
    *,
    skip: int,
    count: int,
) -> list[np.ndarray]:
    """Copy up to ``count`` actions after dropping a leading prefix."""

    chunk = _checked_joint_chunk(actions, name="robot action chunk")
    start = int(skip)
    window_size = int(count)
    if start < 0:
        raise ValueError(f"chunk skip must be non-negative, got {start}")
    if window_size <= 0:
        raise ValueError(f"action count must be positive, got {window_size}")
    if start >= len(chunk):
        raise ValueError(
            f"Cannot skip {start} actions from a chunk of length {len(chunk)}"
        )
    stop = min(len(chunk), start + window_size)
    return [action.copy() for action in chunk[start:stop]]


def process_robot_action(
    action: Any,
    current_action: Any,
    *,
    gripper_threshold: float,
    max_joint_delta: float | None,
) -> np.ndarray:
    """Limit arm-joint deltas and binarize the two robot-order grippers."""

    result = checked_joint_vector(action, name="robot action")
    current = checked_joint_vector(current_action, name="current robot action")

    if not math.isfinite(float(gripper_threshold)):
        raise ValueError("gripper_threshold must be finite")
    if max_joint_delta is not None:
        max_joint_delta = float(max_joint_delta)
        if not math.isfinite(max_joint_delta):
            raise ValueError("max_joint_delta must be finite or None")
        if max_joint_delta > 0:
            arm_indices = np.r_[0:7, 8:15]
            result[arm_indices] = np.clip(
                result[arm_indices],
                current[arm_indices] - max_joint_delta,
                current[arm_indices] + max_joint_delta,
            )

    result[ROBOT_LEFT_GRIPPER] = float(
        result[ROBOT_LEFT_GRIPPER] > gripper_threshold
    )
    result[ROBOT_RIGHT_GRIPPER] = float(
        result[ROBOT_RIGHT_GRIPPER] > gripper_threshold
    )
    return result.astype(np.float32, copy=False)


def fake_obs() -> dict[str, Any]:
    """Return a deterministic observation with the real runtime field shapes."""

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "image/head": image.copy(),
        "image/left_wrist": image.copy(),
        "image/right_wrist": image.copy(),
        "state/joint": np.zeros(JOINT_DIM, dtype=np.float32),
    }


@dataclasses.dataclass
class JointArgs:
    action_horizon: int

    task_name: str = "beijing_benchmark_setting"
    remote_host: str = "127.0.0.1"
    remote_port: int = 2379

    # 通过本地 SSH 端口转发访问远端策略服务时的超时与重试配置。
    # 两个 wait timeout 取 0 表示不设总上限；response timeout 取 0 表示
    # metadata 和单次推理接收不限时；inference_retries 默认 0，不自动重发。
    ws_open_timeout: float = 10.0
    ws_close_timeout: float = 10.0
    ws_response_timeout: float = 300.0
    ws_retry_initial_delay: float = 1.0
    ws_retry_max_delay: float = 10.0
    ws_server_wait_timeout: float = 0.0
    ws_reconnect_wait_timeout: float = 30.0
    ws_inference_retries: int = 0

    max_timesteps: int = 6000
    execution_horizon: int = 1
    chunk_skip: int = 0
    control_frequency: float = 30.0
    trajectory_time: float | None = None

    resize_images: bool = True
    image_size: int = 224

    gripper_threshold: float = 0.6
    max_joint_delta: float | None = 0.15
    use_smooth_trajectory: bool = False

    prompt: str | None = None
    prompt_key: str | None = None

    use_fake_obs: bool = False
    debug: bool = False


def validate_joint_args(args: JointArgs) -> None:
    """Validate joint-policy configuration without resolving prompt contents."""

    validate_websocket_args(args)
    if args.action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if args.max_timesteps <= 0:
        raise ValueError("max_timesteps must be positive")
    if not math.isfinite(args.control_frequency) or args.control_frequency <= 0:
        raise ValueError("control_frequency must be finite and positive")
    if not 1 <= args.execution_horizon <= args.action_horizon:
        raise ValueError(
            f"execution_horizon must be in [1, {args.action_horizon}], "
            f"got {args.execution_horizon}"
        )
    if not 0 <= args.chunk_skip < args.action_horizon:
        raise ValueError(
            f"chunk_skip must be in [0, {args.action_horizon - 1}], "
            f"got {args.chunk_skip}"
        )
    if args.image_size <= 0:
        raise ValueError("image_size must be positive")
    if args.trajectory_time is not None and (
        not math.isfinite(args.trajectory_time) or args.trajectory_time <= 0
    ):
        raise ValueError("trajectory_time must be finite and positive")
    if not 0.0 <= args.gripper_threshold <= 1.0 or not math.isfinite(
        args.gripper_threshold
    ):
        raise ValueError("gripper_threshold must be finite and within [0, 1]")
    if args.max_joint_delta is not None and (
        not math.isfinite(args.max_joint_delta) or args.max_joint_delta <= 0
    ):
        raise ValueError("max_joint_delta must be finite and positive or None")
    if not 1 <= args.remote_port <= 65535:
        raise ValueError("remote_port must be within [1, 65535]")
    if not args.remote_host.strip():
        raise ValueError("remote_host must not be empty")
    if not args.task_name.strip():
        raise ValueError("task_name must not be empty")
    if args.prompt is not None and not args.prompt.strip():
        raise ValueError("prompt must not be empty when provided")
    if args.prompt_key is not None and not args.prompt_key.strip():
        raise ValueError("prompt_key must not be empty when provided")
