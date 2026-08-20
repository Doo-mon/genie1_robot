"""末端执行器（EE）策略的数据合同、参数配置与纯数组变换。

请求状态固定为 ``[joint16, ee14]`` 共 30 维，响应动作固定为 14 维：
``[L_xyz3, L_axis_angle3, L_hand, R_xyz3, R_axis_angle3, R_hand]``。
观测中的夹爪 state 保留 0--120 的编码器位置，模型 action 使用连续 0--1；
当前训练数据约定 0=张开、1=闭合，且两者数值同方向变化。二者虽然占据
相同索引，但属于不同数值空间，RTC 只能在网络边界进行显式线性映射。
本模块负责图像打包、形状与有限值校验、动作窗口、左臂冻结、单步限幅，
以及同步 EE 入口使用的 chunk 时间重采样；重采样不会被 RTC 自动启用。
``EEArgs`` 还集中声明本地转发的 WS 超时/重试参数并调用轻量配置校验。
它仅依赖 NumPy 和 ``inference_common`` 纯工具，不加载机器人、WebSocket、CLI 或录制后端。
同步 EE 入口、``policy_ee_rtc`` 与 ``rollout_schemas`` 共同调用这里的公开接口。
安全边界要求动作维度严格匹配，并在下发前限制位置/旋转增量及按需二值化夹爪。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from inference_common import resize_image, validate_websocket_args


JOINT_DIM = 16
EE_DIM = 14
WIRE_STATE_DIM = JOINT_DIM + EE_DIM

LEFT_POSITION = slice(0, 3)
LEFT_ROTATION = slice(3, 6)
LEFT_GRIPPER = 6
LEFT_EE = slice(0, 7)
RIGHT_POSITION = slice(7, 10)
RIGHT_ROTATION = slice(10, 13)
RIGHT_GRIPPER = 13
RIGHT_EE = slice(7, 14)

GRIPPER_ACTION_MIN = 0.0
GRIPPER_ACTION_MAX = 1.0
GRIPPER_STATE_MIN = 0.0
GRIPPER_STATE_MAX = 120.0


@dataclasses.dataclass(frozen=True)
class RetimedEEChunk:
    """A variable-length executable EE trajectory and its source-row map."""

    commands: np.ndarray
    source_indices: np.ndarray
    segment_steps: np.ndarray


def checked_vector(value: Any, *, dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (dim,):
        raise ValueError(f"Expected {name} shape ({dim},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf: {array}")
    return array.copy()


def build_wire_state(obs: dict[str, Any]) -> np.ndarray:
    """Build the raw 30-D state expected before the server-side EE slice."""
    joint = checked_vector(obs["state/joint"], dim=JOINT_DIM, name="joint state")
    ee = checked_vector(obs["state/pose"], dim=EE_DIM, name="EE pose state")
    state = np.concatenate([joint, ee]).astype(np.float32, copy=False)
    if state.shape != (WIRE_STATE_DIM,):
        raise AssertionError(f"Internal state packing error: {state.shape}")
    return state


def prepare_policy_request(
    obs: dict[str, Any],
    prompt: str,
    *,
    resize_images: bool,
    image_size: int,
) -> dict[str, Any]:
    required = (
        "image/head",
        "image/left_wrist",
        "image/right_wrist",
        "state/joint",
        "state/pose",
    )
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(
            f"RobotEnv observation is missing {missing}; available={sorted(obs.keys())}. "
            "Use RobotEnv(control_mode='camera_position')."
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
        "observation/state": build_wire_state(obs),
        "prompt": str(prompt),
    }


def extract_ee_actions(
    response: dict[str, Any],
    *,
    max_horizon: int,
) -> np.ndarray:
    if "actions" not in response:
        raise KeyError(f"Policy response has no 'actions'; keys={sorted(response.keys())}")

    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError(f"Expected action batch size 1, got {actions.shape}")
        actions = actions[0]
    elif actions.ndim == 1:
        actions = actions[None, :]

    if actions.ndim != 2 or actions.shape[-1] != EE_DIM:
        raise ValueError(f"Expected EE actions shape (H, {EE_DIM}), got {actions.shape}")
    if max_horizon <= 0:
        raise ValueError(f"max_horizon must be positive, got {max_horizon}")
    if not 1 <= len(actions) <= max_horizon:
        raise ValueError(
            f"Expected 1..{max_horizon} action steps, got {len(actions)}"
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy actions contain NaN or Inf")
    return actions


def select_action_window(
    actions: np.ndarray,
    *,
    skip: int,
    count: int,
) -> list[np.ndarray]:
    """Drop the leading chunk prefix and copy the requested execution window."""
    start = int(skip)
    if start < 0:
        raise ValueError(f"chunk skip must be non-negative, got {start}")
    if start >= len(actions):
        raise ValueError(
            f"Cannot skip {start} actions from a chunk of length {len(actions)}"
        )
    stop = min(len(actions), start + int(count))
    return [np.asarray(action, dtype=np.float32).copy() for action in actions[start:stop]]


def freeze_ee_arm(
    actions: np.ndarray,
    initial_ee: np.ndarray,
    *,
    arm: str,
) -> np.ndarray:
    """Freeze one complete EE7 target at its episode-start command.

    ``arm="left"`` freezes left pose and gripper while the right EE remains
    untouched; ``arm="right"`` does the inverse.  The operation happens in
    action space, so the gripper anchor is already normalized to 0--1.
    """

    result = np.asarray(actions, dtype=np.float32).copy()
    if result.ndim not in (1, 2) or result.shape[-1] != EE_DIM:
        raise ValueError(f"Expected EE action shape (14,) or (H, 14), got {result.shape}")
    normalized_arm = str(arm).strip().casefold()
    if normalized_arm == "left":
        frozen_slice = LEFT_EE
    elif normalized_arm == "right":
        frozen_slice = RIGHT_EE
    else:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    anchor = checked_vector(
        initial_ee,
        dim=EE_DIM,
        name=f"{normalized_arm}-arm freeze anchor",
    )
    result[..., frozen_slice] = anchor[frozen_slice]
    return result


def freeze_left_ee(actions: np.ndarray, initial_ee: np.ndarray) -> np.ndarray:
    """Backward-compatible left-EE freeze used by the existing RTC path."""

    return freeze_ee_arm(actions, initial_ee, arm="left")


def state_ee_to_action_command(state_ee: np.ndarray) -> np.ndarray:
    """Convert one EE feedback state into the local EE action space.

    Pose components already share the same units.  Only the two gripper
    components differ: feedback uses the 0--120 encoder range while policy
    actions and robot commands use 0--1.  The numeric direction is preserved:
    0 means open and 1/120 means closed.
    """

    result = checked_vector(
        state_ee,
        dim=EE_DIM,
        name="EE state used as command baseline",
    )
    state_span = GRIPPER_STATE_MAX - GRIPPER_STATE_MIN
    for index in (LEFT_GRIPPER, RIGHT_GRIPPER):
        state_value = float(result[index])
        if not GRIPPER_STATE_MIN <= state_value <= GRIPPER_STATE_MAX:
            raise ValueError(
                f"EE gripper state at index {index} must be within "
                f"[{GRIPPER_STATE_MIN}, {GRIPPER_STATE_MAX}], got {state_value}"
            )
        result[index] = (state_value - GRIPPER_STATE_MIN) / state_span
    return result


def _limit_vector_delta(
    target: np.ndarray,
    current: np.ndarray,
    max_norm: float | None,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    current = np.asarray(current, dtype=np.float32)
    if max_norm is None or max_norm <= 0:
        return target.copy()
    delta = target - current
    norm = float(np.linalg.norm(delta))
    if norm <= float(max_norm) or norm < 1e-8:
        return target.copy()
    return current + delta * (float(max_norm) / norm)


def _limit_rotation_delta(
    target: np.ndarray,
    current: np.ndarray,
    max_angle: float | None,
) -> np.ndarray:
    """Limit one rotation-vector command along the shortest SO(3) path.

    Rotation vectors are not ordinary Euclidean coordinates near the ``pi``
    representation boundary.  Interpolating their three components directly
    can turn a physically small orientation change into an almost full turn.
    Use the same quaternion/SO(3) metric as retiming and physical arrival
    checks, while preserving the policy's exact endpoint representation once
    it is within the configured angular step.
    """

    target = np.asarray(target, dtype=np.float32).reshape(3)
    current = np.asarray(current, dtype=np.float32).reshape(3)
    if max_angle is None or max_angle <= 0:
        return target.copy()
    distance = _rotation_distance(current, target)
    if distance <= float(max_angle) or distance < 1e-8:
        return target.copy()
    return _slerp_rotation_vector(current, target, float(max_angle) / distance)


def process_ee_action(
    action: np.ndarray,
    previous_command: np.ndarray,
    *,
    max_position_delta: float | None = 0.004,
    max_rotation_delta: float | None = 0.02,
    gripper_threshold: float | None = None,
    gripper_close_threshold: float = 0.90,
    gripper_open_threshold: float = 0.10,
    binarize_gripper: bool,
) -> np.ndarray:
    """Apply EE limits and optionally binarize grippers with hysteresis.

    Gripper action polarity is ``0=open`` and ``1=closed``.  When binary mode
    is enabled, an open gripper closes only after the target reaches
    ``gripper_close_threshold``; a closed gripper opens only after the target
    falls to ``gripper_open_threshold``.  Values between those thresholds keep
    the previous binary command, avoiding premature release and threshold
    chatter.  ``gripper_threshold`` is the legacy single-threshold override.
    """
    result = checked_vector(action, dim=EE_DIM, name="EE action")
    previous = checked_vector(
        previous_command, dim=EE_DIM, name="previous EE command"
    )

    for part in (LEFT_POSITION, RIGHT_POSITION):
        result[part] = _limit_vector_delta(
            result[part], previous[part], max_position_delta
        )
    for part in (LEFT_ROTATION, RIGHT_ROTATION):
        result[part] = _limit_rotation_delta(
            result[part], previous[part], max_rotation_delta
        )

    close_threshold, open_threshold = resolve_gripper_thresholds(
        gripper_threshold=gripper_threshold,
        gripper_close_threshold=gripper_close_threshold,
        gripper_open_threshold=gripper_open_threshold,
    )
    for index in (LEFT_GRIPPER, RIGHT_GRIPPER):
        value = float(
            np.clip(result[index], GRIPPER_ACTION_MIN, GRIPPER_ACTION_MAX)
        )
        if not binarize_gripper:
            result[index] = value
            continue

        if gripper_threshold is not None:
            # Preserve the old one-threshold behavior exactly.  In particular,
            # a value equal to the threshold remains open and cannot oscillate.
            result[index] = float(value > close_threshold)
            continue

        was_closed = float(previous[index]) >= 0.5
        if was_closed:
            result[index] = 0.0 if value <= open_threshold else 1.0
        else:
            result[index] = 1.0 if value >= close_threshold else 0.0
    return result.astype(np.float32, copy=False)


def resolve_gripper_thresholds(
    *,
    gripper_threshold: float | None,
    gripper_close_threshold: float,
    gripper_open_threshold: float,
) -> tuple[float, float]:
    """Resolve legacy/split gripper thresholds and validate their ordering."""

    if gripper_threshold is not None:
        close_threshold = open_threshold = float(gripper_threshold)
    else:
        close_threshold = float(gripper_close_threshold)
        open_threshold = float(gripper_open_threshold)

    if not np.isfinite(close_threshold) or not np.isfinite(open_threshold):
        raise ValueError("gripper thresholds must be finite")
    if gripper_threshold is not None:
        if not 0.0 <= close_threshold <= 1.0:
            raise ValueError("gripper_threshold must be within [0, 1]")
    elif not 0.0 <= open_threshold < close_threshold <= 1.0:
        raise ValueError(
            "gripper thresholds must satisfy "
            "0 <= gripper_open_threshold < gripper_close_threshold <= 1; "
            f"got open={open_threshold}, close={close_threshold}"
        )
    return close_threshold, open_threshold


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    """Convert one rotation vector to a normalized ``[w, x, y, z]`` quaternion."""

    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        quaternion = np.concatenate(([1.0], 0.5 * vector))
    else:
        half = 0.5 * angle
        quaternion = np.concatenate(
            ([np.cos(half)], vector * (np.sin(half) / angle))
        )
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    """Convert a quaternion to the principal rotation-vector representation."""

    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Cannot convert a zero or non-finite quaternion")
    value = value / norm
    # q and -q encode the same rotation.  The positive scalar hemisphere gives
    # the principal rotation (angle <= pi) and a stable shortest-path SLERP.
    if value[0] < 0.0:
        value = -value
    sine_half = float(np.linalg.norm(value[1:]))
    if sine_half < 1e-12:
        return (2.0 * value[1:]).astype(np.float32)
    angle = 2.0 * np.arctan2(sine_half, float(value[0]))
    return (value[1:] * (angle / sine_half)).astype(np.float32)


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest SO(3) angle between two rotation vectors."""

    first_q = _rotvec_to_quaternion(first)
    second_q = _rotvec_to_quaternion(second)
    cosine_half = float(np.clip(abs(np.dot(first_q, second_q)), 0.0, 1.0))
    return float(2.0 * np.arccos(cosine_half))


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Public shortest SO(3) angle used by feedback-based RTC arrival checks."""

    return _rotation_distance(first, second)


def _slerp_rotation_vector(
    first: np.ndarray,
    second: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Interpolate two rotation vectors along the shortest SO(3) path."""

    alpha = float(fraction)
    if alpha <= 0.0:
        return np.asarray(first, dtype=np.float32).copy()
    if alpha >= 1.0:
        # Preserve the checkpoint's exact endpoint representation.
        return np.asarray(second, dtype=np.float32).copy()

    first_q = _rotvec_to_quaternion(first)
    second_q = _rotvec_to_quaternion(second)
    dot = float(np.dot(first_q, second_q))
    if dot < 0.0:
        second_q = -second_q
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        result = first_q + alpha * (second_q - first_q)
        result /= np.linalg.norm(result)
    else:
        angle = float(np.arccos(dot))
        sine = float(np.sin(angle))
        result = (
            np.sin((1.0 - alpha) * angle) / sine * first_q
            + np.sin(alpha * angle) / sine * second_q
        )
    return _quaternion_to_rotvec(result)


def _required_retime_steps(
    previous: np.ndarray,
    target: np.ndarray,
    *,
    max_position_delta: float | None,
    max_rotation_delta: float | None,
) -> int:
    steps = 1
    if max_position_delta is not None:
        for part in (LEFT_POSITION, RIGHT_POSITION):
            ratio = float(np.linalg.norm(target[part] - previous[part])) / float(
                max_position_delta
            )
            steps = max(steps, int(np.ceil(max(0.0, ratio))))
    if max_rotation_delta is not None:
        for part in (LEFT_ROTATION, RIGHT_ROTATION):
            ratio = _rotation_distance(previous[part], target[part]) / float(
                max_rotation_delta
            )
            steps = max(steps, int(np.ceil(max(0.0, ratio))))
    return steps


def retime_ee_action_window(
    actions: np.ndarray,
    initial_command: np.ndarray,
    *,
    max_position_delta: float | None,
    max_rotation_delta: float | None,
    binarize_gripper: bool,
    gripper_threshold: float | None,
    gripper_close_threshold: float,
    gripper_open_threshold: float,
    max_output_steps: int,
) -> RetimedEEChunk:
    """Expand an absolute EE action window into a rate-constrained trajectory.

    Each source-row segment is time-scaled just enough for both arms to satisfy
    the position and SO(3) rotation limits.  Positions use linear interpolation;
    rotations use shortest-path quaternion SLERP.  With binary grippers, each
    gripper keeps its previous latch during inserted substeps and changes only
    on the source segment's final substep, so arm retiming cannot make grasp or
    release happen early.  Continuous grippers use the same endpoint timing:
    inserted substeps keep the previous value and the target is applied only
    when the arm segment reaches its endpoint.

    ``source_indices`` maps every executable command back to the corresponding
    input row for action logging.  Exceeding ``max_output_steps`` rejects the
    complete chunk instead of silently truncating an endpoint.
    """

    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != EE_DIM:
        raise ValueError(f"Expected EE action window shape (H, {EE_DIM}), got {chunk.shape}")
    if not np.all(np.isfinite(chunk)):
        raise ValueError("EE action window contains NaN or Inf")
    if (
        isinstance(max_output_steps, bool)
        or not isinstance(max_output_steps, (int, np.integer))
        or max_output_steps <= 0
    ):
        raise ValueError("max_output_steps must be a positive integer")
    max_output_steps = int(max_output_steps)
    for name, value in (
        ("max_position_delta", max_position_delta),
        ("max_rotation_delta", max_rotation_delta),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive or None")

    previous = checked_vector(
        initial_command,
        dim=EE_DIM,
        name="EE retiming initial command",
    )
    if binarize_gripper:
        # The first baseline may come from a 0..120 sensor reading normalized
        # to a continuous 0..1 value.  Retimed intermediate commands bypass the
        # old per-tick processor, so establish the same binary latch explicitly
        # before any inserted substep can be sent.
        previous[LEFT_GRIPPER] = float(previous[LEFT_GRIPPER] >= 0.5)
        previous[RIGHT_GRIPPER] = float(previous[RIGHT_GRIPPER] >= 0.5)
    if len(chunk) == 0:
        return RetimedEEChunk(
            commands=np.empty((0, EE_DIM), dtype=np.float32),
            source_indices=np.empty((0,), dtype=np.int64),
            segment_steps=np.empty((0,), dtype=np.int64),
        )

    commands: list[np.ndarray] = []
    source_indices: list[int] = []
    segment_steps: list[int] = []
    for source_index, raw_target in enumerate(chunk):
        # Reuse the established gripper hysteresis/continuous clipping exactly,
        # but do not apply the old pose limiter: interpolation below owns the
        # physical timeline and reaches this absolute endpoint exactly.
        endpoint = process_ee_action(
            raw_target,
            previous,
            max_position_delta=None,
            max_rotation_delta=None,
            binarize_gripper=binarize_gripper,
            gripper_threshold=gripper_threshold,
            gripper_close_threshold=gripper_close_threshold,
            gripper_open_threshold=gripper_open_threshold,
        )
        count = _required_retime_steps(
            previous,
            endpoint,
            max_position_delta=max_position_delta,
            max_rotation_delta=max_rotation_delta,
        )
        if len(commands) + count > max_output_steps:
            raise ValueError(
                "Retimed EE chunk would exceed max_output_steps="
                f"{max_output_steps} at source row {source_index}: "
                f"needed at least {len(commands) + count} commands. Increase "
                "the cap, reduce execution_horizon, or relax the per-tick limits."
            )
        start = previous.copy()
        for substep in range(1, count + 1):
            alpha = substep / count
            command = start + alpha * (endpoint - start)
            for rotation in (LEFT_ROTATION, RIGHT_ROTATION):
                command[rotation] = _slerp_rotation_vector(
                    start[rotation], endpoint[rotation], alpha
                )
            if substep < count:
                command[LEFT_GRIPPER] = start[LEFT_GRIPPER]
                command[RIGHT_GRIPPER] = start[RIGHT_GRIPPER]
            if substep == count:
                command = endpoint.copy()
            commands.append(command.astype(np.float32, copy=False))
            source_indices.append(source_index)
        segment_steps.append(count)
        previous = endpoint

    result = np.stack(commands, axis=0).astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError("Retimed EE commands contain NaN or Inf")
    # Defense in depth: validate the generated physical timeline itself, not
    # only the subdivision count.  This check uses the same SO(3) metric as the
    # planner and runs before the caller can issue the first hardware command.
    check_previous = checked_vector(
        initial_command,
        dim=EE_DIM,
        name="EE retiming validation baseline",
    )
    for command_index, command in enumerate(result):
        if max_position_delta is not None:
            tolerance = max(1e-7, float(max_position_delta) * 1e-5)
            for part in (LEFT_POSITION, RIGHT_POSITION):
                distance = float(np.linalg.norm(command[part] - check_previous[part]))
                if distance > float(max_position_delta) + tolerance:
                    raise AssertionError(
                        "Internal EE retiming position-limit error at command "
                        f"{command_index}: {distance} > {max_position_delta}"
                    )
        if max_rotation_delta is not None:
            tolerance = max(1e-7, float(max_rotation_delta) * 1e-5)
            for part in (LEFT_ROTATION, RIGHT_ROTATION):
                distance = _rotation_distance(
                    check_previous[part], command[part]
                )
                if distance > float(max_rotation_delta) + tolerance:
                    raise AssertionError(
                        "Internal EE retiming rotation-limit error at command "
                        f"{command_index}: {distance} > {max_rotation_delta}"
                    )
        check_previous = command
    return RetimedEEChunk(
        commands=result,
        source_indices=np.asarray(source_indices, dtype=np.int64),
        segment_steps=np.asarray(segment_steps, dtype=np.int64),
    )


def fake_obs() -> dict[str, Any]:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "image/head": image.copy(),
        "image/left_wrist": image.copy(),
        "image/right_wrist": image.copy(),
        "state/joint": np.zeros(JOINT_DIM, dtype=np.float32),
        "state/pose": np.zeros(EE_DIM, dtype=np.float32),
    }


@dataclasses.dataclass
class EEArgs:
    action_horizon: int
    task_name: str = "perdon"
    remote_host: str = "127.0.0.1"
    remote_port: int = 2379

    # WebSocket/SSH 端口转发配置。server_wait_timeout=0 表示启动时持续等待
    # 远端模型就绪；response_timeout=0 表示 metadata/推理响应不限时；
    # inference_retries=0 表示已经发送的推理请求不自动重发。
    ws_open_timeout: float = 10.0
    ws_close_timeout: float = 10.0
    ws_response_timeout: float = 300.0
    ws_retry_initial_delay: float = 1.0
    ws_retry_max_delay: float = 10.0
    ws_server_wait_timeout: float = 0.0
    ws_reconnect_wait_timeout: float = 30.0
    ws_inference_retries: int = 0

    max_timesteps: int = 2000
    execution_horizon: int = 1
    chunk_skip: int = 3
    control_frequency: float = 20.0
    trajectory_time: float | None = None

    resize_images: bool = True
    image_size: int = 224

    # Conservative per-command limits. At 20 Hz these defaults cap the
    # commanded translation/rotation rates at about 0.2 m/s and 1 rad/s.
    max_position_delta: float | None = 0.01
    max_rotation_delta: float | None = 0.05
    # Backward-compatible common hard floor in the robot base_link ABS_POSE
    # frame.  Per-arm values below override it independently.  All floors are
    # checked after camera-frame retargeting and immediately before the SDK.
    min_ee_base_z: float | None = None
    min_left_ee_base_z: float | None = None
    min_right_ee_base_z: float | None = None
    binarize_gripper: bool = False
    # 夹爪 action 约定 0=张开、1=闭合。二值模式从开到闭必须达到较高
    # close 阈值，从闭到开必须降到较低 open 阈值，中间带保持上次命令。
    gripper_close_threshold: float = 0.90
    gripper_open_threshold: float = 0.10
    # Backward-compatible single-threshold override.  Leave unset to use the
    # two hysteresis thresholds above.
    gripper_threshold: float | None = None
    freeze_left_arm: bool = False

    prompt: str | None = None
    prompt_key: str | None = None

    use_fake_obs: bool = False
    debug: bool = False


def validate_ee_args(args: EEArgs) -> None:
    validate_websocket_args(args)
    if not np.isfinite(args.control_frequency) or args.control_frequency <= 0:
        raise ValueError("control_frequency must be finite and positive")
    if args.max_timesteps <= 0:
        raise ValueError("max_timesteps must be positive")
    if args.action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
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
        not np.isfinite(args.trajectory_time) or args.trajectory_time <= 0
    ):
        raise ValueError("trajectory_time must be finite and positive")
    for name in ("max_position_delta", "max_rotation_delta"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive or None")
    for name in (
        "min_ee_base_z",
        "min_left_ee_base_z",
        "min_right_ee_base_z",
    ):
        value = getattr(args, name)
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{name} must be finite or None")
    resolve_gripper_thresholds(
        gripper_threshold=args.gripper_threshold,
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
    )
    if args.prompt is not None and not args.prompt.strip():
        raise ValueError("prompt must not be empty when provided")
    if args.prompt_key is not None and not args.prompt_key.strip():
        raise ValueError("prompt_key must not be empty when provided")
    if not args.task_name.strip():
        raise ValueError("task_name must not be empty")
