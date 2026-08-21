"""同步 joint 重采样队列的异步预取协议与纯数组轨迹生成。

机器人、模型、prefix 和执行命令统一采用16维
``[left_arm7, left_hand, right_arm7, right_hand]``，不做索引重排。每个模型
绝对关节 target 会按 ``max_joint_delta`` 线性展开，左右14个手臂关节共享同一
子步数；两个夹爪在该模型 row 的最后一个子步才切换。当前队列尾部的实际
command 作为服务端 physical prefix，响应前缀被丢弃，只有未来尾部交给下一
轮重采样。

本文件只由独立 joint prefetch 入口导入。原 joint 同步和 joint RTC 保持原状。
后台线程、观测快照、取消和错误传播与 EE prefetch 使用相同生命周期，但 joint
prefix 不涉及0--120反馈空间，夹爪始终是0=open、1=closed的动作端点。
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any

import numpy as np

import policy_joint


@dataclasses.dataclass
class JointPrefetchArgs(policy_joint.JointArgs):
    """独立 joint 重采样预取入口参数。"""

    rollout_output_dir: str | None = None
    rollout_fps: float = 30.0
    rollout_enabled: bool = False
    max_retimed_chunk_steps: int = 500
    prefetch_prefix_len: int = 18
    prefetch_wait_poll: float = 0.10


@dataclasses.dataclass(frozen=True)
class RetimedJointChunk:
    commands: np.ndarray
    source_indices: np.ndarray
    segment_steps: np.ndarray


@dataclasses.dataclass(frozen=True)
class PrefetchedJointWindow:
    raw_actions: np.ndarray
    prefix_len: int
    infer_ms: float


def validate_joint_prefetch_args(args: JointPrefetchArgs) -> None:
    policy_joint.validate_joint_args(args)
    if args.use_smooth_trajectory:
        raise ValueError(
            "joint prefetch requires direct fixed-rate commands; "
            "use_smooth_trajectory would block one tick for a complete SDK trajectory"
        )
    if not np.isfinite(args.rollout_fps) or args.rollout_fps <= 0:
        raise ValueError("rollout_fps must be finite and positive")
    if (
        isinstance(args.max_retimed_chunk_steps, bool)
        or not isinstance(args.max_retimed_chunk_steps, (int, np.integer))
        or args.max_retimed_chunk_steps <= 0
    ):
        raise ValueError("max_retimed_chunk_steps must be a positive integer")
    if (
        isinstance(args.prefetch_prefix_len, bool)
        or not isinstance(args.prefetch_prefix_len, (int, np.integer))
        or not 1 <= int(args.prefetch_prefix_len) < args.action_horizon
    ):
        raise ValueError(
            "prefetch_prefix_len must satisfy "
            f"1 <= value < action_horizon ({args.action_horizon})"
        )
    if not np.isfinite(args.prefetch_wait_poll) or args.prefetch_wait_poll <= 0:
        raise ValueError("prefetch_wait_poll must be finite and positive")


def retime_joint_action_window(
    actions: Any,
    initial_command: Any,
    *,
    max_joint_delta: float | None,
    gripper_threshold: float,
    max_output_steps: int,
) -> RetimedJointChunk:
    """把绝对 joint16 模型窗口展开为满足逐关节变化上限的命令序列。"""

    chunk = policy_joint.openpi_chunk_to_robot_chunk(actions)
    previous = policy_joint.checked_joint_vector(
        initial_command,
        name="joint retiming initial command",
    )
    # Robot feedback deployments have used both normalized 0--1 hand state
    # and encoder-distance 0--120 state, while commands are always 0--1.  The
    # ordinary sync path binarizes the first target before sending it; retiming
    # additionally needs a safe action-space latch for inserted substeps.
    for index in (
        policy_joint.ROBOT_LEFT_GRIPPER,
        policy_joint.ROBOT_RIGHT_GRIPPER,
    ):
        value = float(previous[index])
        if 0.0 <= value <= 1.0:
            normalized = value
        elif 0.0 <= value <= 120.0:
            normalized = value / 120.0
        else:
            raise ValueError(
                f"joint gripper feedback at index {index} must be in "
                f"[0, 1] or [0, 120], got {value}"
            )
        previous[index] = float(normalized >= 0.5)
    if (
        isinstance(max_output_steps, bool)
        or not isinstance(max_output_steps, (int, np.integer))
        or max_output_steps <= 0
    ):
        raise ValueError("max_output_steps must be a positive integer")
    max_output_steps = int(max_output_steps)
    if max_joint_delta is not None and (
        not np.isfinite(max_joint_delta) or max_joint_delta <= 0
    ):
        raise ValueError("max_joint_delta must be finite and positive or None")

    arm_indices = np.r_[0:7, 8:15]
    commands: list[np.ndarray] = []
    source_indices: list[int] = []
    segment_steps: list[int] = []
    for source_index, raw_target in enumerate(chunk):
        endpoint = policy_joint.process_robot_action(
            raw_target,
            previous,
            gripper_threshold=gripper_threshold,
            max_joint_delta=None,
        )
        count = 1
        if max_joint_delta is not None:
            largest = float(
                np.max(np.abs(endpoint[arm_indices] - previous[arm_indices]))
            )
            count = max(1, int(np.ceil(largest / float(max_joint_delta))))
        if len(commands) + count > max_output_steps:
            raise ValueError(
                "Retimed joint chunk would exceed max_output_steps="
                f"{max_output_steps} at source row {source_index}: needed at "
                f"least {len(commands) + count} commands"
            )
        start = previous.copy()
        for substep in range(1, count + 1):
            alpha = substep / count
            command = start + alpha * (endpoint - start)
            if substep < count:
                command[policy_joint.ROBOT_LEFT_GRIPPER] = start[
                    policy_joint.ROBOT_LEFT_GRIPPER
                ]
                command[policy_joint.ROBOT_RIGHT_GRIPPER] = start[
                    policy_joint.ROBOT_RIGHT_GRIPPER
                ]
            else:
                command = endpoint.copy()
            commands.append(command.astype(np.float32, copy=False))
            source_indices.append(source_index)
        segment_steps.append(count)
        previous = endpoint

    return RetimedJointChunk(
        commands=np.stack(commands, axis=0),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        segment_steps=np.asarray(segment_steps, dtype=np.int64),
    )


def _checked_prefix(value: Any, *, action_horizon: int) -> np.ndarray:
    prefix = policy_joint.robot_chunk_to_openpi_chunk(value)
    if not 1 <= len(prefix) < action_horizon:
        raise ValueError(
            f"Joint prefetch prefix length must be in [1, {action_horizon - 1}], "
            f"got {len(prefix)}"
        )
    for index in (
        policy_joint.ROBOT_LEFT_GRIPPER,
        policy_joint.ROBOT_RIGHT_GRIPPER,
    ):
        values = prefix[:, index]
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(
                f"Joint prefix gripper index {index} must remain within [0, 1]"
            )
    return prefix


class AsyncJointPrefetch:
    """管理至多一个后台 joint16 prefix infer。"""

    def __init__(self, *, policy: Any, args: JointPrefetchArgs, prompt: str) -> None:
        self._policy = policy
        self._args = args
        self._prompt = str(prompt)
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: PrefetchedJointWindow | None = None
        self._error: BaseException | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._thread is not None

    @property
    def ready(self) -> bool:
        return self._done.is_set()

    def start(self, obs: dict[str, Any], remaining_commands: Any) -> int:
        prefix = _checked_prefix(
            remaining_commands,
            action_horizon=self._args.action_horizon,
        )
        if len(prefix) > self._args.prefetch_prefix_len:
            raise ValueError(
                f"Joint prefix has {len(prefix)} commands, exceeding "
                f"prefetch_prefix_len={self._args.prefetch_prefix_len}"
            )
        obs_snapshot = {
            str(key): (
                np.asarray(value).copy()
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in obs.items()
        }
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("A joint prefetch request is already active")
            self._result = None
            self._error = None
            self._done.clear()
            thread = threading.Thread(
                target=self._run,
                args=(obs_snapshot, prefix.copy(), len(prefix)),
                name="joint-policy-prefetch",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        return len(prefix)

    def _run(
        self,
        obs: dict[str, Any],
        prefix: np.ndarray,
        prefix_len: int,
    ) -> None:
        try:
            request = policy_joint.prepare_policy_request(
                obs,
                self._prompt,
                resize_images=self._args.resize_images,
                image_size=self._args.image_size,
            )
            request["remaining_action_chunk"] = prefix.copy()
            request["prefix_len"] = prefix_len
            start = time.perf_counter()
            response = self._policy.infer(request)
            infer_ms = (time.perf_counter() - start) * 1000.0
            actions = policy_joint.extract_robot_actions(
                response,
                max_horizon=self._args.action_horizon,
            )
            if len(actions) != self._args.action_horizon:
                raise ValueError(
                    f"Server returned horizon {len(actions)}, but joint "
                    f"prefetch expects exactly {self._args.action_horizon}"
                )
            raw_tail = actions[prefix_len:].copy()
            if len(raw_tail) == 0:
                raise ValueError("Joint prefetch response has no future tail")
            with self._lock:
                self._result = PrefetchedJointWindow(
                    raw_actions=raw_tail,
                    prefix_len=prefix_len,
                    infer_ms=infer_ms,
                )
        except BaseException as error:
            with self._lock:
                self._error = error
        finally:
            self._done.set()

    def take(self) -> PrefetchedJointWindow:
        while not self._done.wait(timeout=self._args.prefetch_wait_poll):
            pass
        with self._lock:
            thread = self._thread
            result = self._result
            error = self._error
            self._thread = None
            self._result = None
            self._error = None
            self._done.clear()
        if thread is not None:
            thread.join(timeout=0.0)
        if error is not None:
            raise RuntimeError("Joint asynchronous prefetch failed") from error
        if result is None:
            raise RuntimeError("Joint prefetch completed without a result")
        return result

    def cancel(self, *, join_timeout: float = 5.0) -> None:
        with self._lock:
            thread = self._thread
        if thread is None:
            return
        if thread.is_alive():
            cancel = getattr(self._policy, "cancel_pending_inference", None)
            if callable(cancel):
                cancel()
            thread.join(timeout=join_timeout)
        if thread.is_alive():
            raise RuntimeError("Joint prefetch worker did not stop after cancellation")
        with self._lock:
            self._thread = None
            self._result = None
            self._error = None
            self._done.clear()


__all__ = [
    "AsyncJointPrefetch",
    "JointPrefetchArgs",
    "PrefetchedJointWindow",
    "RetimedJointChunk",
    "retime_joint_action_window",
    "validate_joint_prefetch_args",
]
