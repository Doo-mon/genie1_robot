"""同步 EE 重采样队列的异步预取协议与后台工作器。

本模块只服务于独立的 ``inference_pi05_g1_desk_ee_prefetch.py`` 入口；原同步
入口和 RTC 入口均不导入它。前台始终完整执行当前重采样队列，队列尾部尚未
下发的 EE14 command 以 0--1 夹爪空间作为服务端 prefix，后台只缓存响应中
prefix 之后的未来动作。它不会在当前 chunk 中途替换动作，也不会按推理耗时
丢弃当前队列。若后台响应未及时完成，前台在 chunk 边界等待而不接入旧响应。

后台线程只负责一次网络 infer；机器人观测由控制线程在启动预取前获取，确保
观测时刻和 prefix 的第一条未执行命令对齐。所有数组进入线程前都会复制，避免
执行队列继续 pop 时改变请求。退出时调用 ``cancel_pending_inference`` 可打断
阻塞的 WebSocket 接收，线程使用 daemon 仅作为最后的进程退出保护。
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any

import numpy as np

import policy_ee


@dataclasses.dataclass
class EEPrefetchArgs(policy_ee.EEArgs):
    """独立同步预取入口参数；不会被 RTC 参数类继承。"""

    rollout_output_dir: str | None = None
    rollout_fps: float = 30.0
    rollout_enabled: bool = False
    max_retimed_chunk_steps: int = 500
    freeze_arm: str = "none"
    # 当当前重采样队列缩短到该长度时，整段剩余 command 作为 prefix。
    # 必须小于 action_horizon，保证服务端响应至少留下一个未来 action。
    prefetch_prefix_len: int = 24
    prefetch_wait_poll: float = 0.10


@dataclasses.dataclass(frozen=True)
class PrefetchedEEWindow:
    """服务端响应中去除已承诺 prefix 后的原始/冻结未来动作。"""

    raw_actions: np.ndarray
    effective_actions: np.ndarray
    prefix_len: int
    infer_ms: float


def validate_prefetch_args(args: EEPrefetchArgs) -> str:
    """校验预取专用参数并返回规范化后的冻结手臂名称。"""

    policy_ee.validate_ee_args(args)
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
    if (
        not np.isfinite(args.prefetch_wait_poll)
        or args.prefetch_wait_poll <= 0
    ):
        raise ValueError("prefetch_wait_poll must be finite and positive")

    freeze_arm = str(args.freeze_arm).strip().casefold()
    if freeze_arm not in {"none", "left", "right"}:
        raise ValueError("freeze_arm must be one of: none, left, right")
    if args.freeze_left_arm:
        if freeze_arm == "right":
            raise ValueError(
                "freeze_left_arm conflicts with freeze_arm='right'; use only "
                "freeze_arm for the single-arm test"
            )
        freeze_arm = "left"
    return freeze_arm


def _checked_prefix(value: Any, *, action_horizon: int) -> np.ndarray:
    prefix = np.asarray(value, dtype=np.float32)
    if prefix.ndim != 2 or prefix.shape[1] != policy_ee.EE_DIM:
        raise ValueError(
            f"Expected prefetch prefix shape (P, {policy_ee.EE_DIM}), "
            f"got {prefix.shape}"
        )
    if not 1 <= len(prefix) < action_horizon:
        raise ValueError(
            f"Prefetch prefix length must be in [1, {action_horizon - 1}], "
            f"got {len(prefix)}"
        )
    if not np.all(np.isfinite(prefix)):
        raise ValueError("Prefetch prefix contains NaN or Inf")
    # The server action contract is EE14 with 0=open and 1=closed.  Sensor
    # feedback's 0--120 gripper values must never enter this request field.
    for index in (policy_ee.LEFT_GRIPPER, policy_ee.RIGHT_GRIPPER):
        values = prefix[:, index]
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(
                "Prefetch gripper prefix must stay in action space [0, 1]; "
                f"index {index} has range [{values.min()}, {values.max()}]"
            )
    return prefix.copy()


class AsyncEEPrefetch:
    """管理至多一个后台 EE prefix infer，并安全交付未来动作尾部。"""

    def __init__(
        self,
        *,
        policy: Any,
        args: EEPrefetchArgs,
        prompt: str,
        freeze_arm: str,
        frozen_arm_target: np.ndarray,
    ) -> None:
        self._policy = policy
        self._args = args
        self._prompt = str(prompt)
        self._freeze_arm = freeze_arm
        self._frozen_arm_target = policy_ee.checked_vector(
            frozen_arm_target,
            dim=policy_ee.EE_DIM,
            name="prefetch frozen-arm target",
        )
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: PrefetchedEEWindow | None = None
        self._error: BaseException | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._thread is not None

    @property
    def ready(self) -> bool:
        return self._done.is_set()

    def start(self, obs: dict[str, Any], remaining_commands: Any) -> int:
        """从与 prefix 同一控制时刻的观测启动一次后台请求。"""

        prefix = _checked_prefix(
            remaining_commands,
            action_horizon=self._args.action_horizon,
        )
        if len(prefix) > self._args.prefetch_prefix_len:
            raise ValueError(
                f"Prefetch prefix has {len(prefix)} commands, exceeding "
                f"configured prefetch_prefix_len={self._args.prefetch_prefix_len}"
            )
        # Keep request image resizing/packing off the fixed-rate control
        # thread. RobotEnv returns a fresh observation mapping; copy all array
        # values here so the worker owns an immutable snapshot while execution
        # continues to pop the separate command queue.
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
                raise RuntimeError("An EE prefetch request is already active")
            self._result = None
            self._error = None
            self._done.clear()
            thread = threading.Thread(
                target=self._run,
                args=(obs_snapshot, prefix, len(prefix)),
                name="ee-policy-prefetch",
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
            request = policy_ee.prepare_policy_request(
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
            actions = policy_ee.extract_ee_actions(
                response,
                max_horizon=self._args.action_horizon,
            )
            if len(actions) != self._args.action_horizon:
                raise ValueError(
                    f"Server returned horizon {len(actions)}, but prefetch "
                    f"expects exactly {self._args.action_horizon}"
                )
            raw_tail = actions[prefix_len:].copy()
            if len(raw_tail) == 0:
                raise ValueError("Prefetch response has no future actions after prefix")
            effective_tail = raw_tail.copy()
            if self._freeze_arm != "none":
                effective_tail = policy_ee.freeze_ee_arm(
                    effective_tail,
                    self._frozen_arm_target,
                    arm=self._freeze_arm,
                )
            result = PrefetchedEEWindow(
                raw_actions=raw_tail,
                effective_actions=effective_tail,
                prefix_len=prefix_len,
                infer_ms=infer_ms,
            )
            with self._lock:
                self._result = result
        except BaseException as error:
            with self._lock:
                self._error = error
        finally:
            self._done.set()

    def take(self) -> PrefetchedEEWindow:
        """等待并取走一次结果；等待期间保持 Ctrl-C 可响应。"""

        while not self._done.wait(timeout=self._args.prefetch_wait_poll):
            pass
        with self._lock:
            thread = self._thread
            error = self._error
            result = self._result
            self._thread = None
            self._error = None
            self._result = None
            self._done.clear()
        if thread is not None:
            thread.join(timeout=0.0)
        if error is not None:
            raise RuntimeError("EE asynchronous prefetch failed") from error
        if result is None:
            raise RuntimeError("EE asynchronous prefetch completed without a result")
        return result

    def cancel(self, *, join_timeout: float = 5.0) -> None:
        """取消尚未完成的网络等待；完成结果可直接丢弃。"""

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
            raise RuntimeError("EE prefetch worker did not stop after cancellation")
        with self._lock:
            self._thread = None
            self._result = None
            self._error = None
            self._done.clear()


__all__ = [
    "AsyncEEPrefetch",
    "EEPrefetchArgs",
    "PrefetchedEEWindow",
    "validate_prefetch_args",
]
