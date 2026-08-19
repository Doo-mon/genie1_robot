"""推理输入与动作执行日志的轻量运行时。

每次 Python 推理进程都会创建或复用 Shell 指定的时间戳目录，并立即写入
``run.json``、``policy_io.jsonl`` 与 ``actions.jsonl``。策略请求只完整保存
数值状态、RTC prefix 和 prompt；三路图像仅记录形状与 dtype，避免日志被原始
图像淹没。策略响应的原始 action chunk 会逐请求保存，成功执行的每一步则额外
保存“原始服务端输出”“选中的有效模型目标”“成功提交给 SDK 的 command”和
“提交后即时读取的真实 feedback”，供离线逐维画图。command 不是到位反馈；
控制器是异步的，因此 feedback 也只是采样时刻的物理状态，不代表轨迹已完成。

所有 JSONL 记录均逐行 flush，因此正常结束、异常和 Ctrl-C 后已经写完的行仍可
读取；末尾偶发的不完整行由绘图器忽略。日志失败只会打印一次警告并停用日志，
不会在机器人已经收到命令后反过来中断控制线程。模块只依赖标准库与 NumPy，
不加载硬件、WebSocket、OpenCV、PyArrow 或绘图库。
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

import numpy as np


LOG_SCHEMA_VERSION = "action-log-v3"
LOG_DIR_ENV = "INFERENCE_ACTION_LOG_DIR"
LOG_ENABLED_ENV = "INFERENCE_ACTION_LOG_ENABLED"

EE_DIMENSION_NAMES = (
    "left_x",
    "left_y",
    "left_z",
    "left_axis_angle_x",
    "left_axis_angle_y",
    "left_axis_angle_z",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_axis_angle_x",
    "right_axis_angle_y",
    "right_axis_angle_z",
    "right_gripper",
)

JOINT_DIMENSION_NAMES = (
    *(f"left_joint_{index}" for index in range(1, 8)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(1, 8)),
    "right_gripper",
)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
    return cleaned or "inference"


def _default_log_directory(entrypoint: str) -> Path:
    workspace = Path(__file__).resolve().parent
    root = workspace / "logs"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    base = root / f"{stamp}_{_safe_name(entrypoint)}_pid{os.getpid()}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _json_compatible(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def _summarize_policy_value(key: str, value: Any) -> Any:
    """Keep numeric policy inputs exact while summarizing large image arrays."""

    try:
        array = np.asarray(value)
    except Exception:
        return _json_compatible(value)

    if array.ndim >= 2 and "image" in key.lower():
        return {
            "kind": "image_summary",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    if array.dtype.kind in "biufc" and array.size <= 100_000:
        return {
            "kind": "numeric_array",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "values": array.tolist(),
        }
    return _json_compatible(value)


class ActionLogSession:
    """Own one run directory and append crash-tolerant JSONL records."""

    def __init__(self, *, mode: str, entrypoint: str) -> None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"ee", "joint"}:
            raise ValueError(f"action log mode must be 'ee' or 'joint', got {mode!r}")

        supplied = os.environ.get(LOG_DIR_ENV, "").strip()
        if supplied:
            self.directory = Path(supplied).expanduser().resolve()
            self.directory.mkdir(parents=True, exist_ok=True)
        else:
            self.directory = _default_log_directory(entrypoint).resolve()

        self.mode = normalized_mode
        self.entrypoint = str(entrypoint)
        self.dimension_names = (
            EE_DIMENSION_NAMES if self.mode == "ee" else JOINT_DIMENSION_NAMES
        )
        self._started_wall = datetime.now().astimezone()
        self._started_mono = time.monotonic()
        self._lock = threading.Lock()
        self._episode_index = -1
        self._episode_action_start = 0
        self._episode_open = False
        self._request_index = 0
        self._action_count = 0
        self._closed = False
        self._disabled = False
        self._warning_printed = False
        self._control_period_s: float | None = None
        self._action_write_count = 0
        self._action_write_total_s = 0.0
        self._action_write_max_s = 0.0
        self._performance_warning_printed = False
        self._feedback_warning_printed = False
        enabled_value = os.environ.get(LOG_ENABLED_ENV, "true").strip().casefold()
        if enabled_value not in {"1", "true", "yes", "0", "false", "no"}:
            raise ValueError(f"{LOG_ENABLED_ENV} must be true or false")
        self._logging_enabled = enabled_value in {"1", "true", "yes"}
        self._metadata: dict[str, Any] = {
            "schema_version": LOG_SCHEMA_VERSION,
            "entrypoint": self.entrypoint,
            "mode": self.mode,
            "dimension_names": list(self.dimension_names),
            "action_pair_semantics": {
                "raw_model_action": (
                    "original server output row before freeze, RTC hard-lock, "
                    "alignment replacement, and command safety processing"
                ),
                "model_action": (
                    "effective policy target selected for this execution step; "
                    "RTC freeze/hard-lock/alignment may already be applied"
                ),
                "command_action": (
                    "safety-processed command accepted by env.step/SDK; this is "
                    "not measured physical motion or proof that the target was reached"
                ),
                "actual_action": (
                    "legacy alias of command_action retained for old plot readers"
                ),
                "feedback_action": (
                    "fresh SDK state sampled immediately after command submission, "
                    "transformed into the same 14D/16D space; an asynchronous "
                    "instantaneous sample, not command-completion feedback"
                ),
                "raw_server_actions": "policy_io.jsonl policy_response records",
            },
            "started_at": self._started_wall.isoformat(timespec="milliseconds"),
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "argv": list(sys.argv),
        }
        self._policy_file = None
        self._action_file = None
        try:
            if not self._logging_enabled:
                self._disabled = True
                print(
                    "[action-log] disabled by ACTION_LOGGING=false; "
                    "console.log remains enabled"
                )
                return
            self._policy_file = (self.directory / "policy_io.jsonl").open(
                "a", encoding="utf-8", buffering=1
            )
            self._action_file = (self.directory / "actions.jsonl").open(
                "a", encoding="utf-8", buffering=1
            )
            self._write_metadata_locked()
            self._append_policy_locked("run_started", {})
        except Exception as error:
            self._disable_locked(error)
        print(f"[action-log] run directory: {self.directory}")

    @property
    def action_count(self) -> int:
        with self._lock:
            return self._action_count

    def _elapsed(self) -> float:
        return time.monotonic() - self._started_mono

    def _base_record(self, event: str) -> dict[str, Any]:
        return {
            "schema_version": LOG_SCHEMA_VERSION,
            "event": event,
            "wall_time": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "elapsed_s": round(self._elapsed(), 6),
            "episode": self._episode_index,
        }

    def _disable_locked(self, error: BaseException) -> None:
        self._disabled = True
        if not self._warning_printed:
            self._warning_printed = True
            print(
                f"[action-log] warning: logging disabled after error: {error}",
                file=sys.stderr,
            )

    def _write_line_locked(self, file_object, record: dict[str, Any]) -> None:
        if self._disabled or file_object is None:
            return
        try:
            file_object.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            file_object.flush()
        except Exception as error:
            self._disable_locked(error)

    def _append_policy_locked(self, event: str, payload: dict[str, Any]) -> None:
        record = self._base_record(event)
        record.update(payload)
        self._write_line_locked(self._policy_file, record)

    def _write_metadata_locked(self) -> None:
        if self._disabled:
            return
        temporary = self.directory / f".run.json.tmp-{os.getpid()}"
        try:
            temporary.write_text(
                json.dumps(
                    self._metadata,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.directory / "run.json")
        except Exception as error:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass
            self._disable_locked(error)

    def update_config(self, args: Any) -> None:
        with self._lock:
            if self._closed:
                return
            frequency = getattr(args, "control_frequency", None)
            try:
                frequency = float(frequency)
            except (TypeError, ValueError):
                frequency = None
            if frequency is not None and np.isfinite(frequency) and frequency > 0:
                self._control_period_s = 1.0 / frequency
            self._metadata["config"] = _json_compatible(args)
            self._write_metadata_locked()

    def begin_episode(self, prompt: str) -> int:
        with self._lock:
            if self._closed:
                return self._episode_index
            if self._episode_open:
                self._append_policy_locked(
                    "episode_ended",
                    {
                        "reason": "superseded",
                        "successful_actions": (
                            self._action_count - self._episode_action_start
                        ),
                    },
                )
            self._episode_index += 1
            self._episode_action_start = self._action_count
            self._episode_open = True
            self._append_policy_locked(
                "episode_started",
                {"prompt": str(prompt)},
            )
            return self._episode_index

    def end_episode(self, reason: str = "completed") -> None:
        with self._lock:
            if self._closed or not self._episode_open:
                return
            self._episode_open = False
            self._append_policy_locked(
                "episode_ended",
                {
                    "reason": str(reason),
                    "successful_actions": (
                        self._action_count - self._episode_action_start
                    ),
                },
            )

    def wrap_policy(self, policy: Any) -> "LoggedPolicy":
        return LoggedPolicy(policy, self)

    def log_policy_request(self, request: dict[str, Any]) -> int:
        with self._lock:
            request_index = self._request_index
            self._request_index += 1
            values = {
                str(key): _summarize_policy_value(str(key), value)
                for key, value in request.items()
            }
            self._append_policy_locked(
                "policy_request",
                {"request_index": request_index, "input": values},
            )
            return request_index

    def log_policy_response(self, request_index: int, response: Any) -> None:
        with self._lock:
            payload = (
                {
                    str(key): _summarize_policy_value(str(key), value)
                    for key, value in response.items()
                }
                if isinstance(response, dict)
                else _json_compatible(response)
            )
            self._append_policy_locked(
                "policy_response",
                {"request_index": int(request_index), "output": payload},
            )

    def log_policy_error(self, request_index: int, error: BaseException) -> None:
        with self._lock:
            self._append_policy_locked(
                "policy_error",
                {
                    "request_index": int(request_index),
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )

    def log_action(
        self,
        model_action: Any,
        actual_action: Any,
        *,
        raw_model_action: Any | None = None,
        episode_step: int,
        source: str,
        environment: Any | None = None,
        feedback_sample: tuple[Any, int] | None = None,
    ) -> None:
        """Log one successful SDK command and its immediate physical feedback.

        ``environment`` is deliberately optional for fake-observation runs.  A
        real environment must expose ``get_control_feedback()`` returning
        ``(vector, local_timestamp_ns)``.  Feedback sampling failures do not
        erase the already submitted command and do not stop control; that row
        is retained with ``feedback_action=null`` and one warning is printed.
        EE RTC can pass ``feedback_sample`` from its atomic command+feedback
        gate, avoiding a second SDK/FK read and preserving the exact gate sample.
        """

        # Do not add an SDK/FK read to the control loop when action logging was
        # explicitly disabled or already failed.
        with self._lock:
            if self._closed or self._disabled:
                return
        logging_started = time.perf_counter()

        feedback_action = None
        feedback_timestamp_ns = None
        feedback_kind = "simulated_command"
        if feedback_sample is not None:
            feedback_action, feedback_timestamp_ns = feedback_sample
            feedback_kind = "atomic_hardware_state_after_command_submit"
        elif environment is None:
            feedback_action = actual_action
            feedback_timestamp_ns = time.time_ns()
        else:
            feedback_kind = "hardware_state_after_command_submit"
            try:
                feedback_action, feedback_timestamp_ns = (
                    environment.get_control_feedback()
                )
            except Exception as error:
                feedback_kind = "hardware_feedback_unavailable"
                with self._lock:
                    if not self._feedback_warning_printed:
                        self._feedback_warning_printed = True
                        print(
                            "[action-log] warning: the SDK command was submitted "
                            "but physical feedback sampling failed; command is "
                            f"still logged without feedback: {error}",
                            file=sys.stderr,
                        )

        try:
            model = np.asarray(model_action, dtype=np.float64).reshape(-1)
            actual = np.asarray(actual_action, dtype=np.float64).reshape(-1)
            raw_model = np.asarray(
                model_action if raw_model_action is None else raw_model_action,
                dtype=np.float64,
            ).reshape(-1)
            expected = len(self.dimension_names)
            if any(
                value.shape != (expected,)
                for value in (raw_model, model, actual)
            ):
                raise ValueError(
                    f"expected {expected}D raw/model/actual actions, got "
                    f"{raw_model.shape}, {model.shape}, and {actual.shape}"
                )
            if not all(
                np.all(np.isfinite(value)) for value in (raw_model, model, actual)
            ):
                raise ValueError("raw/model/actual action contains NaN or Inf")
        except Exception as error:
            with self._lock:
                self._disable_locked(error)
            return

        feedback = None
        if feedback_action is not None:
            try:
                feedback = np.asarray(
                    feedback_action, dtype=np.float64
                ).reshape(-1)
                if feedback.shape != (expected,):
                    raise ValueError(
                        f"expected {expected}D feedback action, got {feedback.shape}"
                    )
                if not np.all(np.isfinite(feedback)):
                    raise ValueError("feedback action contains NaN or Inf")
                if feedback_timestamp_ns is not None:
                    feedback_timestamp_ns = int(feedback_timestamp_ns)
                    if feedback_timestamp_ns < 0:
                        raise ValueError("feedback timestamp must be non-negative")
            except Exception as error:
                feedback = None
                feedback_timestamp_ns = None
                feedback_kind = "hardware_feedback_unavailable"
                with self._lock:
                    if not self._feedback_warning_printed:
                        self._feedback_warning_printed = True
                        print(
                            "[action-log] warning: invalid physical feedback was "
                            "ignored; command logging continues: "
                            f"{error}",
                            file=sys.stderr,
                        )

        with self._lock:
            if self._closed or self._disabled:
                return
            self._action_count += 1
            record = self._base_record("command_applied")
            record.update(
                {
                    "record_index": self._action_count - 1,
                    "episode_step": int(episode_step),
                    "source": str(source),
                    "raw_model_action": raw_model.tolist(),
                    "model_action": model.tolist(),
                    "command_action": actual.tolist(),
                    # Compatibility for action-log-v1/v2 plot readers.  New
                    # code must call this a command, not a measured action.
                    "actual_action": actual.tolist(),
                    "feedback_action": (
                        None if feedback is None else feedback.tolist()
                    ),
                    "feedback_timestamp_ns": feedback_timestamp_ns,
                    "feedback_kind": feedback_kind,
                    "safety_changed_l2": float(np.linalg.norm(actual - model)),
                    "command_feedback_l2": (
                        None
                        if feedback is None
                        else float(np.linalg.norm(actual - feedback))
                    ),
                }
            )
            self._write_line_locked(self._action_file, record)
            # Include hardware feedback sampling, FK/packing, validation and
            # flush—not merely disk write—when deciding whether logging is
            # distorting the control period.
            write_s = time.perf_counter() - logging_started
            self._action_write_count += 1
            self._action_write_total_s += write_s
            self._action_write_max_s = max(self._action_write_max_s, write_s)
            self._maybe_warn_logging_overhead_locked()

    def _maybe_warn_logging_overhead_locked(self) -> None:
        """Suggest disabling logs when feedback+write work threatens the loop."""

        if (
            self._performance_warning_printed
            or self._control_period_s is None
            or self._action_write_count < 30
        ):
            return
        mean_s = self._action_write_total_s / self._action_write_count
        # A sustained 10% budget or one 25% spike is large enough to distort a
        # 20/30 Hz loop, while avoiding warnings for ordinary sub-millisecond IO.
        if mean_s < self._control_period_s * 0.10 and (
            self._action_write_max_s < self._control_period_s * 0.25
        ):
            return
        self._performance_warning_printed = True
        print(
            "[action-log] warning: action logging is consuming a noticeable "
            "part of the control period "
            f"(mean={mean_s * 1000.0:.2f} ms, "
            f"max={self._action_write_max_s * 1000.0:.2f} ms, "
            f"period={self._control_period_s * 1000.0:.2f} ms). "
            "For timing-sensitive runs, restart with ACTION_LOGGING=false.",
            file=sys.stderr,
        )

    def close(self, reason: str = "process_exit") -> None:
        with self._lock:
            if self._closed:
                return
            if self._episode_open:
                self._episode_open = False
                self._append_policy_locked(
                    "episode_ended",
                    {
                        "reason": str(reason),
                        "successful_actions": (
                            self._action_count - self._episode_action_start
                        ),
                    },
                )
            self._closed = True
            self._metadata["ended_at"] = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )
            self._metadata["duration_s"] = round(self._elapsed(), 6)
            self._metadata["successful_actions"] = self._action_count
            self._metadata["exit_reason"] = str(reason)
            self._append_policy_locked(
                "run_ended",
                {
                    "reason": str(reason),
                    "successful_actions": self._action_count,
                },
            )
            self._write_metadata_locked()
            for file_object in (self._policy_file, self._action_file):
                try:
                    if file_object is not None:
                        file_object.flush()
                        file_object.close()
                except Exception as error:
                    self._disable_locked(error)


class LoggedPolicy:
    """Transparent policy proxy that records complete numeric request/response data."""

    def __init__(self, policy: Any, session: ActionLogSession) -> None:
        self._policy = policy
        self._session = session

    def infer(self, request: dict[str, Any]) -> Any:
        request_index = self._session.log_policy_request(request)
        try:
            response = self._policy.infer(request)
        except BaseException as error:
            self._session.log_policy_error(request_index, error)
            raise
        self._session.log_policy_response(request_index, response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)


__all__ = [
    "ActionLogSession",
    "EE_DIMENSION_NAMES",
    "JOINT_DIMENSION_NAMES",
    "LOG_DIR_ENV",
    "LOG_SCHEMA_VERSION",
    "LoggedPolicy",
]
