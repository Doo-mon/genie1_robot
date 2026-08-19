"""推理入口共用的轻量基础设施与可靠 WebSocket 客户端。

图像部分负责把 RobotEnv 的 HWC 图像规范为 RGB ``uint8``，并按需用 Pillow
等比例缩放、补成方图；日志工具只打印请求字段形状和稀疏的夹爪切换事件，
不会输出大数组内容或在固定频率控制循环中逐帧刷屏。
运行时工厂延迟导入 RobotEnv、OpenPI 和 WebSocket，因此纯策略模块可在没有
机器人 SDK 时被检查。OpenPI 未安装时才回退到工作区 ``openpi-client/src``。

网络部分面向“本机端口经 SSH 转发到远端推理服务”的部署：首次连接失败、
握手或 metadata 超时会按指数退避重试。已经发送的推理请求默认不重发；只有
显式配置正数重试次数时，隧道断开后才重新握手并重发同一请求。服务端返回
文本错误或非法 msgpack 属于协议/模型错误，会立即上抛而不会被误判为网络抖动。
所有等待时间和重试上限由 EEArgs/JointArgs 透传：response 的 ``0`` 表示关闭
单次接收超时，两类 wait timeout 的 ``0`` 表示不限制累计等待时间。
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import importlib.util
import math
from pathlib import Path
import socket
import ssl
import sys
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import numpy as np


WORKSPACE_DIR = Path(__file__).resolve().parent
OPENPI_CLIENT_SRC = WORKSPACE_DIR / "openpi-client" / "src"


class PolicyConnectionError(RuntimeError):
    """策略服务器在配置的等待/重连预算内不可用。"""


class PolicyProtocolError(RuntimeError):
    """服务端已连接，但 metadata 或推理响应不符合 OpenPI 协议。"""


class PolicyClientClosedError(RuntimeError):
    """客户端已关闭，不能继续连接或发送推理请求。"""


@dataclasses.dataclass(frozen=True)
class WebsocketPolicyConfig:
    """WebSocket 超时与重试配置，所有数值单位均为秒。

    ``server_wait_timeout`` 控制程序启动时最多等多久；默认 0 表示持续等到
    SSH 隧道和远端模型就绪。``reconnect_wait_timeout`` 控制单次运行中断线后
    的重连预算。``response_timeout`` 同时约束 metadata 和一次推理响应；设为
    0 可关闭响应超时。``inference_retries`` 是原请求断线后允许重发的次数，
    默认 0，避免服务端重复处理已经送达但响应丢失的请求。
    """

    open_timeout: float = 10.0
    close_timeout: float = 10.0
    response_timeout: float = 300.0
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 10.0
    server_wait_timeout: float = 0.0
    reconnect_wait_timeout: float = 30.0
    inference_retries: int = 0

    def __post_init__(self) -> None:
        positive = {
            "open_timeout": self.open_timeout,
            "close_timeout": self.close_timeout,
            "retry_initial_delay": self.retry_initial_delay,
            "retry_max_delay": self.retry_max_delay,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")

        non_negative = {
            "response_timeout": self.response_timeout,
            "server_wait_timeout": self.server_wait_timeout,
            "reconnect_wait_timeout": self.reconnect_wait_timeout,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value}"
                )
        if self.retry_max_delay < self.retry_initial_delay:
            raise ValueError(
                "retry_max_delay must be greater than or equal to "
                "retry_initial_delay"
            )
        if (
            isinstance(self.inference_retries, bool)
            or not isinstance(self.inference_retries, int)
            or self.inference_retries < 0
        ):
            raise ValueError("inference_retries must be a non-negative integer")

    @property
    def response_timeout_or_none(self) -> float | None:
        return self.response_timeout or None

    @property
    def server_wait_timeout_or_none(self) -> float | None:
        return self.server_wait_timeout or None

    @property
    def reconnect_wait_timeout_or_none(self) -> float | None:
        return self.reconnect_wait_timeout or None


def _ensure_local_openpi_client() -> None:
    """Make the bundled ``openpi_client`` importable from standalone scripts.

    The repository keeps the client in a ``src`` layout but is commonly copied
    to robots without being installed as a package.  Centralising this one path
    adjustment avoids every entry point mutating ``sys.path`` independently.
    """

    if importlib.util.find_spec("openpi_client") is not None:
        return

    if not OPENPI_CLIENT_SRC.is_dir():
        raise RuntimeError(
            "openpi_client is not installed and the bundled source tree was not "
            f"found at {OPENPI_CLIENT_SRC}. Install openpi-client before running."
        )

    source = str(OPENPI_CLIENT_SRC)
    if source not in sys.path:
        sys.path.insert(0, source)


def resize_image(image: np.ndarray, *, enabled: bool, image_size: int) -> np.ndarray:
    """Return an RGB uint8 image, optionally resized with square padding."""

    result = np.asarray(image)
    if result.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {result.shape}")
    if result.shape[-1] == 4:
        result = result[..., :3]
    elif result.shape[-1] != 3:
        raise ValueError(f"Expected 3 or 4 image channels, got shape {result.shape}")
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)

    if enabled:
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        try:
            from PIL import Image
        except ModuleNotFoundError as error:
            raise RuntimeError("Image resizing requires Pillow.") from error

        image_object = Image.fromarray(result)
        current_width, current_height = image_object.size
        ratio = max(current_width / image_size, current_height / image_size)
        resized_width = max(1, int(current_width / ratio))
        resized_height = max(1, int(current_height / ratio))
        resampling = getattr(Image, "Resampling", Image)
        resized = image_object.resize(
            (resized_width, resized_height), resample=resampling.BILINEAR
        )
        canvas = Image.new(resized.mode, (image_size, image_size), 0)
        canvas.paste(
            resized,
            ((image_size - resized_width) // 2, (image_size - resized_height) // 2),
        )
        result = np.asarray(canvas)
    return result


def print_policy_request(request: dict[str, Any]) -> None:
    """Print request fields without dumping full image or state arrays."""

    print("\nFirst request to policy server:")
    for key, value in request.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:36s} shape={value.shape} dtype={value.dtype}")
        else:
            print(f"  {key:36s} {type(value).__name__}: {value}")
    print()


def format_binary_gripper_transitions(
    raw_action: Any,
    previous_command: Any,
    executed_command: Any,
    *,
    grippers: tuple[tuple[str, int], ...],
    prefix: str,
    step: int | None = None,
) -> list[str]:
    """Format sparse logs for actual binary gripper state transitions.

    Callers invoke this only after their normal shape/finiteness checks.  A
    message is returned only when the executed endpoint changes, so normal
    20--30 Hz control does not flood the terminal.  ``raw_action`` is included
    to distinguish an early model prediction from a client-side thresholding
    problem during real-robot tuning.
    """

    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    previous = np.asarray(previous_command, dtype=np.float32).reshape(-1)
    executed = np.asarray(executed_command, dtype=np.float32).reshape(-1)
    messages: list[str] = []
    step_text = "" if step is None else f" step={int(step)}"
    state_name = {0: "open", 1: "closed"}
    for side, index in grippers:
        if not 0 <= index < len(raw) or index >= len(previous) or index >= len(executed):
            raise ValueError(
                f"Gripper log index {index} is outside action shapes "
                f"raw={raw.shape}, previous={previous.shape}, executed={executed.shape}"
            )
        before = int(float(previous[index]) >= 0.5)
        after = int(float(executed[index]) >= 0.5)
        if before == after:
            continue
        messages.append(
            f"[{prefix}]{step_text} {side}: "
            f"{state_name[before]}({before}) -> {state_name[after]}({after}); "
            f"raw_action={float(raw[index]):.4f}"
        )
    return messages


def create_robot_environment(**kwargs: Any):
    """Construct ``RobotEnv`` without importing hardware SDKs at module load."""

    try:
        from robot_env import RobotEnv
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Robot dependencies are unavailable. Source env_for_g1.sh (or "
            "the equivalent robot SDK environment) before real-robot execution."
        ) from error
    return RobotEnv(**kwargs)


def build_websocket_uri(host: str, port: int) -> str:
    """把主机名或完整 ws/wss 地址规范成唯一、不重复端口的 URI。"""

    raw_host = str(host).strip()
    if not raw_host:
        raise ValueError("remote_host must not be empty")
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError(f"remote_port must be an integer, got {port!r}")
    if not 1 <= port <= 65535:
        raise ValueError(f"remote_port must be within [1, 65535], got {port}")

    if "://" not in raw_host:
        # urlsplit requires brackets around a bare IPv6 literal.
        if raw_host.count(":") >= 2 and not raw_host.startswith("["):
            raw_host = f"[{raw_host}]"
        raw_host = f"ws://{raw_host}"

    parsed = urlsplit(raw_host)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError(
            f"remote_host scheme must be ws or wss, got {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise ValueError(f"remote_host does not contain a hostname: {host!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentials must not be embedded in remote_host")
    if parsed.fragment:
        raise ValueError("remote_host must not contain a URI fragment")
    try:
        embedded_port = parsed.port
    except ValueError as error:
        raise ValueError(f"remote_host contains an invalid port: {host!r}") from error

    if embedded_port is not None:
        if embedded_port != port:
            raise ValueError(
                f"remote_host already contains port {embedded_port}, which conflicts "
                f"with remote_port={port}"
            )
        netloc = parsed.netloc
    else:
        hostname = parsed.hostname
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = f"{hostname}:{port}"

    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, "")
    )


def websocket_policy_config_from_args(args: Any) -> WebsocketPolicyConfig:
    """从 EEArgs、JointArgs 或兼容 Args 构造统一网络配置。"""

    return WebsocketPolicyConfig(
        open_timeout=args.ws_open_timeout,
        close_timeout=args.ws_close_timeout,
        response_timeout=args.ws_response_timeout,
        retry_initial_delay=args.ws_retry_initial_delay,
        retry_max_delay=args.ws_retry_max_delay,
        server_wait_timeout=args.ws_server_wait_timeout,
        reconnect_wait_timeout=args.ws_reconnect_wait_timeout,
        inference_retries=args.ws_inference_retries,
    )


def validate_websocket_args(args: Any) -> None:
    """在初始化机器人之前验证网络地址、端口、超时与重试参数。"""

    build_websocket_uri(args.remote_host, args.remote_port)
    websocket_policy_config_from_args(args)


def create_websocket_policy(
    host: str,
    port: int,
    *,
    config: WebsocketPolicyConfig | None = None,
):
    """延迟构造所有入口共用的可靠 WebSocket 策略客户端。"""

    return LongInferenceWebsocketPolicy(host, port, config=config)


def create_websocket_policy_from_args(args: Any):
    """用一个 CLI 参数对象创建客户端，避免各入口遗漏网络字段。"""

    validate_websocket_args(args)
    return create_websocket_policy(
        args.remote_host,
        args.remote_port,
        config=websocket_policy_config_from_args(args),
    )


class LongInferenceWebsocketPolicy:
    """适配慢模型启动、SSH 隧道抖动和长推理的同步策略客户端。

    初始连接默认持续重试，重试间隔从 ``retry_initial_delay`` 指数增长到
    ``retry_max_delay``。每次连接先完成 WebSocket 握手，再读取服务端第一条
    msgpack metadata。``infer`` 严格串行发送/接收；传输错误会关闭旧 socket，
    默认立即上抛，只有 ``inference_retries > 0`` 才重新握手并重发已打包请求。
    协议错误始终立即终止。调用 ``close`` 可打断退避等待，
    ``cancel_pending_inference`` 还可用于 RTC 停止正在进行的请求。
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        config: WebsocketPolicyConfig | None = None,
        _connect_fn: Callable[..., Any] | None = None,
        _msgpack_module: Any | None = None,
        _monotonic: Callable[[], float] = time.monotonic,
        _wait_fn: Callable[[float], None] | None = None,
        _websocket_exception: type[Exception] | tuple[type[Exception], ...] = (),
        _invalid_uri_exception: type[Exception] | tuple[type[Exception], ...] = (),
    ) -> None:
        self._uri = build_websocket_uri(host, port)
        self._config = config or WebsocketPolicyConfig()
        self._monotonic = _monotonic
        self._wait_fn = _wait_fn
        self._closed = threading.Event()
        self._wake_retry = threading.Event()
        self._cancel_lock = threading.Lock()
        self._cancel_epoch = 0
        self._io_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._ws = None
        self._connecting_ws = None
        self._metadata: Any = None

        if _connect_fn is None:
            try:
                import websockets.exceptions as websocket_exceptions
                import websockets.sync.client as websocket_client
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "WebSocket client dependencies are unavailable. Install "
                    "the dependencies declared by openpi-client before connecting."
                ) from error
            _connect_fn = websocket_client.connect
            _websocket_exception = websocket_exceptions.WebSocketException
            _invalid_uri_exception = websocket_exceptions.InvalidURI

        if _msgpack_module is None:
            _ensure_local_openpi_client()
            try:
                from openpi_client import msgpack_numpy
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "OpenPI msgpack support is unavailable. Install openpi-client "
                    "before connecting to the server."
                ) from error
            _msgpack_module = msgpack_numpy

        self._connect_fn = _connect_fn
        self._msgpack_numpy = _msgpack_module
        self._websocket_exception = _websocket_exception
        self._invalid_uri_exception = _invalid_uri_exception
        self._packer = _msgpack_module.Packer()
        self._connect_parameters = self._read_connect_parameters(_connect_fn)
        self._connect(
            max_wait=self._config.server_wait_timeout_or_none,
            purpose="initial connection",
            cancel_epoch=None,
        )

    @staticmethod
    def _read_connect_parameters(connect_fn: Callable[..., Any]) -> set[str]:
        try:
            return set(inspect.signature(connect_fn).parameters)
        except (TypeError, ValueError):
            return set()

    def _connect_kwargs(self, open_timeout: float) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "compression": None,
            "max_size": None,
            "open_timeout": open_timeout,
            "close_timeout": self._config.close_timeout,
        }
        # websockets 11--14 的同步 connect 没有 ping_interval；15+ 才提供。
        # 新版本显式关闭 ping，避免远端首次 JAX 编译阻塞事件循环时误断线。
        if "ping_interval" in self._connect_parameters:
            kwargs["ping_interval"] = None
        return kwargs

    def _is_retryable_transport_error(self, error: Exception) -> bool:
        if self._invalid_uri_exception and isinstance(
            error, self._invalid_uri_exception
        ):
            return False
        # Certificate/protocol configuration doesn't heal by waiting for a
        # model to load.  Retrying forever would hide a bad wss deployment.
        if isinstance(error, ssl.SSLError):
            return False
        if isinstance(error, socket.gaierror):
            return error.errno == getattr(socket, "EAI_AGAIN", None)
        if isinstance(error, (TimeoutError, EOFError)):
            return True
        if isinstance(error, OSError):
            return not isinstance(error, (PermissionError, FileNotFoundError))
        # websockets ConcurrencyError is a RuntimeError.  Our I/O lock prevents
        # normal concurrent calls; seeing it indicates a lifecycle bug, not a
        # tunnel fluctuation, so blindly reconnecting would hide the real issue.
        if isinstance(error, RuntimeError):
            return False
        if not (
            self._websocket_exception
            and isinstance(error, self._websocket_exception)
        ):
            return False

        exception_names = {cls.__name__ for cls in type(error).__mro__}
        if "ConnectionClosed" in exception_names:
            return True
        # A local SSH -L listener closes the TCP channel without an HTTP reply
        # while the remote model port isn't listening.  websockets reports this
        # as InvalidMessage, so it is a key slow-start retry case.
        if "InvalidMessage" in exception_names:
            return True
        if {"InvalidStatus", "InvalidStatusCode"} & exception_names:
            status = getattr(error, "status_code", None)
            response = getattr(error, "response", None)
            if status is None and response is not None:
                status = getattr(response, "status_code", None)
            return status in {408, 425, 429} or (
                isinstance(status, int) and 500 <= status <= 599
            )
        # InvalidURI, TLS/proxy/authentication failures, invalid headers,
        # negotiation errors and payload/protocol violations fail immediately.
        return False

    def _remaining(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - self._monotonic())

    def _bounded_timeout(
        self,
        configured: float | None,
        deadline: float | None,
    ) -> float | None:
        remaining = self._remaining(deadline)
        if remaining is not None and remaining <= 0:
            raise TimeoutError("connection wait budget exhausted")
        if configured is None:
            return remaining
        if remaining is None:
            return configured
        return min(configured, remaining)

    @staticmethod
    def _safe_close(connection: Any) -> None:
        if connection is None:
            return
        try:
            connection.close()
        except Exception:
            logging.debug("Ignoring websocket close failure", exc_info=True)

    def _discard_connection(self, connection: Any) -> None:
        """从共享状态移除指定连接，并在锁外执行可能阻塞的 close。"""

        with self._state_lock:
            if self._ws is connection:
                self._ws = None
            if self._connecting_ws is connection:
                self._connecting_ws = None
        self._safe_close(connection)

    def _raise_if_stopped(self, *, cancel_epoch: int | None) -> None:
        if self._closed.is_set():
            raise PolicyClientClosedError("Policy websocket client is closed")
        if cancel_epoch is not None:
            with self._cancel_lock:
                cancelled = cancel_epoch != self._cancel_epoch
            if cancelled:
                raise PolicyClientClosedError(
                    "Pending policy inference was cancelled"
                )

    def _wait_before_retry(
        self,
        delay: float,
        *,
        cancel_epoch: int | None,
    ) -> None:
        if self._wait_fn is not None:
            self._wait_fn(delay)
        else:
            self._wake_retry.wait(delay)
        self._raise_if_stopped(cancel_epoch=cancel_epoch)
        # Clear only while holding the same lock used by cancellation.  This
        # prevents a newly-set wake signal from being erased between the epoch
        # check and clear.
        with self._cancel_lock:
            if cancel_epoch is not None and cancel_epoch != self._cancel_epoch:
                raise PolicyClientClosedError(
                    "Pending policy inference was cancelled"
                )
            self._wake_retry.clear()

    def _connect(
        self,
        *,
        max_wait: float | None,
        purpose: str,
        cancel_epoch: int | None,
    ) -> tuple[Any, Any]:
        started = self._monotonic()
        deadline = None if max_wait is None else started + max_wait
        delay = self._config.retry_initial_delay
        attempt = 0
        last_error: Exception | None = None
        logging.info("Waiting for policy server at %s", self._uri)

        while True:
            self._raise_if_stopped(cancel_epoch=cancel_epoch)
            attempt += 1
            connection = None
            try:
                open_timeout = self._bounded_timeout(
                    self._config.open_timeout, deadline
                )
                if open_timeout is None:
                    open_timeout = self._config.open_timeout
                connection = self._connect_fn(
                    self._uri, **self._connect_kwargs(open_timeout)
                )
                with self._state_lock:
                    self._connecting_ws = connection
                # cancel_pending_inference cannot close a socket until
                # connect_fn returns it.  Check the generation immediately
                # after publishing it, before entering the potentially long
                # metadata recv.
                self._raise_if_stopped(cancel_epoch=cancel_epoch)
                metadata_timeout = self._bounded_timeout(
                    self._config.response_timeout_or_none, deadline
                )
                metadata_frame = connection.recv(timeout=metadata_timeout)
                self._raise_if_stopped(cancel_epoch=cancel_epoch)
                if isinstance(metadata_frame, str):
                    raise PolicyProtocolError(
                        "Policy server returned a text error during metadata "
                        f"handshake:\n{metadata_frame}"
                    )
                try:
                    metadata = self._msgpack_numpy.unpackb(metadata_frame)
                except Exception as error:
                    raise PolicyProtocolError(
                        "Policy server metadata is not valid OpenPI msgpack"
                    ) from error
                if not isinstance(metadata, dict):
                    raise PolicyProtocolError(
                        "Policy server metadata must be a dictionary, got "
                        f"{type(metadata).__name__}"
                    )
                # Publish the fully handshaken socket before hiding the
                # connecting handle.  cancel/close therefore sees it in one of
                # the two slots throughout the entire handoff.
                with self._state_lock:
                    self._ws = connection
                    self._metadata = metadata
                    self._connecting_ws = None
                self._raise_if_stopped(cancel_epoch=cancel_epoch)
                if attempt > 1:
                    logging.warning(
                        "Policy server is available at %s after %d attempts "
                        "(%.1fs).",
                        self._uri,
                        attempt,
                        self._monotonic() - started,
                    )
                return connection, metadata
            except (KeyboardInterrupt, SystemExit):
                self._discard_connection(connection)
                raise
            except Exception as error:
                self._discard_connection(connection)
                self._raise_if_stopped(cancel_epoch=cancel_epoch)
                if isinstance(error, PolicyProtocolError):
                    raise
                if not self._is_retryable_transport_error(error):
                    raise PolicyConnectionError(
                        f"Cannot establish {purpose} to {self._uri}: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                last_error = error

            remaining = self._remaining(deadline)
            if remaining is not None and remaining <= 0:
                raise PolicyConnectionError(
                    f"Policy server at {self._uri} was unavailable for "
                    f"{max_wait:.1f}s during {purpose}. Check the local port "
                    "forward and the remote model process."
                ) from last_error
            sleep_for = delay if remaining is None else min(delay, remaining)
            logging.warning(
                "Policy server not ready at %s (attempt %d, %.1fs elapsed, "
                "%s: %s); retrying in %.1fs. Check SSH port forwarding and "
                "remote model startup.",
                self._uri,
                attempt,
                self._monotonic() - started,
                type(last_error).__name__,
                last_error,
                sleep_for,
            )
            self._wait_before_retry(
                sleep_for,
                cancel_epoch=cancel_epoch,
            )
            delay = min(self._config.retry_max_delay, delay * 2.0)

    def get_server_metadata(self) -> Any:
        """返回最近一次成功连接收到的服务端 metadata。"""

        with self._state_lock:
            return self._metadata

    def reset(self) -> None:
        """兼容 OpenPI BasePolicy；当前协议没有远端 reset 消息。"""

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """发送一个只读推理请求；仅在显式配置后才有限重连、重发。"""

        # Snapshot before waiting for the serialization lock.  A cancellation
        # also invalidates infer calls already queued behind another request;
        # calls that begin after cancellation take the new generation normally.
        with self._cancel_lock:
            cancel_epoch = self._cancel_epoch

        with self._io_lock:
            with self._cancel_lock:
                if cancel_epoch != self._cancel_epoch:
                    raise PolicyClientClosedError(
                        "Pending policy inference was cancelled"
                    )
                self._wake_retry.clear()
            self._raise_if_stopped(cancel_epoch=cancel_epoch)
            payload = self._packer.pack(observation)

            for request_attempt in range(self._config.inference_retries + 1):
                self._raise_if_stopped(cancel_epoch=cancel_epoch)
                with self._state_lock:
                    connection = self._ws
                if connection is None:
                    connection, _ = self._connect(
                        max_wait=self._config.reconnect_wait_timeout_or_none,
                        purpose="inference reconnect",
                        cancel_epoch=cancel_epoch,
                    )
                    # _connect publishes before returning, then checks the
                    # cancel generation.  Keep a caller-side check as a final
                    # guard around the handoff-to-send boundary.
                    self._raise_if_stopped(cancel_epoch=cancel_epoch)
                try:
                    connection.send(payload)
                    response = connection.recv(
                        timeout=self._config.response_timeout_or_none
                    )
                    self._raise_if_stopped(cancel_epoch=cancel_epoch)
                except (KeyboardInterrupt, SystemExit):
                    self._discard_connection(connection)
                    raise
                except Exception as error:
                    self._discard_connection(connection)
                    self._raise_if_stopped(cancel_epoch=cancel_epoch)
                    if not self._is_retryable_transport_error(error):
                        raise PolicyConnectionError(
                            f"Policy inference transport failed at {self._uri}: "
                            f"{type(error).__name__}: {error}"
                        ) from error
                    if request_attempt >= self._config.inference_retries:
                        raise PolicyConnectionError(
                            f"Policy inference at {self._uri} failed after "
                            f"{request_attempt + 1} attempts"
                        ) from error
                    logging.warning(
                        "Policy inference connection failed (%s: %s); "
                        "reconnecting and replaying request (%d/%d).",
                        type(error).__name__,
                        error,
                        request_attempt + 1,
                        self._config.inference_retries,
                    )
                    continue

                if isinstance(response, str):
                    self._discard_connection(connection)
                    raise RuntimeError(f"Error in inference server:\n{response}")
                try:
                    unpacked = self._msgpack_numpy.unpackb(response)
                except Exception as error:
                    self._discard_connection(connection)
                    raise PolicyProtocolError(
                        "Policy inference response is not valid OpenPI msgpack"
                    ) from error
                if not isinstance(unpacked, dict):
                    self._discard_connection(connection)
                    raise PolicyProtocolError(
                        "Policy inference response must be a dictionary, got "
                        f"{type(unpacked).__name__}"
                    )
                self._raise_if_stopped(cancel_epoch=cancel_epoch)
                return unpacked

        raise AssertionError("unreachable inference retry state")

    def cancel_pending_inference(self) -> None:
        """中断 RTC 正在等待的响应；下一次 infer 会自动建立新连接。"""

        with self._cancel_lock:
            self._cancel_epoch += 1
            self._wake_retry.set()
        with self._state_lock:
            current = self._ws
            connecting = self._connecting_ws
            self._ws = None
            self._connecting_ws = None
        self._safe_close(current)
        if connecting is not current:
            self._safe_close(connecting)

    def close(self) -> None:
        """永久关闭客户端并唤醒所有退避等待。"""

        self._closed.set()
        with self._cancel_lock:
            self._cancel_epoch += 1
            self._wake_retry.set()
        with self._state_lock:
            current = self._ws
            connecting = self._connecting_ws
            self._ws = None
            self._connecting_ws = None
        self._safe_close(current)
        if connecting is not current:
            self._safe_close(connecting)
