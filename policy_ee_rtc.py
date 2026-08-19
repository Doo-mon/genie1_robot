"""面向 EE14 动作块的可复用异步 RTC 执行运行时。

推理线程按阈值补充 ``H x 14`` 队列，执行线程按固定频率消费绝对 EE 目标。
请求前会从末次真实下发命令开始，顺序模拟剩余目标经过限幅和可选夹爪
二值化后的物理命令；响应按已消费步数对齐，并硬锁真正承诺执行的前缀。
本地 committed queue、机器人命令和 WebSocket prefix 都固定使用 0--1
夹爪动作空间，可保留连续值或用双阈值滞回变为端点；按当前训练数据约定
0=张开、1=闭合。0--120 只属于反馈 state，不允许作为 RTC action prefix。
若响应返回时执行进度已经越过承诺前缀，则保留当前队列并重新请求，避免把
没有被服务端锁定的历史动作接到正在执行的时间线上。
本模块依赖 ``policy_ee``、NumPy 与请求摘要工具；进度条仅在运行时延迟导入。
机器人环境、websocket policy 和可选 action callback 均由调用方注入，不在此构造。
``inference_pi05_g1_desk_ee_rtc`` 与 EE rollout 入口复用该运行时。
所有目标在下发前经过 EE 单步安全限幅，工作线程异常会停止双方并回传主线程。
二值夹爪切换默认与对应手臂目标同步：若手臂因单步限幅尚未到达切换帧位姿，
执行线程保持原夹爪并暂不消费该模型行；门控期间的旧推理响应会被安全拒绝。
正常结束或 Ctrl-C 时会取消尚未返回的网络请求，防止遗留线程跨 episode 读取。
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Callable

import numpy as np

import policy_ee
from inference_common import (
    PolicyClientClosedError,
    format_binary_gripper_transitions,
    print_policy_request,
    validate_websocket_args,
)


RTC_PREFIX_GRIPPER_RANGE = "0-1"


def _tqdm():
    try:
        import tqdm
    except ModuleNotFoundError as error:
        raise RuntimeError("RTC progress reporting requires tqdm.") from error
    return tqdm


def _checked_ee_chunk(value, *, name: str) -> np.ndarray:
    """Validate and copy one ``H x EE_DIM`` chunk, including an empty chunk."""

    chunk = np.asarray(value, dtype=np.float32)
    if chunk.ndim == 1 and len(chunk) == 0:
        chunk = chunk.reshape(0, policy_ee.EE_DIM)
    if chunk.ndim != 2 or chunk.shape[1] != policy_ee.EE_DIM:
        raise ValueError(
            f"Expected {name} shape (H, {policy_ee.EE_DIM}), got {chunk.shape}"
        )
    if not np.all(np.isfinite(chunk)):
        raise ValueError(f"{name} contains NaN or Inf")
    return chunk.copy()


def _aligned_replacement(
    actions: np.ndarray, elapsed_pops: int
) -> tuple[int, list[np.ndarray]]:
    """Discard response rows whose corresponding commands already executed."""

    chunk = _checked_ee_chunk(actions, name="EE action chunk")
    use_index = min(max(0, int(elapsed_pops)), len(chunk))
    return use_index, [action.copy() for action in chunk[use_index:]]


def _hard_lock_prefix(
    actions: np.ndarray,
    committed_commands: np.ndarray,
    lock_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Overwrite a response prefix with exact safety-processed EE commands."""

    result = _checked_ee_chunk(actions, name="EE action chunk")
    committed = _checked_ee_chunk(
        committed_commands, name="committed EE command chunk"
    )
    count = min(max(0, int(lock_len)), len(result), len(committed))
    if count == 0:
        return result, np.empty((0,), dtype=np.float32)

    errors = np.linalg.norm(result[:count] - committed[:count], axis=-1)
    result[:count] = committed[:count]
    return result, errors.astype(np.float32, copy=False)


def _simulate_committed_commands(
    remaining_targets,
    last_command,
    args: "RTCArgs",
) -> np.ndarray:
    """Return the exact EE commands pending on the physical timeline.

    Safety processing is sequential: each target is limited relative to the
    command immediately before it.  When a gripper transition needs more than
    one arm tick, the execution thread will hold that queue row instead of
    consuming it.  Such a row and everything after it are therefore not yet a
    one-row-per-tick committed timeline and must not be sent as an RTC prefix.
    """

    targets = _checked_ee_chunk(
        remaining_targets, name="remaining EE target chunk"
    )
    previous = policy_ee.checked_vector(
        last_command,
        dim=policy_ee.EE_DIM,
        name="last executed EE command",
    )
    committed: list[np.ndarray] = []
    for target in targets:
        command, pending_arms = _process_command_with_gripper_sync(
            target, previous, args
        )
        if pending_arms:
            break
        previous = command
        committed.append(previous.copy())

    if not committed:
        return np.empty((0, policy_ee.EE_DIM), dtype=np.float32)
    return np.stack(committed, axis=0).astype(np.float32, copy=False)


def _arm_target_reached(
    previous_command: np.ndarray,
    target: np.ndarray,
    *,
    position: slice,
    rotation: slice,
) -> bool:
    """Whether the previously published command reached a grip-event target."""

    # ``process_ee_action`` returns the exact target component once it is inside
    # the configured step limit.  A small absolute tolerance only absorbs
    # float32 round-off; it is not a physical arrival tolerance.
    position_error = float(
        np.linalg.norm(previous_command[position] - target[position])
    )
    rotation_error = float(
        np.linalg.norm(previous_command[rotation] - target[rotation])
    )
    return position_error <= 1e-6 and rotation_error <= 1e-6


def _process_command_with_gripper_sync(
    target: np.ndarray,
    previous_command: np.ndarray,
    args: "RTCArgs",
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Limit one EE target and defer grip changes until its arm reaches it.

    The returned arm command is identical to the ordinary RTC limiter.  In
    binary mode, however, a requested gripper transition is held at its prior
    value while the associated arm still has limited pose distance remaining.
    ``pending_arms`` tells the caller to keep the current model row at the head
    of the queue and execute it again on the next physical tick.
    """

    checked_target = policy_ee.checked_vector(
        target, dim=policy_ee.EE_DIM, name="EE gripper-sync target"
    )
    previous = policy_ee.checked_vector(
        previous_command,
        dim=policy_ee.EE_DIM,
        name="previous EE gripper-sync command",
    )
    command = policy_ee.process_ee_action(
        checked_target,
        previous,
        max_position_delta=args.max_position_delta,
        max_rotation_delta=args.max_rotation_delta,
        binarize_gripper=args.binarize_gripper,
        gripper_threshold=args.gripper_threshold,
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
    )
    if not args.sync_gripper_to_arm or not args.binarize_gripper:
        return command, ()

    pending: list[str] = []
    arm_specs = (
        (
            "left",
            policy_ee.LEFT_POSITION,
            policy_ee.LEFT_ROTATION,
            policy_ee.LEFT_GRIPPER,
        ),
        (
            "right",
            policy_ee.RIGHT_POSITION,
            policy_ee.RIGHT_ROTATION,
            policy_ee.RIGHT_GRIPPER,
        ),
    )
    for name, position, rotation, gripper in arm_specs:
        transition_requested = abs(
            float(command[gripper]) - float(previous[gripper])
        ) >= 0.5
        if not transition_requested:
            continue
        # Test the previously published command, not the command being built
        # for this tick.  Thus the arm receives the endpoint while the old
        # gripper is still held, gets one complete control period to execute
        # that last SDK segment, and only switches the gripper on the next tick.
        if not _arm_target_reached(
            previous,
            checked_target,
            position=position,
            rotation=rotation,
        ):
            command[gripper] = previous[gripper]
            pending.append(name)
    return command.astype(np.float32, copy=False), tuple(pending)


def _encode_prefix_for_server(
    committed_commands,
    gripper_range: str,
) -> np.ndarray:
    """Copy a committed EE chunk into the fixed physical-action wire contract."""

    encoded = _checked_ee_chunk(
        committed_commands, name="committed EE command chunk"
    )
    mode = str(gripper_range).strip()
    if mode != RTC_PREFIX_GRIPPER_RANGE:
        raise ValueError(
            "EE RTC prefix grippers are fixed to the physical action range '0-1'; "
            f"got {gripper_range!r}"
        )
    gripper_indices = (policy_ee.LEFT_GRIPPER, policy_ee.RIGHT_GRIPPER)
    grippers = encoded[:, gripper_indices]
    if np.any(grippers < policy_ee.GRIPPER_ACTION_MIN) or np.any(
        grippers > policy_ee.GRIPPER_ACTION_MAX
    ):
        raise ValueError("EE RTC prefix gripper actions must stay within [0, 1]")
    return encoded


@dataclasses.dataclass
class RTCArgs(policy_ee.EEArgs):
    # RTC always consumes the full configured output. Prefix skipping would shift the
    # committed timeline and must remain disabled.
    execution_horizon: int = dataclasses.field(init=False, default=0)
    chunk_skip: int = 0
    control_frequency: float = 30.0

    rtc_prefix_len: int = 8
    """Number of leading pending actions hard-locked into the new chunk."""

    rtc_refill_at: int = 12
    """Request a replacement when the pending queue reaches this length."""

    rtc_prefix_gripper_range: str = dataclasses.field(
        init=False, default=RTC_PREFIX_GRIPPER_RANGE
    )
    """Fixed server wire contract: physical EE action grippers use 0..1."""

    # Conservative EE command limits at 30 Hz.
    max_position_delta: float | None = 0.002
    max_rotation_delta: float | None = 0.01
    # Delay a binary gripper transition until the associated arm command has
    # reached that model row's absolute EE target.  This keeps rate limiting
    # from making grasp/release happen earlier than the arm phase.
    sync_gripper_to_arm: bool = True
    gripper_sync_timeout: float = 2.0
    # ``none`` keeps both arms active. ``left``/``right`` freezes the selected
    # arm in both the RTC policy timeline and RobotEnv hardware command path.
    freeze_arm: str = "none"


def resolve_freeze_arm(args: RTCArgs) -> str:
    """Resolve the RTC single-arm selector with legacy left-freeze support."""

    freeze_arm = str(args.freeze_arm).strip().casefold()
    if freeze_arm not in {"none", "left", "right"}:
        raise ValueError("freeze_arm must be one of: none, left, right")
    if args.freeze_left_arm:
        if freeze_arm == "right":
            raise ValueError(
                "freeze_left_arm conflicts with freeze_arm='right'; use only "
                "freeze_arm for the single-arm RTC test"
            )
        freeze_arm = "left"
    return freeze_arm


def validate_rtc_args(args: RTCArgs) -> None:
    validate_websocket_args(args)
    if not np.isfinite(args.control_frequency) or args.control_frequency <= 0:
        raise ValueError("control_frequency must be finite and positive")
    if args.max_timesteps <= 0:
        raise ValueError("max_timesteps must be positive")
    if args.action_horizon <= 1:
        raise ValueError("action_horizon must be greater than 1 for RTC")
    if args.chunk_skip != 0:
        raise ValueError("RTC timeline alignment requires chunk_skip=0")
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
    if (
        not np.isfinite(args.gripper_sync_timeout)
        or args.gripper_sync_timeout <= 0
    ):
        raise ValueError("gripper_sync_timeout must be finite and positive")
    if not 1 <= args.rtc_prefix_len < args.action_horizon:
        raise ValueError(
            f"rtc_prefix_len must be in [1, {args.action_horizon - 1}]"
        )
    if not args.rtc_prefix_len <= args.rtc_refill_at < args.action_horizon:
        raise ValueError(
            "rtc_refill_at must be >= rtc_prefix_len and smaller than the horizon"
        )
    if args.rtc_prefix_gripper_range != RTC_PREFIX_GRIPPER_RANGE:
        raise ValueError(
            "EE RTC prefix grippers are fixed to '0-1', "
            f"got {args.rtc_prefix_gripper_range!r}"
        )
    resolve_freeze_arm(args)
    policy_ee.resolve_gripper_thresholds(
        gripper_threshold=args.gripper_threshold,
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
    )
    if not args.task_name.strip():
        raise ValueError("task_name must not be empty")
    if args.prompt is not None and not args.prompt.strip():
        raise ValueError("prompt must not be empty when provided")
    if args.prompt_key is not None and not args.prompt_key.strip():
        raise ValueError("prompt_key must not be empty when provided")


class RTCRunner:
    """Run inference and fixed-rate EE execution on two worker threads."""

    def __init__(
        self,
        args: RTCArgs,
        env,
        policy,
        prompt: str,
        initial_obs: dict,
        action_callback: Callable[[np.ndarray], None] | None = None,
        action_trace_callback: (
            Callable[[np.ndarray, np.ndarray, int, np.ndarray], None] | None
        ) = None,
    ) -> None:
        self.args = args
        self.env = env
        self.policy = policy
        self.prompt = str(prompt)
        self.initial_obs = initial_obs
        self.action_callback = action_callback
        # The trace callback receives effective target, applied command,
        # 1-based step, and the untouched server row.
        self.action_trace_callback = action_trace_callback
        self.period = 1.0 / args.control_frequency
        initial_command = policy_ee.state_ee_to_action_command(
            initial_obs["state/pose"],
        )
        if args.binarize_gripper:
            # Establish an endpoint-valued latch before the first RTC command;
            # feedback state may contain an intermediate 0..120 opening.
            initial_command[policy_ee.LEFT_GRIPPER] = float(
                initial_command[policy_ee.LEFT_GRIPPER] >= 0.5
            )
            initial_command[policy_ee.RIGHT_GRIPPER] = float(
                initial_command[policy_ee.RIGHT_GRIPPER] >= 0.5
            )
        self._freeze_arm = resolve_freeze_arm(args)
        self._frozen_arm_target = initial_command.copy()

        self._lock = threading.Lock()
        # Keep a sensor observation and its queue/last-command snapshot in the
        # same physical epoch.  Execution takes this gate from dequeue through
        # env.step, while network inference remains fully outside it.
        self._observation_gate = threading.Lock()
        self._stop = threading.Event()
        self._queue: list[np.ndarray] = []
        self._raw_queue: list[np.ndarray] = []
        self._last_command = initial_command.copy()
        self._pop_count = 0
        self._steps_done = 0
        # A gate tick publishes a physical command without consuming the model
        # row.  ``_timeline_revision`` makes every inference response that
        # crossed such a tick stale even though its pop-count did not change.
        self._timeline_revision = 0
        self._gripper_gate_active = False
        self._gripper_gate_steps = 0
        self._gripper_gate_max_steps = max(
            1,
            int(
                np.ceil(
                    float(args.gripper_sync_timeout)
                    * float(args.control_frequency)
                )
            ),
        )
        self._gripper_gate_events = 0
        self._error: BaseException | None = None
        self._printed_first_request = False
        self._handoff_rejections = 0
        self._overrun_count = 0
        self._overrun_total_ms = 0.0
        self._overrun_max_ms = 0.0

    def _fail(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
            self._stop.set()

    def _request(
        self,
        obs: dict,
        remaining_targets: list[np.ndarray],
        last_command: np.ndarray,
        lock_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Infer an EE chunk and return it with its committed physical prefix."""

        request = policy_ee.prepare_policy_request(
            obs,
            self.prompt,
            resize_images=self.args.resize_images,
            image_size=self.args.image_size,
        )
        if remaining_targets:
            remaining_chunk = np.stack(remaining_targets, axis=0).astype(
                np.float32, copy=False
            )
        else:
            remaining_chunk = np.empty(
                (0, policy_ee.EE_DIM), dtype=np.float32
            )
        committed = _simulate_committed_commands(
            remaining_chunk, last_command, self.args
        )
        lock_len = min(max(0, int(lock_len)), len(committed))
        if lock_len > 0:
            # The adapted server transform accepts physical EE14 commands.
            # Normal observation/state remains [joint16, ee14].
            request["remaining_action_chunk"] = _encode_prefix_for_server(
                committed,
                self.args.rtc_prefix_gripper_range,
            )
            request["prefix_len"] = lock_len

        if not self._printed_first_request:
            print_policy_request(request)
            self._printed_first_request = True

        if self._stop.is_set():
            raise PolicyClientClosedError(
                "EE RTC stopped before starting the inference request"
            )
        start = time.perf_counter()
        response = self.policy.infer(request)
        infer_ms = (time.perf_counter() - start) * 1000.0
        actions = policy_ee.extract_ee_actions(
            response, max_horizon=self.args.action_horizon
        )
        raw_actions = actions.copy()
        if self._freeze_arm != "none":
            actions = policy_ee.freeze_ee_arm(
                actions,
                self._frozen_arm_target,
                arm=self._freeze_arm,
            )
        if len(actions) != self.args.action_horizon:
            raise ValueError(
                f"Server returned horizon {len(actions)}, but client expects "
                f"{self.args.action_horizon}"
            )
        actions, prefix_errors = _hard_lock_prefix(
            actions, committed, lock_len
        )

        if self.args.debug:
            diagnostic = ""
            if len(prefix_errors):
                diagnostic = (
                    f" hard_lock_pre_mean={float(prefix_errors.mean()):.4f}"
                    f" hard_lock_pre_max={float(prefix_errors.max()):.4f} post=0"
                )
            _tqdm().tqdm.write(
                f"infer_ms={infer_ms:.1f} remaining_len={len(committed)} "
                f"lock_len={lock_len} prefix_gripper_range="
                f"{self.args.rtc_prefix_gripper_range} "
                f"chunk={len(actions)}{diagnostic}"
            )
        return actions, committed, raw_actions

    def _inference_loop(self) -> None:
        try:
            # First chunk has no previous trajectory to guide it.
            with self._lock:
                initial_last_command = self._last_command.copy()
            actions, _, raw_actions = self._request(
                self.initial_obs, [], initial_last_command, 0
            )
            with self._lock:
                if not self._stop.is_set():
                    self._queue = [action.copy() for action in actions]
                    self._raw_queue = [action.copy() for action in raw_actions]

            while not self._stop.is_set():
                with self._lock:
                    queue_len = len(self._queue)
                    gripper_gate_active = self._gripper_gate_active
                # While a transition row is held, its arm command changes but
                # the model-row pop count does not.  Wait for the gate to finish
                # before taking another RTC snapshot.  A request that started
                # just before the gate is caught by the revision check below.
                if gripper_gate_active or queue_len > self.args.rtc_refill_at:
                    time.sleep(self.period / 2.0)
                    continue

                # Keep the physical observation and the committed timeline in
                # one epoch.  Execution uses the same gate from dequeue through
                # env.step, so it cannot advance between these two operations.
                with self._observation_gate:
                    obs = (
                        self.env.get_obs()
                        if self.env is not None
                        else policy_ee.fake_obs()
                    )
                    with self._lock:
                        # Stop may have been requested while get_obs() blocked.
                        if self._stop.is_set():
                            return
                        remaining = [action.copy() for action in self._queue]
                        request_pop_count = self._pop_count
                        request_timeline_revision = self._timeline_revision
                        request_last_command = self._last_command.copy()
                lock_len = min(self.args.rtc_prefix_len, len(remaining))
                actions, committed, raw_actions = self._request(
                    obs, remaining, request_last_command, lock_len
                )
                # Committed-prefix simulation deliberately stops before a row
                # that will activate the gripper/arm gate.
                lock_len = min(lock_len, len(committed))

                with self._lock:
                    elapsed_pops = max(0, self._pop_count - request_pop_count)
                    use_index, replacement = _aligned_replacement(
                        actions, elapsed_pops
                    )
                    _, raw_replacement = _aligned_replacement(
                        raw_actions, elapsed_pops
                    )
                    timeline_changed = (
                        self._timeline_revision != request_timeline_revision
                    )
                    handoff_rejected = timeline_changed or (
                        lock_len > 0 and elapsed_pops > lock_len
                    )
                    if handoff_rejected:
                        self._handoff_rejections += 1
                    if not self._stop.is_set() and not handoff_rejected:
                        self._queue = replacement
                        self._raw_queue = raw_replacement

                if handoff_rejected:
                    if timeline_changed:
                        logging.warning(
                            "Rejecting EE RTC response because a gripper/arm "
                            "gate advanced the physical command timeline "
                            "without consuming a model row. Keeping the "
                            "current queue and requesting a fresh chunk."
                        )
                    else:
                        logging.warning(
                            "Rejecting EE RTC response beyond committed prefix: "
                            "elapsed_pops=%d, lock_len=%d. Keeping the current "
                            "queue and requesting a fresh chunk.",
                            elapsed_pops,
                            lock_len,
                        )

                if self.args.debug:
                    committed_l2 = "unlocked"
                    if (
                        use_index < lock_len
                        and use_index < len(committed)
                        and use_index < len(actions)
                    ):
                        committed_l2 = (
                            f"{float(np.linalg.norm(actions[use_index] - committed[use_index])):.6f}"
                        )
                    _tqdm().tqdm.write(
                        f"[RTC-handoff] request_remaining={len(remaining)} "
                        f"lock_len={lock_len} elapsed_pops={elapsed_pops} "
                        f"timeline_changed={timeline_changed} "
                        f"use_index={use_index} replacement_len={len(replacement)} "
                        f"accepted={not handoff_rejected} "
                        f"committed_l2={committed_l2}"
                    )
        except PolicyClientClosedError as error:
            # ``run`` 会在正常结束或 Ctrl-C 时主动打断阻塞中的 recv；此时
            # stop 已设置，不应把预期的取消包装成一次 RTC 运行失败。
            if not self._stop.is_set():
                self._fail(error)
        except BaseException as error:
            self._fail(error)

    def _execution_loop(self) -> None:
        starving = False
        try:
            while not self._stop.is_set():
                tick_start = time.perf_counter()
                transition_messages: list[str] = []
                gate_message = None
                model_target = None
                executed_step = None
                with self._observation_gate:
                    with self._lock:
                        if self._stop.is_set():
                            command = None
                            finished = True
                        elif self._steps_done >= self.args.max_timesteps:
                            command = None
                            finished = True
                        elif self._queue:
                            # Peek first.  A gripper-transition row remains at
                            # the head until the limited arm command reaches
                            # that row's absolute pose target.
                            target = self._queue[0]
                            raw_model_target = (
                                self._raw_queue[0]
                                if self._raw_queue
                                else target.copy()
                            )
                            model_target = target.copy()
                            previous_command = self._last_command.copy()
                            command, pending_arms = (
                                _process_command_with_gripper_sync(
                                    target,
                                    self._last_command,
                                    self.args,
                                )
                            )
                            was_gated = self._gripper_gate_active
                            if pending_arms:
                                self._gripper_gate_active = True
                                self._gripper_gate_steps += 1
                                if not was_gated:
                                    self._gripper_gate_events += 1
                                    gate_message = (
                                        "[EE RTC gripper-sync] holding model "
                                        "row until arm target is reached: "
                                        + ",".join(pending_arms)
                                    )
                                if (
                                    self._gripper_gate_steps
                                    > self._gripper_gate_max_steps
                                ):
                                    raise RuntimeError(
                                        "EE RTC gripper/arm synchronization "
                                        "timed out after "
                                        f"{self.args.gripper_sync_timeout:g}s "
                                        f"for {','.join(pending_arms)} arm; "
                                        "the gripper was kept at its previous "
                                        "state and the episode was stopped"
                                    )
                            else:
                                # This row is now complete (or never needed a
                                # gate), so it advances the model/RTC timeline.
                                self._queue.pop(0)
                                if self._raw_queue:
                                    self._raw_queue.pop(0)
                                self._pop_count += 1
                                self._gripper_gate_active = False
                                self._gripper_gate_steps = 0
                                if was_gated:
                                    gate_message = (
                                        "[EE RTC gripper-sync] arm target "
                                        "reached; applying gripper transition "
                                        "and resuming RTC row consumption"
                                    )

                            # Every physical command issued while a row is or
                            # was gated invalidates an inference snapshot whose
                            # pop count alone cannot observe that movement.
                            if pending_arms or was_gated:
                                self._timeline_revision += 1
                            if self.args.binarize_gripper:
                                transition_messages = (
                                    format_binary_gripper_transitions(
                                        target,
                                        previous_command,
                                        command,
                                        grippers=(
                                            ("left", policy_ee.LEFT_GRIPPER),
                                            ("right", policy_ee.RIGHT_GRIPPER),
                                        ),
                                        prefix="EE RTC gripper",
                                        step=self._steps_done + 1,
                                    )
                                )
                            self._last_command = command.copy()
                            self._steps_done += 1
                            executed_step = self._steps_done
                            finished = False
                        else:
                            command = None
                            finished = False

                    if command is not None and self.env is not None:
                        self.env.step(command.copy(), update_obs=False)

                if finished:
                    break
                if command is None:
                    if self.args.debug and not starving:
                        _tqdm().tqdm.write(
                            "[RTC] action queue starved; holding last command"
                        )
                    starving = True
                else:
                    if self.args.debug and starving:
                        _tqdm().tqdm.write("[RTC] action queue recovered")
                    starving = False
                    if gate_message is not None:
                        _tqdm().tqdm.write(gate_message)
                    for message in transition_messages:
                        _tqdm().tqdm.write(message)
                    if (
                        self.action_trace_callback is not None
                        and model_target is not None
                        and executed_step is not None
                    ):
                        self.action_trace_callback(
                            model_target.copy(),
                            command.copy(),
                            executed_step,
                            raw_model_target.copy(),
                        )
                    if self.action_callback is not None:
                        self.action_callback(command.copy())

                elapsed = time.perf_counter() - tick_start
                if elapsed < self.period:
                    time.sleep(self.period - elapsed)
                else:
                    elapsed_ms = elapsed * 1000.0
                    self._overrun_count += 1
                    self._overrun_total_ms += elapsed_ms
                    self._overrun_max_ms = max(self._overrun_max_ms, elapsed_ms)
                    if self.args.debug and (
                        self._overrun_count == 1 or self._overrun_count % 30 == 0
                    ):
                        _tqdm().tqdm.write(
                            "[EE RTC] control overruns: "
                            f"count={self._overrun_count} "
                            f"mean={self._overrun_total_ms / self._overrun_count:.1f} ms "
                            f"max={self._overrun_max_ms:.1f} ms"
                        )
            with self._lock:
                self._stop.set()
        except BaseException as error:
            self._fail(error)

    def run(self) -> None:
        validate_rtc_args(self.args)
        inference_thread = threading.Thread(
            target=self._inference_loop, name="ee-rtc-inference", daemon=True
        )
        execution_thread = threading.Thread(
            target=self._execution_loop, name="ee-rtc-execution", daemon=True
        )
        inference_thread.start()
        execution_thread.start()
        try:
            with _tqdm().tqdm(
                total=self.args.max_timesteps, desc="running", unit="step"
            ) as progress:
                shown = 0
                while not self._stop.wait(0.1):
                    with self._lock:
                        done = self._steps_done
                    if done > shown:
                        progress.update(done - shown)
                        shown = done
                with self._lock:
                    done = self._steps_done
                if done > shown:
                    progress.update(done - shown)
        except KeyboardInterrupt:
            with self._lock:
                self._stop.set()
            raise
        finally:
            with self._lock:
                self._stop.set()
            cancel = getattr(self.policy, "cancel_pending_inference", None)
            execution_thread.join(timeout=2.0)
            # Re-issue cancellation while joining.  This closes the narrow
            # window where stop is set after _request's check but immediately
            # before policy.infer establishes its own cancellation epoch.
            deadline = time.monotonic() + 5.0
            while inference_thread.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if callable(cancel):
                    cancel()
                inference_thread.join(timeout=min(0.2, remaining))
            lingering = [
                thread.name
                for thread in (execution_thread, inference_thread)
                if thread.is_alive()
            ]
            if lingering:
                close_policy = getattr(self.policy, "close", None)
                if callable(close_policy):
                    close_policy()
                # Raise inside finally on purpose: even while propagating
                # KeyboardInterrupt, a live worker must terminate the process
                # instead of allowing the entry point to start another episode.
                raise RuntimeError(
                    "RTC worker threads did not stop: " + ", ".join(lingering)
                )

        with self._lock:
            error = self._error
            steps_done = self._steps_done
            pending = len(self._queue)
            handoff_rejections = self._handoff_rejections
            gripper_gate_events = self._gripper_gate_events
            overrun_count = self._overrun_count
            overrun_total_ms = self._overrun_total_ms
            overrun_max_ms = self._overrun_max_ms
        if error is not None:
            raise RuntimeError("EE RTC worker failed") from error
        _tqdm().tqdm.write(
            f"[EE RTC] episode complete: steps={steps_done}, pending={pending}, "
            f"late_responses_rejected={handoff_rejections}, "
            f"gripper_sync_events={gripper_gate_events}, "
            f"overruns={overrun_count}, "
            f"overrun_mean_ms="
            f"{(overrun_total_ms / overrun_count if overrun_count else 0.0):.1f}, "
            f"overrun_max_ms={overrun_max_ms:.1f}"
        )


def warmup_policy(args: RTCArgs, policy, obs: dict) -> None:
    """Compile both JAX inference variants before robot execution starts."""
    request = policy_ee.prepare_policy_request(
        obs,
        "warmup",
        resize_images=args.resize_images,
        image_size=args.image_size,
    )
    print("Warming up no-prefix inference variant...")
    policy.infer(dict(request))

    current_ee = policy_ee.state_ee_to_action_command(
        obs["state/pose"]
    )
    if args.binarize_gripper:
        current_ee[policy_ee.LEFT_GRIPPER] = float(
            current_ee[policy_ee.LEFT_GRIPPER] >= 0.5
        )
        current_ee[policy_ee.RIGHT_GRIPPER] = float(
            current_ee[policy_ee.RIGHT_GRIPPER] >= 0.5
        )
    repeated_targets = np.repeat(
        current_ee[None, :], args.rtc_prefix_len, axis=0
    ).astype(np.float32, copy=False)
    committed = _simulate_committed_commands(
        repeated_targets, current_ee, args
    )
    rtc_request = dict(request)
    rtc_request["remaining_action_chunk"] = _encode_prefix_for_server(
        committed,
        args.rtc_prefix_gripper_range,
    )
    rtc_request["prefix_len"] = args.rtc_prefix_len
    print("Warming up RTC prefix inference variant...")
    policy.infer(rtc_request)
    print("Policy warmup complete.")
