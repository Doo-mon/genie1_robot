"""面向统一 robot/model 顺序关节命令的异步 RTC 运行时。

队列、末次命令、承诺前缀模拟与硬锁全部保持 ``[L7, Lhand, R7, Rhand]`` 16 维顺序。
推理线程负责补充和按已执行步数对齐 ``H x 16`` 动作块，执行线程按固定频率下发。
本模块依赖 ``policy_joint``、NumPy 与请求摘要工具；环境、policy 和 callback 由外部注入。
joint RTC 与 joint rollout 两个入口复用 ``JointRTCArgs``、``RTCRunner`` 和预热流程。
前缀先按限幅/夹爪规则顺序模拟，再锁入新响应，避免承诺轨迹与实际命令不一致。
所有共享时间线状态均受锁保护，回调只接收安全处理后的命令副本，异常会终止双线程。
正常结束时会取消阻塞中的策略请求，使下一 episode 从一条新连接开始。
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Callable, Literal

import numpy as np

import policy_joint
from inference_common import (
    PolicyClientClosedError,
    format_binary_gripper_transitions,
    print_policy_request,
)


PrefixSpace = Literal["openpi", "robot"]


def _tqdm():
    """Load progress reporting only when an RTC episode is actually run."""

    try:
        import tqdm
    except ModuleNotFoundError as error:
        raise RuntimeError("Joint RTC progress reporting requires tqdm.") from error
    return tqdm


def _checked_robot_chunk(value, *, name: str) -> np.ndarray:
    chunk = np.asarray(value, dtype=np.float32)
    if chunk.ndim == 1 and len(chunk) == 0:
        chunk = chunk.reshape(0, policy_joint.JOINT_DIM)
    if chunk.ndim != 2 or chunk.shape[1] != policy_joint.JOINT_DIM:
        raise ValueError(
            f"Expected {name} shape (H, {policy_joint.JOINT_DIM}), "
            f"got {chunk.shape}"
        )
    if not np.all(np.isfinite(chunk)):
        raise ValueError(f"{name} contains NaN or Inf")
    return chunk.copy()


def _aligned_replacement(
    actions: np.ndarray, elapsed_pops: int
) -> tuple[int, list[np.ndarray]]:
    """Discard response rows whose corresponding commands already executed."""

    chunk = _checked_robot_chunk(actions, name="robot action chunk")
    use_index = min(max(0, int(elapsed_pops)), len(chunk))
    return use_index, [action.copy() for action in chunk[use_index:]]


def _hard_lock_prefix(
    actions: np.ndarray,
    committed_commands: np.ndarray,
    lock_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Overwrite a response prefix with exact committed robot commands."""

    result = _checked_robot_chunk(actions, name="robot action chunk")
    committed = _checked_robot_chunk(
        committed_commands, name="committed robot command chunk"
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
    args: "JointRTCArgs",
) -> np.ndarray:
    """Simulate the exact safety-processed commands pending on the timeline.

    ``remaining_targets`` and the result are both in robot order.  Processing is
    sequential because each delta limit is relative to the command immediately
    before it, not to the observation captured for the inference request.
    """

    targets = _checked_robot_chunk(
        remaining_targets, name="remaining robot target chunk"
    )
    previous = policy_joint.checked_joint_vector(
        last_command, name="last executed robot command"
    )
    committed: list[np.ndarray] = []
    for target in targets:
        previous = policy_joint.process_robot_action(
            target,
            previous,
            gripper_threshold=args.gripper_threshold,
            max_joint_delta=args.max_joint_delta,
        )
        committed.append(previous.copy())

    if not committed:
        return np.empty((0, policy_joint.JOINT_DIM), dtype=np.float32)
    return np.stack(committed, axis=0).astype(np.float32, copy=False)


def _encode_prefix_chunk(
    committed_robot_commands,
    prefix_space: PrefixSpace,
) -> np.ndarray:
    """Return the unified-order prefix for either legacy space label.

    ``rtc_prefix_space`` remains as a compatibility flag for launchers and
    older configs.  Both accepted values now describe the same physical/model
    layout and must therefore be identity operations.
    """

    committed = _checked_robot_chunk(
        committed_robot_commands, name="committed robot command chunk"
    )
    if prefix_space in ("openpi", "robot"):
        return committed
    raise ValueError(
        "rtc_prefix_space must be either 'openpi' or 'robot', "
        f"got {prefix_space!r}"
    )


@dataclasses.dataclass
class JointRTCArgs(policy_joint.JointArgs):
    # RTC consumes the complete server horizon.  Prefix skipping or a bounded
    # execution window would shift the physical timeline used for alignment.
    execution_horizon: int = dataclasses.field(init=False, default=0)
    chunk_skip: int = 0

    rtc_prefix_len: int = 8
    rtc_refill_at: int = 12
    rtc_prefix_space: PrefixSpace = "openpi"


def validate_joint_rtc_args(args: JointRTCArgs) -> None:
    """Validate base joint settings plus RTC timeline constraints."""

    # Reuse the base validation without weakening its synchronous
    # execution_horizon invariant.  RTC's zero value means "consume the whole
    # asynchronously replenished timeline" and is intentionally init=False.
    base_kwargs = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(policy_joint.JointArgs)
        if field.init
    }
    base_kwargs["execution_horizon"] = 1
    base_kwargs["chunk_skip"] = 0
    policy_joint.validate_joint_args(policy_joint.JointArgs(**base_kwargs))

    if args.execution_horizon != 0:
        raise ValueError("Joint RTC requires execution_horizon=0")
    if args.chunk_skip != 0:
        raise ValueError("Joint RTC timeline alignment requires chunk_skip=0")
    if args.use_smooth_trajectory:
        raise ValueError(
            "Joint RTC requires direct fixed-rate commands; "
            "use_smooth_trajectory is only supported by the synchronous entry"
        )
    if args.action_horizon <= 1:
        raise ValueError("action_horizon must be greater than 1 for joint RTC")
    if not 1 <= args.rtc_prefix_len < args.action_horizon:
        raise ValueError(
            f"rtc_prefix_len must be in [1, {args.action_horizon - 1}]"
        )
    if not args.rtc_prefix_len <= args.rtc_refill_at < args.action_horizon:
        raise ValueError(
            "rtc_refill_at must be >= rtc_prefix_len and smaller than "
            "action_horizon"
        )
    if args.rtc_prefix_space not in ("openpi", "robot"):
        raise ValueError(
            "rtc_prefix_space must be either 'openpi' or 'robot', "
            f"got {args.rtc_prefix_space!r}"
        )


class RTCRunner:
    """Run inference and fixed-rate joint execution on two worker threads."""

    def __init__(
        self,
        args: JointRTCArgs,
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
        # Keep diagnostics independent of rollout recording: the trace gets
        # both the selected 16-D target and the successfully applied command.
        self.action_trace_callback = action_trace_callback
        self.period = 1.0 / args.control_frequency

        self._lock = threading.Lock()
        # Serialize one physical observation+timeline snapshot against a
        # dequeue+env.step transition.  Network inference never holds this gate.
        self._observation_gate = threading.Lock()
        self._stop = threading.Event()
        # Every item and state vector below is robot order R.
        self._queue: list[np.ndarray] = []
        self._raw_queue: list[np.ndarray] = []
        self._last_command = policy_joint.checked_joint_vector(
            initial_obs["state/joint"], name="initial robot joint command"
        )
        self._pop_count = 0
        self._steps_done = 0
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
        """Infer an R-space chunk and return it with its committed R prefix."""

        request = policy_joint.prepare_policy_request(
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
                (0, policy_joint.JOINT_DIM), dtype=np.float32
            )

        committed = _simulate_committed_commands(
            remaining_chunk, last_command, self.args
        )
        lock_len = min(max(0, int(lock_len)), len(committed))
        if lock_len > 0:
            request["remaining_action_chunk"] = _encode_prefix_chunk(
                committed, self.args.rtc_prefix_space
            )
            request["prefix_len"] = lock_len

        if not self._printed_first_request:
            print_policy_request(request)
            self._printed_first_request = True

        if self._stop.is_set():
            raise PolicyClientClosedError(
                "Joint RTC stopped before starting the inference request"
            )
        start = time.perf_counter()
        response = self.policy.infer(request)
        infer_ms = (time.perf_counter() - start) * 1000.0
        # Policy and robot responses share [L7, Lhand, R7, Rhand].  The policy
        # helper validates that contract; the runtime then stays in R space.
        actions = policy_joint.extract_robot_actions(
            response, max_horizon=self.args.action_horizon
        )
        raw_actions = actions.copy()
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
                f"lock_len={lock_len} prefix_space={self.args.rtc_prefix_space} "
                f"chunk={len(actions)}{diagnostic}"
            )
        return actions, committed, raw_actions

    def _inference_loop(self) -> None:
        try:
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
                if queue_len > self.args.rtc_refill_at:
                    time.sleep(self.period / 2.0)
                    continue

                # The observation and committed prefix must refer to the same
                # physical epoch.  Execution takes this gate through env.step.
                with self._observation_gate:
                    obs = (
                        self.env.get_obs()
                        if self.env is not None
                        else policy_joint.fake_obs()
                    )
                    with self._lock:
                        # Stop may have been requested while get_obs() blocked.
                        if self._stop.is_set():
                            return
                        remaining = [action.copy() for action in self._queue]
                        request_pop_count = self._pop_count
                        request_last_command = self._last_command.copy()
                lock_len = min(self.args.rtc_prefix_len, len(remaining))
                actions, committed, raw_actions = self._request(
                    obs, remaining, request_last_command, lock_len
                )

                with self._lock:
                    elapsed_pops = max(0, self._pop_count - request_pop_count)
                    use_index, replacement = _aligned_replacement(
                        actions, elapsed_pops
                    )
                    _, raw_replacement = _aligned_replacement(
                        raw_actions, elapsed_pops
                    )
                    handoff_rejected = (
                        lock_len > 0 and elapsed_pops > lock_len
                    )
                    if handoff_rejected:
                        self._handoff_rejections += 1
                    if not self._stop.is_set() and not handoff_rejected:
                        self._queue = replacement
                        self._raw_queue = raw_replacement

                if handoff_rejected:
                    logging.warning(
                        "Rejecting joint RTC response beyond committed prefix: "
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
                        f"[joint-RTC-handoff] request_remaining={len(remaining)} "
                        f"lock_len={lock_len} elapsed_pops={elapsed_pops} "
                        f"use_index={use_index} replacement_len={len(replacement)} "
                        f"accepted={not handoff_rejected} "
                        f"committed_l2={committed_l2}"
                    )
        except PolicyClientClosedError as error:
            # 正常结束时 run 会关闭当前 recv，避免遗留线程继续读取共享 socket。
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
                            target = self._queue.pop(0)
                            raw_model_target = (
                                self._raw_queue.pop(0)
                                if self._raw_queue
                                else target.copy()
                            )
                            model_target = target.copy()
                            previous_command = self._last_command.copy()
                            # The gate extends this atomic state transition
                            # through the corresponding hardware publication.
                            command = policy_joint.process_robot_action(
                                target,
                                self._last_command,
                                gripper_threshold=self.args.gripper_threshold,
                                max_joint_delta=self.args.max_joint_delta,
                            )
                            transition_messages = (
                                format_binary_gripper_transitions(
                                    target,
                                    previous_command,
                                    command,
                                    grippers=(
                                        ("left", policy_joint.ROBOT_LEFT_GRIPPER),
                                        ("right", policy_joint.ROBOT_RIGHT_GRIPPER),
                                    ),
                                    prefix="joint RTC gripper",
                                    step=self._steps_done + 1,
                                )
                            )
                            self._last_command = command.copy()
                            self._pop_count += 1
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
                            "[joint RTC] action queue starved; holding last command"
                        )
                    starving = True
                else:
                    if self.args.debug and starving:
                        _tqdm().tqdm.write("[joint RTC] action queue recovered")
                    starving = False
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
                    # Hardware and user callbacks must never hold the timeline
                    # lock.  Both receive the actual safety-processed R command.
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
                            "[joint RTC] control overruns: "
                            f"count={self._overrun_count} "
                            f"mean={self._overrun_total_ms / self._overrun_count:.1f} ms "
                            f"max={self._overrun_max_ms:.1f} ms"
                        )
            with self._lock:
                self._stop.set()
        except BaseException as error:
            self._fail(error)

    def run(self) -> None:
        validate_joint_rtc_args(self.args)
        inference_thread = threading.Thread(
            target=self._inference_loop,
            name="joint-rtc-inference",
            daemon=True,
        )
        execution_thread = threading.Thread(
            target=self._execution_loop,
            name="joint-rtc-execution",
            daemon=True,
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
                # This intentionally replaces an in-flight KeyboardInterrupt:
                # the caller must exit rather than reuse a policy while an old
                # RTC worker may still hold its I/O lock or hardware callback.
                raise RuntimeError(
                    "RTC worker threads did not stop: " + ", ".join(lingering)
                )

        with self._lock:
            error = self._error
            steps_done = self._steps_done
            pending = len(self._queue)
            handoff_rejections = self._handoff_rejections
            overrun_count = self._overrun_count
            overrun_total_ms = self._overrun_total_ms
            overrun_max_ms = self._overrun_max_ms
        if error is not None:
            raise RuntimeError("Joint RTC worker failed") from error
        _tqdm().tqdm.write(
            f"[joint RTC] episode complete: steps={steps_done}, pending={pending}, "
            f"late_responses_rejected={handoff_rejections}, "
            f"overruns={overrun_count}, "
            f"overrun_mean_ms="
            f"{(overrun_total_ms / overrun_count if overrun_count else 0.0):.1f}, "
            f"overrun_max_ms={overrun_max_ms:.1f}"
        )


def warmup_policy(args: JointRTCArgs, policy, obs: dict) -> None:
    """Compile the no-prefix and configured-prefix inference variants."""

    request = policy_joint.prepare_policy_request(
        obs,
        "warmup",
        resize_images=args.resize_images,
        image_size=args.image_size,
    )
    print("Warming up no-prefix joint inference variant...")
    policy.infer(dict(request))

    current_robot = policy_joint.checked_joint_vector(
        obs["state/joint"], name="warmup robot joint command"
    )
    repeated_targets = np.repeat(
        current_robot[None, :], args.rtc_prefix_len, axis=0
    ).astype(np.float32, copy=False)
    committed = _simulate_committed_commands(
        repeated_targets, current_robot, args
    )
    rtc_request = dict(request)
    rtc_request["remaining_action_chunk"] = _encode_prefix_chunk(
        committed, args.rtc_prefix_space
    )
    rtc_request["prefix_len"] = args.rtc_prefix_len
    print(
        "Warming up joint RTC prefix inference variant "
        f"({args.rtc_prefix_space} space)..."
    )
    policy.infer(rtc_request)
    print("Joint policy warmup complete.")
