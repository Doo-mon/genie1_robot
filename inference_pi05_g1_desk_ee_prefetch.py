"""pi0.5 G1 同步 EE 重采样 + 异步预取的独立推理入口。

本入口保留同步模式的核心语义：一个模型 chunk 会先完整重采样，当前队列的
每条 command 都按原顺序下发，后台响应绝不会替换或截断它。当前队列缩短到
``prefetch_prefix_len`` 后，控制线程读取一次对齐观测，把当时所有尚未执行的
EE14 command 作为 0--1 prefix；网络 infer 在 daemon 工作线程中运行。当前
队列完全结束后才接入响应中 prefix 之后的未来动作并重新重采样。

该模式用于减少普通同步入口在 chunk 边界等待模型的停顿，但不采用 RTC 的
持续队列替换、elapsed-pop 对齐或迟到响应接入逻辑。若预取仍未及时完成，边界
会明确等待；安全上宁可停一下，也不改变第一次观测生成的当前轨迹。Prompt、
动作日志、逐维绘图、rollout、单臂冻结和左右末端最低高度保护均沿用现有公共
实现。原同步与 RTC 文件不会导入本入口，便于独立真机对照和回退。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import numpy as np

from action_logging import ActionLogSession
from inference_common import (
    create_robot_environment,
    create_websocket_policy_from_args,
    format_binary_gripper_transitions,
    print_policy_request,
)
import policy_ee
from policy_ee_prefetch import (
    AsyncEEPrefetch,
    EEPrefetchArgs,
    PrefetchedEEWindow,
    validate_prefetch_args,
)
from policy_prompts import select_and_confirm_prompt
from rollout_recorder import RolloutManager
from rollout_schemas import EE_PREFETCH_ROLLOUT_SCHEMA


WORKSPACE_DIR = Path(__file__).resolve().parent


def _retime_queues(
    raw_actions: np.ndarray,
    effective_actions: np.ndarray,
    previous_command: np.ndarray,
    args: EEPrefetchArgs,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int]:
    """重采样一个窗口并建立 raw/effective/command 的逐命令对应关系。"""

    raw = np.asarray(raw_actions, dtype=np.float32)
    effective = np.asarray(effective_actions, dtype=np.float32)
    if raw.shape != effective.shape or raw.ndim != 2 or raw.shape[1] != policy_ee.EE_DIM:
        raise ValueError(
            "Raw/effective prefetch windows must share shape (H, 14), got "
            f"{raw.shape} and {effective.shape}"
        )
    retimed = policy_ee.retime_ee_action_window(
        effective,
        previous_command,
        max_position_delta=args.max_position_delta,
        max_rotation_delta=args.max_rotation_delta,
        binarize_gripper=args.binarize_gripper,
        gripper_threshold=args.gripper_threshold,
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
        max_output_steps=args.max_retimed_chunk_steps,
    )
    commands = [command.copy() for command in retimed.commands]
    model_queue = [
        effective[int(source_index)].copy()
        for source_index in retimed.source_indices
    ]
    raw_queue = [
        raw[int(source_index)].copy() for source_index in retimed.source_indices
    ]
    max_segment_steps = int(retimed.segment_steps.max())
    return commands, model_queue, raw_queue, max_segment_steps


def _initial_window(
    *,
    policy,
    obs: dict,
    prompt: str,
    args: EEPrefetchArgs,
    freeze_arm: str,
    frozen_arm_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """同步获取 episode 第一块；它是唯一不带 prefix 的模型请求。"""

    request = policy_ee.prepare_policy_request(
        obs,
        prompt,
        resize_images=args.resize_images,
        image_size=args.image_size,
    )
    start = time.perf_counter()
    response = policy.infer(request)
    infer_ms = (time.perf_counter() - start) * 1000.0
    actions = policy_ee.extract_ee_actions(
        response,
        max_horizon=args.action_horizon,
    )
    if len(actions) != args.action_horizon:
        raise ValueError(
            f"Server returned horizon {len(actions)}, but prefetch expects "
            f"exactly {args.action_horizon}"
        )
    raw_actions = actions.copy()
    if freeze_arm != "none":
        actions = policy_ee.freeze_ee_arm(
            actions,
            frozen_arm_target,
            arm=freeze_arm,
        )
    effective = np.stack(
        policy_ee.select_action_window(
            actions,
            skip=args.chunk_skip,
            count=args.execution_horizon,
        ),
        axis=0,
    )
    raw = np.stack(
        policy_ee.select_action_window(
            raw_actions,
            skip=args.chunk_skip,
            count=args.execution_horizon,
        ),
        axis=0,
    )
    return raw, effective, infer_ms, request


def main(args: EEPrefetchArgs) -> None:
    import tqdm

    action_log = ActionLogSession(
        mode="ee",
        entrypoint="inference_pi05_g1_desk_ee_prefetch",
    )
    try:
        freeze_arm = validate_prefetch_args(args)
    except BaseException:
        action_log.close("validation_error")
        raise

    if args.rollout_output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.rollout_output_dir = str(
            WORKSPACE_DIR
            / "rollout_datasets"
            / f"ee_rollout_sync_prefetch_{stamp}"
        )
    action_log.update_config(args)
    try:
        rollout = RolloutManager(
            EE_PREFETCH_ROLLOUT_SCHEMA,
            Path(args.rollout_output_dir).resolve(),
            args.rollout_fps,
            args.rollout_enabled,
            0,
        )
    except BaseException:
        action_log.close("rollout_startup_error")
        raise

    left_z_floor = (
        args.min_ee_base_z
        if args.min_left_ee_base_z is None
        else args.min_left_ee_base_z
    )
    right_z_floor = (
        args.min_ee_base_z
        if args.min_right_ee_base_z is None
        else args.min_right_ee_base_z
    )
    control_period = 1.0 / args.control_frequency
    trajectory_time = args.trajectory_time or control_period * 1.1

    print("=" * 72)
    print("pi0.5 G1 desk EE synchronous retiming + asynchronous prefetch")
    print(f"  server: {args.remote_host}:{args.remote_port}")
    print(f"  task_name: {args.task_name}")
    print(f"  model horizon / initial rows: {args.action_horizon} / {args.execution_horizon}")
    print(f"  control frequency: {args.control_frequency:g} Hz")
    print(f"  trajectory time: {trajectory_time:.4f} s")
    print(
        "  retiming limits: "
        f"position={args.max_position_delta} m/tick, "
        f"rotation={args.max_rotation_delta} rad/tick, "
        f"max_chunk_steps={args.max_retimed_chunk_steps}"
    )
    print(
        f"  prefetch: start when remaining commands <= {args.prefetch_prefix_len}; "
        "prefix grippers fixed to action space 0--1"
    )
    print(
        "  server requirement: serve_policy must enable RTC prefix guidance; "
        "current metadata does not expose this flag for automatic verification"
    )
    print("  handoff: current queue must finish; only response tail becomes next queue")
    print(f"  frozen EE arm at episode start: {freeze_arm}")
    print(
        "  base_link EE z floors: "
        f"left={'disabled' if left_z_floor is None else f'{left_z_floor:.4f} m'}, "
        f"right={'disabled' if right_z_floor is None else f'{right_z_floor:.4f} m'}"
    )
    print(
        f"  rollout: enabled={rollout.enabled}, fps={rollout.fps:g}, "
        f"dir={rollout.output_root}"
    )
    print("=" * 72)

    env = None
    policy = None
    active_prefetch: AsyncEEPrefetch | None = None
    try:
        print("Connecting policy server (waiting for tunnel/model if needed)...")
        policy = action_log.wrap_policy(create_websocket_policy_from_args(args))
        print("Connected.")

        if not args.use_fake_obs:
            env = create_robot_environment(
                control_mode="camera_position",
                hz=args.control_frequency,
                trajectory_time=trajectory_time,
                freeze_arm=freeze_arm,
                min_ee_base_z=args.min_ee_base_z,
                min_left_ee_base_z=args.min_left_ee_base_z,
                min_right_ee_base_z=args.min_right_ee_base_z,
            )

        while True:
            prompt = select_and_confirm_prompt(
                "ee",
                supplied_prompt=args.prompt,
                prompt_key=args.prompt_key,
            )
            if prompt == "q":
                break
            action_log.begin_episode(prompt)
            episode_reason = "failed"
            active_prefetch = None
            try:
                if env is not None:
                    print(f"Resetting robot to initial pose: {args.task_name}")
                    obs = env.reset(task_name=args.task_name)
                    print("Initial-pose reset complete.")
                else:
                    obs = policy_ee.fake_obs()

                previous_command = policy_ee.state_ee_to_action_command(
                    obs["state/pose"]
                )
                frozen_arm_target = previous_command.copy()
                active_prefetch = AsyncEEPrefetch(
                    policy=policy,
                    args=args,
                    prompt=prompt,
                    freeze_arm=freeze_arm,
                    frozen_arm_target=frozen_arm_target,
                )
                obs_provider = (
                    env.get_obs_with_last_action
                    if env is not None
                    else policy_ee.fake_obs
                )
                rollout.begin_episode(prompt, obs_provider)

                raw_window, effective_window, infer_ms, first_request = _initial_window(
                    policy=policy,
                    obs=obs,
                    prompt=prompt,
                    args=args,
                    freeze_arm=freeze_arm,
                    frozen_arm_target=frozen_arm_target,
                )
                print_policy_request(first_request)
                action_queue, model_queue, raw_queue, max_segment = _retime_queues(
                    raw_window,
                    effective_window,
                    previous_command,
                    args,
                )
                tqdm.tqdm.write(
                    "[EE prefetch] initial "
                    f"model_rows={len(effective_window)} -> commands={len(action_queue)} "
                    f"({len(action_queue) / len(effective_window):.2f}x), "
                    f"max_segment_steps={max_segment}, infer_ms={infer_ms:.1f}, "
                    f"duration={len(action_queue) / args.control_frequency:.2f}s"
                )

                for step_index in tqdm.tqdm(
                    range(args.max_timesteps),
                    desc="running-prefetch",
                    unit="step",
                ):
                    if not action_queue:
                        if active_prefetch is None or not active_prefetch.active:
                            raise RuntimeError(
                                "Current EE queue ended without an active prefetch"
                            )
                        if not active_prefetch.ready:
                            tqdm.tqdm.write(
                                "[EE prefetch] current chunk complete; waiting for "
                                "the already-started future response"
                            )
                        future: PrefetchedEEWindow = active_prefetch.take()
                        action_queue, model_queue, raw_queue, max_segment = (
                            _retime_queues(
                                future.raw_actions,
                                future.effective_actions,
                                previous_command,
                                args,
                            )
                        )
                        tqdm.tqdm.write(
                            "[EE prefetch] handoff accepted after current chunk: "
                            f"locked_prefix={future.prefix_len}, "
                            f"future_rows={len(future.effective_actions)}, "
                            f"commands={len(action_queue)}, "
                            f"max_segment_steps={max_segment}, "
                            f"infer_ms={future.infer_ms:.1f}"
                        )

                    # Capture the observation before the first command in this
                    # exact prefix is submitted.  Network inference then runs
                    # asynchronously while the immutable queue is consumed.
                    if (
                        active_prefetch is not None
                        and not active_prefetch.active
                        and 1 <= len(action_queue) <= args.prefetch_prefix_len
                    ):
                        prefix_obs = (
                            env.get_obs() if env is not None else policy_ee.fake_obs()
                        )
                        prefix = np.stack(action_queue, axis=0)
                        prefix_len = active_prefetch.start(prefix_obs, prefix)
                        tqdm.tqdm.write(
                            "[EE prefetch] request started: "
                            f"committed_remaining_commands={prefix_len}, "
                            f"budget={prefix_len / args.control_frequency:.2f}s"
                        )

                    step_start = time.perf_counter()
                    raw_server_action = raw_queue.pop(0)
                    model_action = model_queue.pop(0)
                    action = policy_ee.checked_vector(
                        action_queue.pop(0),
                        dim=policy_ee.EE_DIM,
                        name="prefetched retimed EE command",
                    )
                    previous_before = previous_command.copy()
                    if args.binarize_gripper:
                        for message in format_binary_gripper_transitions(
                            model_action,
                            previous_before,
                            action,
                            grippers=(
                                ("left", policy_ee.LEFT_GRIPPER),
                                ("right", policy_ee.RIGHT_GRIPPER),
                            ),
                            prefix="EE prefetch gripper",
                            step=step_index + 1,
                        ):
                            tqdm.tqdm.write(message)
                    if env is not None:
                        env.step(action, update_obs=False)
                    rollout.set_last_action(action.copy())
                    action_log.log_action(
                        model_action,
                        action,
                        raw_model_action=raw_server_action,
                        episode_step=step_index + 1,
                        source="ee_sync_prefetch_retimed_command",
                        environment=env,
                    )
                    previous_command = action

                    elapsed = time.perf_counter() - step_start
                    if elapsed < control_period:
                        time.sleep(control_period - elapsed)
                    elif args.debug:
                        tqdm.tqdm.write(
                            f"control overrun: {elapsed * 1000.0:.1f} ms"
                        )
            except KeyboardInterrupt:
                print("\nEpisode interrupted.")
                episode_reason = "keyboard_interrupt"
            else:
                episode_reason = "completed"
            finally:
                try:
                    if active_prefetch is not None:
                        active_prefetch.cancel()
                finally:
                    try:
                        rollout.end_episode()
                    finally:
                        action_log.end_episode(episode_reason)
                active_prefetch = None

            answer = input("\nContinue with another task? (y/n): ").strip().lower()
            if answer not in {"y", "yes"}:
                break
    finally:
        try:
            if active_prefetch is not None:
                active_prefetch.cancel()
        finally:
            try:
                rollout.end_episode()
            finally:
                action_log.close()
                if policy is not None:
                    policy.close()
                if env is not None:
                    env.close()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(EEPrefetchArgs))


__all__ = ["EEPrefetchArgs", "main"]
