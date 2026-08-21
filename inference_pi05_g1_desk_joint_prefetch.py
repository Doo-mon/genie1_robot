"""pi0.5 G1 joint16 同步重采样 + 异步预取的独立入口。

首个模型 chunk 同步获取并按逐关节变化上限展开；当前命令队列始终完整执行。
队列尾部尚未下发的 joint16 command 会和同一时刻的新观测组成服务端 prefix，
网络请求在后台运行。响应的 prefix 部分不会再次执行，只有未来尾部会在当前
队列归零后重新插值接入。该模式不导入、不修改原 joint 同步或 joint RTC。

joint16 在机器人、客户端和当前 checkpoint 中均采用
``[left_arm7, left_hand, right_arm7, right_hand]``。插值只作用于14个手臂关节，
两个夹爪在对应模型 row 的最后一个子步切换。可选 rollout 使用独立 processing
mode；动作日志仍区分原始模型 target、有效 target 和成功下发 command。
"""

from __future__ import annotations

from datetime import datetime
import faulthandler
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
import policy_joint
from policy_joint_prefetch import (
    AsyncJointPrefetch,
    JointPrefetchArgs,
    PrefetchedJointWindow,
    retime_joint_action_window,
    validate_joint_prefetch_args,
)
from policy_prompts import select_and_confirm_prompt
from rollout_recorder import RolloutManager
from rollout_schemas import JOINT_PREFETCH_ROLLOUT_SCHEMA


WORKSPACE_DIR = Path(__file__).resolve().parent


def _retime_queues(
    actions: np.ndarray,
    current_command: np.ndarray,
    args: JointPrefetchArgs,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    chunk = policy_joint.openpi_chunk_to_robot_chunk(actions)
    retimed = retime_joint_action_window(
        chunk,
        current_command,
        max_joint_delta=args.max_joint_delta,
        gripper_threshold=args.gripper_threshold,
        max_output_steps=args.max_retimed_chunk_steps,
    )
    commands = [command.copy() for command in retimed.commands]
    targets = [
        chunk[int(source_index)].copy()
        for source_index in retimed.source_indices
    ]
    return commands, targets, int(retimed.segment_steps.max())


def _initial_window(
    *,
    policy,
    obs: dict,
    prompt: str,
    args: JointPrefetchArgs,
) -> tuple[np.ndarray, float, dict]:
    request = policy_joint.prepare_policy_request(
        obs,
        prompt,
        resize_images=args.resize_images,
        image_size=args.image_size,
    )
    start = time.perf_counter()
    response = policy.infer(request)
    infer_ms = (time.perf_counter() - start) * 1000.0
    actions = policy_joint.extract_robot_actions(
        response,
        max_horizon=args.action_horizon,
    )
    if len(actions) != args.action_horizon:
        raise ValueError(
            f"Server returned horizon {len(actions)}, but joint prefetch "
            f"expects exactly {args.action_horizon}"
        )
    window = np.stack(
        policy_joint.select_action_window(
            actions,
            skip=args.chunk_skip,
            count=args.execution_horizon,
        ),
        axis=0,
    )
    return window, infer_ms, request


def main(args: JointPrefetchArgs) -> None:
    import tqdm

    action_log = ActionLogSession(
        mode="joint",
        entrypoint="inference_pi05_g1_desk_joint_prefetch",
    )
    try:
        validate_joint_prefetch_args(args)
    except BaseException:
        action_log.close("validation_error")
        raise
    faulthandler.enable()

    if args.rollout_output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.rollout_output_dir = str(
            WORKSPACE_DIR
            / "rollout_datasets"
            / f"joint_rollout_sync_prefetch_{stamp}"
        )
    action_log.update_config(args)
    try:
        rollout = RolloutManager(
            JOINT_PREFETCH_ROLLOUT_SCHEMA,
            Path(args.rollout_output_dir).resolve(),
            args.rollout_fps,
            args.rollout_enabled,
            0,
        )
    except BaseException:
        action_log.close("rollout_startup_error")
        raise

    control_period = 1.0 / args.control_frequency
    trajectory_time = args.trajectory_time or control_period * 1.1

    print("=" * 72)
    print("pi0.5 G1 joint synchronous retiming + asynchronous prefetch")
    print(f"  server: {args.remote_host}:{args.remote_port}")
    print(f"  task_name: {args.task_name}")
    print(f"  model horizon / initial rows: {args.action_horizon} / {args.execution_horizon}")
    print(f"  control frequency: {args.control_frequency:g} Hz")
    print(f"  max joint delta/tick: {args.max_joint_delta}")
    print(f"  trajectory time: {trajectory_time:.4f} s")
    print(f"  prefetch remaining-command prefix: {args.prefetch_prefix_len}")
    print("  layout: [left_arm7, left_hand, right_arm7, right_hand] = 16D")
    print(
        "  server requirement: serve_policy must enable RTC prefix guidance; "
        "metadata does not expose this flag for automatic verification"
    )
    print(
        f"  rollout: enabled={rollout.enabled}, fps={rollout.fps:g}, "
        f"dir={rollout.output_root}"
    )
    print("=" * 72)

    env = None
    policy = None
    active_prefetch: AsyncJointPrefetch | None = None
    try:
        print("Connecting policy server (waiting for tunnel/model if needed)...")
        policy = action_log.wrap_policy(create_websocket_policy_from_args(args))
        print("Connected.")
        if not args.use_fake_obs:
            env = create_robot_environment(
                control_mode="joint_position",
                hz=args.control_frequency,
                use_smooth_trajectory=False,
                trajectory_time=trajectory_time,
            )

        while True:
            prompt = select_and_confirm_prompt(
                "joint",
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
                    obs = policy_joint.fake_obs()
                current_command = policy_joint.checked_joint_vector(
                    obs["state/joint"],
                    name="initial robot joint",
                )
                active_prefetch = AsyncJointPrefetch(
                    policy=policy,
                    args=args,
                    prompt=prompt,
                )
                obs_provider = (
                    env.get_obs_with_last_action
                    if env is not None
                    else policy_joint.fake_obs
                )
                rollout.begin_episode(prompt, obs_provider)

                initial, infer_ms, first_request = _initial_window(
                    policy=policy,
                    obs=obs,
                    prompt=prompt,
                    args=args,
                )
                print_policy_request(first_request)
                command_queue, target_queue, max_segment = _retime_queues(
                    initial,
                    current_command,
                    args,
                )
                tqdm.tqdm.write(
                    "[joint prefetch] initial "
                    f"model_rows={len(initial)} -> commands={len(command_queue)} "
                    f"({len(command_queue) / len(initial):.2f}x), "
                    f"max_segment_steps={max_segment}, infer_ms={infer_ms:.1f}"
                )

                for step_index in tqdm.tqdm(
                    range(args.max_timesteps),
                    desc="running-joint-prefetch",
                    unit="step",
                ):
                    if not command_queue:
                        if active_prefetch is None or not active_prefetch.active:
                            raise RuntimeError(
                                "Current joint queue ended without an active prefetch"
                            )
                        if not active_prefetch.ready:
                            tqdm.tqdm.write(
                                "[joint prefetch] current chunk complete; waiting "
                                "for the already-started future response"
                            )
                        future: PrefetchedJointWindow = active_prefetch.take()
                        command_queue, target_queue, max_segment = _retime_queues(
                            future.raw_actions,
                            current_command,
                            args,
                        )
                        tqdm.tqdm.write(
                            "[joint prefetch] handoff accepted after current chunk: "
                            f"locked_prefix={future.prefix_len}, "
                            f"future_rows={len(future.raw_actions)}, "
                            f"commands={len(command_queue)}, "
                            f"max_segment_steps={max_segment}, "
                            f"infer_ms={future.infer_ms:.1f}"
                        )

                    if (
                        active_prefetch is not None
                        and not active_prefetch.active
                        and 1 <= len(command_queue) <= args.prefetch_prefix_len
                    ):
                        prefix_obs = (
                            env.get_obs() if env is not None else policy_joint.fake_obs()
                        )
                        prefix = np.stack(command_queue, axis=0)
                        prefix_len = active_prefetch.start(prefix_obs, prefix)
                        tqdm.tqdm.write(
                            "[joint prefetch] request started: "
                            f"committed_remaining_commands={prefix_len}, "
                            f"budget={prefix_len / args.control_frequency:.2f}s"
                        )

                    step_start = time.perf_counter()
                    raw_target = target_queue.pop(0)
                    command = policy_joint.checked_joint_vector(
                        command_queue.pop(0),
                        name="prefetched retimed joint command",
                    )
                    previous_before = current_command.copy()
                    for message in format_binary_gripper_transitions(
                        raw_target,
                        previous_before,
                        command,
                        grippers=(
                            ("left", policy_joint.ROBOT_LEFT_GRIPPER),
                            ("right", policy_joint.ROBOT_RIGHT_GRIPPER),
                        ),
                        prefix="joint prefetch gripper",
                        step=step_index + 1,
                    ):
                        tqdm.tqdm.write(message)
                    if env is not None:
                        env.step(command, update_obs=False)
                    rollout.set_last_action(command.copy())
                    action_log.log_action(
                        raw_target,
                        command,
                        raw_model_action=raw_target,
                        episode_step=step_index + 1,
                        source="joint_sync_prefetch_retimed_command",
                        environment=env,
                    )
                    current_command = command

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

    main(tyro.cli(JointPrefetchArgs))


__all__ = ["JointPrefetchArgs", "main"]
