"""pi0.5 G1 桌面 EE 策略的异步 RTC 推理入口。

文件职责：组合 ``policy_ee`` 的 EE 数据契约与 ``policy_ee_rtc`` 的双线程
运行时，使推理线程可在执行线程消费旧 chunk 时提前请求并替换后续轨迹。
RTC 会先模拟位置/旋转限幅和可选夹爪二值化后的真实命令，再发送并硬锁这条承诺
前缀，维持模型条件与实际下发时间线一致。Prompt 可由 Shell/CLI 覆盖或交互
选择，但必须在 reset、warmup 和 RTC worker 启动前确认。本文件不包含录像逻辑，适合只执行
实时策略、不采集 LeRobot rollout 的场景。连接阶段先等待本地端口转发和远端
模型；运行中传输失败默认终止，只有显式配置正数重试次数才重连。RTC 结束会
主动打断未完成的网络读取，避免旧 episode 的线程继续占用同一 WebSocket。
"""

from __future__ import annotations

from action_logging import ActionLogSession
import policy_ee
from policy_ee_rtc import (
    RTCArgs,
    RTCRunner,
    resolve_freeze_arm,
    validate_rtc_args,
    warmup_policy,
)
from inference_common import (
    create_robot_environment,
    create_websocket_policy_from_args,
)
from policy_prompts import select_and_confirm_prompt


def main(args: RTCArgs) -> None:
    validate_rtc_args(args)
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
    action_log = ActionLogSession(
        mode="ee", entrypoint="inference_pi05_g1_desk_ee_rtc"
    )
    action_log.update_config(args)
    close_threshold, open_threshold = policy_ee.resolve_gripper_thresholds(
        gripper_threshold=args.gripper_threshold,
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
    )
    trajectory_time = args.trajectory_time or (1.1 / args.control_frequency)
    freeze_arm = resolve_freeze_arm(args)

    print("=" * 72)
    print("pi05 G1 desk asynchronous EE RTC client")
    print(f"  server: {args.remote_host}:{args.remote_port}")
    print(f"  task_name: {args.task_name}")
    print(f"  horizon: {args.action_horizon}; chunk_skip: 0")
    print(
        f"  RTC prefix_len: {args.rtc_prefix_len}; refill_at: {args.rtc_refill_at}"
    )
    print(f"  RTC prefix gripper range: {args.rtc_prefix_gripper_range}")
    print(f"  binarize gripper action: {args.binarize_gripper}")
    print(
        "  synchronize gripper with physical EE feedback: "
        f"{args.sync_gripper_to_arm} "
        f"(pos_tol={args.gripper_sync_position_tolerance:g}m, "
        f"rot_tol={args.gripper_sync_rotation_tolerance:g}rad, "
        f"stable_frames={args.gripper_sync_stable_frames}, "
        f"gripper_hold={args.gripper_settle_time:g}s, "
        f"min_timeout={args.gripper_sync_timeout:g}s, "
        f"max_timeout={args.gripper_sync_max_timeout:g}s, "
        f"replan={args.replan_after_gripper_sync})"
    )
    if args.binarize_gripper:
        if args.gripper_threshold is not None:
            print(f"  gripper legacy single threshold: >{close_threshold:g}")
        else:
            print(
                "  gripper hysteresis (0=open, 1=closed): "
                f"close>={close_threshold:g}, open<={open_threshold:g}"
            )
    print(f"  control frequency: {args.control_frequency:g} Hz")
    print(
        f"  motion limits/tick: position={args.max_position_delta}, "
        f"rotation={args.max_rotation_delta}"
    )
    print(f"  frozen EE/hardware arm at episode start: {freeze_arm}")
    print(
        "  base_link EE z floors: "
        f"left={'disabled' if left_z_floor is None else f'{left_z_floor:.4f} m'}, "
        f"right={'disabled' if right_z_floor is None else f'{right_z_floor:.4f} m'} "
        "(clamp before SDK)"
    )
    print("=" * 72)

    env = None
    policy = None
    try:
        print("Connecting RTC policy server (waiting for tunnel/model if needed)...")
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

        warmed_up = False

        while True:
            prompt = select_and_confirm_prompt(
                "ee",
                supplied_prompt=args.prompt,
                prompt_key=args.prompt_key,
            )
            if prompt == "q":
                break
            action_log.begin_episode(prompt)

            # Prompt confirmation happens before reset or RTC worker startup.
            if env is not None:
                print(f"Resetting robot to initial pose: {args.task_name}")
                initial_obs = env.reset(task_name=args.task_name)
                print("Initial-pose reset complete.")
            else:
                initial_obs = policy_ee.fake_obs()

            if not warmed_up:
                warmup_policy(args, policy, initial_obs)
                warmed_up = True
                # 首次 JAX warmup 可能很久；执行前重新读取姿态，避免用陈旧
                # reset 观测初始化 last-command 与 freeze anchor。
                initial_obs = (
                    env.get_obs() if env is not None else policy_ee.fake_obs()
                )

            try:
                RTCRunner(
                    args,
                    env,
                    policy,
                    prompt,
                    initial_obs,
                    action_trace_callback=lambda model, actual, step, raw, feedback, feedback_ts: (
                        action_log.log_action(
                            model,
                            actual,
                            raw_model_action=raw,
                            episode_step=step,
                            source="ee_rtc_effective_target",
                            environment=env,
                            feedback_sample=(
                                (feedback, feedback_ts)
                                if env is not None and feedback is not None
                                else None
                            ),
                        )
                    ),
                ).run()
            except KeyboardInterrupt:
                print("\nEpisode interrupted.")
                action_log.end_episode("keyboard_interrupt")
            else:
                action_log.end_episode("completed")

            answer = input("\nContinue with another task? (y/n): ").strip().lower()
            if answer not in {"y", "yes"}:
                break
    finally:
        action_log.close()
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(RTCArgs))
