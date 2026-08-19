"""pi0.5 G1 桌面关节策略的异步 RTC 推理入口。

文件职责：把 ``policy_joint`` 的统一 16 维关节合同交给
``policy_joint_rtc`` 双线程运行时，实现推理与固定频率执行并行、低水位补充
chunk、已承诺前缀硬锁和延迟对齐。请求、响应、RTC prefix 与机器人命令始终
采用 ``[L7,Lhand,R7,Rhand]``，避免手部维度错位。Prompt 可由 Shell/CLI
覆盖或通过完整目录选择，但必须确认后才 reset、warmup 和启动 worker；本入口
不录像，适合实时执行与时序调试。公共 WebSocket 客户端负责本地转发慢启动
等待和运行中有限重连；RTC 停止时会取消阻塞的 recv，确保下一 episode 不会
与遗留线程并发读取同一连接。
"""

from __future__ import annotations

from action_logging import ActionLogSession
import policy_joint
from inference_common import (
    create_robot_environment,
    create_websocket_policy_from_args,
)
from policy_joint_rtc import (
    JointRTCArgs,
    RTCRunner,
    validate_joint_rtc_args,
    warmup_policy,
)
from policy_prompts import select_and_confirm_prompt


def main(args: JointRTCArgs) -> None:
    validate_joint_rtc_args(args)
    action_log = ActionLogSession(
        mode="joint", entrypoint="inference_pi05_g1_desk_joint_rtc"
    )
    action_log.update_config(args)
    trajectory_time = args.trajectory_time or (1.1 / args.control_frequency)

    print("=" * 72)
    print("pi0.5 G1 desk asynchronous joint RTC client")
    print(f"  server: {args.remote_host}:{args.remote_port}")
    print(f"  task_name: {args.task_name}")
    print(f"  horizon: {args.action_horizon}; chunk_skip: 0")
    print(
        f"  RTC prefix_len: {args.rtc_prefix_len}; refill_at: {args.rtc_refill_at}"
    )
    print(f"  control frequency: {args.control_frequency:g} Hz")
    print(f"  max joint delta/tick: {args.max_joint_delta}")
    print(f"  gripper threshold (0=open, 1=closed): >{args.gripper_threshold:g}")
    print("  joint layout: [left7, left_hand, right7, right_hand] = 16D")
    print("=" * 72)

    env = None
    policy = None
    try:
        print("Connecting RTC policy server (waiting for tunnel/model if needed)...")
        policy = action_log.wrap_policy(create_websocket_policy_from_args(args))
        print("Connected.")

        if not args.use_fake_obs:
            env = create_robot_environment(
                control_mode="joint_position",
                hz=args.control_frequency,
                use_smooth_trajectory=args.use_smooth_trajectory,
                trajectory_time=trajectory_time,
            )

        warmed_up = False

        while True:
            prompt = select_and_confirm_prompt(
                "joint",
                supplied_prompt=args.prompt,
                prompt_key=args.prompt_key,
            )
            if prompt == "q":
                break
            action_log.begin_episode(prompt)

            # Prompt confirmation happens before reset, warmup and RTC startup.
            if env is not None:
                print(f"Resetting robot to initial pose: {args.task_name}")
                initial_obs = env.reset(task_name=args.task_name)
                print("Initial-pose reset complete.")
            else:
                initial_obs = policy_joint.fake_obs()

            if not warmed_up:
                warmup_policy(args, policy, initial_obs)
                warmed_up = True
                # JAX compilation may take long enough for feedback to change.
                # Re-snapshot the command baseline before starting workers.
                initial_obs = (
                    env.get_obs() if env is not None else policy_joint.fake_obs()
                )

            try:
                RTCRunner(
                    args,
                    env,
                    policy,
                    prompt,
                    initial_obs,
                    action_trace_callback=lambda model, actual, step, raw: (
                        action_log.log_action(
                            model,
                            actual,
                            raw_model_action=raw,
                            episode_step=step,
                            source="joint_rtc_effective_target",
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

    main(tyro.cli(JointRTCArgs))
