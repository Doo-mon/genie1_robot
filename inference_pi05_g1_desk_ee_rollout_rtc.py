"""带 LeRobot v2.1 采集的 pi0.5 G1 桌面 EE RTC 推理入口。

文件职责：在 ``policy_ee_rtc`` 异步执行基础上接入共享
``rollout_recorder``，按独立采样频率保存三路 RGB、30 维状态和实际发送的
EE 命令。录像 schema 由 ``rollout_schemas.EE_ROLLOUT_SCHEMA`` 定义；
OpenCV/PyArrow 只在启用录像时检查，并且会在连接策略或 reset 机器人前失败；
关闭 rollout 时不构成导入依赖。
RTC prefix 使用与录像 action 相同的限幅、可选二值化后命令，而不是原始模型目标。
Prompt 必须先完成覆盖/选择和人工确认，随后才 reset、warmup、创建 recorder
并启动 RTC，避免拒绝任务后仍产生机器人动作或空数据集。策略连接统一经过
可配置的慢启动等待；已发送请求默认不重发，只有重试次数大于 0 才断线重连。
关闭顺序为结束 episode、关闭 WebSocket、最后关闭机器人环境，防止残留线程。
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
import math
from pathlib import Path

from action_logging import ActionLogSession
import policy_ee
import policy_ee_rtc
from inference_common import (
    create_robot_environment,
    create_websocket_policy_from_args,
)
from policy_prompts import select_and_confirm_prompt
from rollout_recorder import EpisodeRecorder, RolloutManager
from rollout_schemas import EE_ROLLOUT_SCHEMA


WORKSPACE_DIR = Path(__file__).resolve().parent


@dataclasses.dataclass
class RolloutArgs(policy_ee_rtc.RTCArgs):
    rollout_output_dir: str | None = None
    rollout_fps: float = 30.0
    rollout_enabled: bool = True


def main(args: RolloutArgs) -> None:
    policy_ee_rtc.validate_rtc_args(args)
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
        mode="ee", entrypoint="inference_pi05_g1_desk_ee_rollout_rtc"
    )
    action_log.update_config(args)
    close_threshold, open_threshold = policy_ee.resolve_gripper_thresholds(
        gripper_threshold=args.gripper_threshold,
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
    )
    if not math.isfinite(args.rollout_fps) or args.rollout_fps <= 0:
        action_log.close("validation_error")
        raise ValueError("rollout_fps must be finite and positive")
    if args.rollout_enabled and abs(args.rollout_fps - args.control_frequency) > 1e-6:
        print(
            f"[ee-rollout] recording at {args.rollout_fps:g} Hz independently "
            f"of the {args.control_frequency:g} Hz control loop"
        )
    if args.rollout_output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.rollout_output_dir = str(
            WORKSPACE_DIR / "rollout_datasets" / f"ee_rollout_rtc_{stamp}"
        )
    action_log.update_config(args)

    try:
        rollout = RolloutManager(
            EE_ROLLOUT_SCHEMA,
            Path(args.rollout_output_dir).resolve(),
            args.rollout_fps,
            args.rollout_enabled,
            args.rtc_prefix_len,
        )
    except BaseException:
        action_log.close("rollout_startup_error")
        raise
    trajectory_time = args.trajectory_time or (1.1 / args.control_frequency)
    freeze_arm = policy_ee_rtc.resolve_freeze_arm(args)

    print("=" * 72)
    print("pi05 G1 desk EE RTC + rollout")
    print(f"  server: {args.remote_host}:{args.remote_port}")
    print(f"  model/action horizon: {args.action_horizon}")
    print(
        f"  RTC: prefix_len={args.rtc_prefix_len} refill_at={args.rtc_refill_at}"
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
    print(f"  rollout: enabled={rollout.enabled} dir={rollout.output_root}")
    print(f"  frozen EE/hardware arm at episode start: {freeze_arm}")
    print(
        "  base_link EE z floors: "
        f"left={'disabled' if left_z_floor is None else f'{left_z_floor:.4f} m'}, "
        f"right={'disabled' if right_z_floor is None else f'{right_z_floor:.4f} m'} "
        "(clamp before SDK)"
    )
    print("  saved state/action layout: [joint16, EE14] = 30D")
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

            # Confirmation precedes reset, warmup, worker startup and recording.
            if env is not None:
                print(f"Resetting robot to initial pose: {args.task_name}")
                initial_obs = env.reset(task_name=args.task_name)
                print("Initial-pose reset complete.")
            else:
                initial_obs = policy_ee.fake_obs()

            if not warmed_up:
                policy_ee_rtc.warmup_policy(args, policy, initial_obs)
                warmed_up = True
                # Warmup 之后重新建立本 episode 的命令基准；录制尚未开始，
                # 因此这次观测不会生成多余 rollout 帧。
                initial_obs = (
                    env.get_obs() if env is not None else policy_ee.fake_obs()
                )

            obs_provider = (
                env.get_obs_with_last_action
                if env is not None
                else policy_ee.fake_obs
            )
            rollout.begin_episode(prompt, obs_provider)
            episode_reason = "failed"
            try:
                policy_ee_rtc.RTCRunner(
                    args,
                    env,
                    policy,
                    prompt,
                    initial_obs,
                    action_callback=rollout.set_last_action,
                    action_trace_callback=lambda model, actual, step, raw, feedback, feedback_ts: (
                        action_log.log_action(
                            model,
                            actual,
                            raw_model_action=raw,
                            episode_step=step,
                            source="ee_rollout_rtc_effective_target",
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
                episode_reason = "keyboard_interrupt"
            else:
                episode_reason = "completed"
            finally:
                try:
                    rollout.end_episode()
                finally:
                    action_log.end_episode(episode_reason)

            answer = input("\nContinue with another task? (y/n): ").strip().lower()
            if answer not in {"y", "yes"}:
                break
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

    main(tyro.cli(RolloutArgs))


__all__ = ["EpisodeRecorder", "RolloutArgs", "main"]
