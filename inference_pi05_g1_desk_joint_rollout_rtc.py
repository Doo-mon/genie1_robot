"""带 LeRobot v2.1 采集的 pi0.5 G1 桌面 joint RTC 推理入口。

文件职责：组合 ``policy_joint_rtc`` 与共享 ``rollout_recorder``，并通过
``JOINT_ROLLOUT_SCHEMA`` 保存三路 RGB、16 维关节反馈和实际下发的 16 维
安全处理后命令。状态、网络动作、RTC prefix 与数据集 action 均遵循
``[left_arm7,left_hand1,right_arm7,right_hand1]``。Prompt 在 reset、warmup、
recorder 创建和 RTC worker 启动前确认；OpenCV/PyArrow 仅在启用录像时
检查，且会在连接策略/reset 真机前失败，普通导入和
``--no-rollout-enabled`` 不依赖它们。网络层
先等待本地端口转发和远端模型；已发送请求默认不重发，仅显式设置
重试次数才允许断线重连重放。退出时依次结束录制、关闭
策略 socket 和机器人环境，避免后台线程及半写入 episode 残留。
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
import math
from pathlib import Path

from action_logging import ActionLogSession
import policy_joint
import policy_joint_rtc
from inference_common import (
    create_robot_environment,
    create_websocket_policy_from_args,
)
from policy_prompts import select_and_confirm_prompt
from rollout_recorder import EpisodeRecorder, RolloutManager
from rollout_schemas import JOINT_ROLLOUT_SCHEMA


WORKSPACE_DIR = Path(__file__).resolve().parent


@dataclasses.dataclass
class JointRolloutArgs(policy_joint_rtc.JointRTCArgs):
    rollout_output_dir: str | None = None
    rollout_fps: float = 30.0
    rollout_enabled: bool = True


def main(args: JointRolloutArgs) -> None:
    policy_joint_rtc.validate_joint_rtc_args(args)
    action_log = ActionLogSession(
        mode="joint", entrypoint="inference_pi05_g1_desk_joint_rollout_rtc"
    )
    action_log.update_config(args)
    if not math.isfinite(args.rollout_fps) or args.rollout_fps <= 0:
        action_log.close("validation_error")
        raise ValueError("rollout_fps must be finite and positive")
    if args.rollout_enabled and abs(args.rollout_fps - args.control_frequency) > 1e-6:
        print(
            f"[joint-rollout] recording at {args.rollout_fps:g} Hz independently "
            f"of the {args.control_frequency:g} Hz control loop"
        )
    if args.rollout_output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.rollout_output_dir = str(
            WORKSPACE_DIR / "rollout_datasets" / f"joint_rollout_rtc_{stamp}"
        )
    action_log.update_config(args)

    try:
        rollout = RolloutManager(
            JOINT_ROLLOUT_SCHEMA,
            Path(args.rollout_output_dir).resolve(),
            args.rollout_fps,
            args.rollout_enabled,
            args.rtc_prefix_len,
        )
    except BaseException:
        action_log.close("rollout_startup_error")
        raise
    trajectory_time = args.trajectory_time or (1.1 / args.control_frequency)

    print("=" * 72)
    print("pi0.5 G1 desk joint RTC + rollout")
    print(f"  server: {args.remote_host}:{args.remote_port}")
    print(f"  model/action horizon: {args.action_horizon}")
    print(
        f"  RTC: prefix_len={args.rtc_prefix_len} refill_at={args.rtc_refill_at}"
    )
    print(f"  RTC prefix space: {args.rtc_prefix_space}")
    print(f"  gripper threshold (0=open, 1=closed): >{args.gripper_threshold:g}")
    print(f"  rollout: enabled={rollout.enabled} dir={rollout.output_root}")
    print("  saved state/action: robot/OpenPI [left7, left_hand, right7, right_hand]")
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

            # Confirmation precedes reset, warmup, worker startup and recording.
            if env is not None:
                print(f"Resetting robot to initial pose: {args.task_name}")
                initial_obs = env.reset(task_name=args.task_name)
                print("Initial-pose reset complete.")
            else:
                initial_obs = policy_joint.fake_obs()

            if not warmed_up:
                policy_joint_rtc.warmup_policy(args, policy, initial_obs)
                warmed_up = True
                # Refresh after potentially long JAX compilation and before
                # recording begins, matching the non-rollout RTC entry.
                initial_obs = (
                    env.get_obs() if env is not None else policy_joint.fake_obs()
                )

            obs_provider = (
                env.get_obs_with_last_action
                if env is not None
                else policy_joint.fake_obs
            )
            rollout.begin_episode(prompt, obs_provider)
            episode_reason = "failed"
            try:
                policy_joint_rtc.RTCRunner(
                    args,
                    env,
                    policy,
                    prompt,
                    initial_obs,
                    action_callback=rollout.set_last_action,
                    action_trace_callback=lambda model, actual, step, raw: (
                        action_log.log_action(
                            model,
                            actual,
                            raw_model_action=raw,
                            episode_step=step,
                            source="joint_rollout_rtc_effective_target",
                            environment=env,
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

    main(tyro.cli(JointRolloutArgs))


__all__ = ["EpisodeRecorder", "JointRolloutArgs", "main"]
