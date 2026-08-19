"""pi0.5 G1 桌面任务的同步末端位姿（EE）推理入口。

文件职责：解析 ``policy_ee.EEArgs``，连接策略服务器，并以同步方式请求动作
chunk、选取执行窗口、执行逐步位置/旋转限幅后发送给机器人。网络状态为
``[joint16, EE14]`` 共 30 维，服务端动作是左右末端位姿与夹爪组成的 EE14。
Prompt 统一由 ``policy_prompts.py`` 选择：Shell/CLI 可覆盖，否则显示菜单，
且在机器人 reset 和执行之前必须得到用户确认。硬件、TQDM、Tyro 与网络依赖
均在真正运行入口后才加载，导入本模块不会初始化机器人。运行时先通过公共
WebSocket 客户端等待本地 SSH 转发和远端模型就绪，再创建机器人环境；握手、
metadata 与推理断线统一按 ``EEArgs.ws_*`` 配置处理，退出时显式关闭连接。
可选 rollout 按独立采样频率保存三路图像、30 维状态和成功下发的命令；
关闭时不会加载 OpenCV/PyArrow，也不会创建数据集。
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
import math
from pathlib import Path
import time

import numpy as np

from action_logging import ActionLogSession
from policy_ee import (
    EEArgs,
    EE_DIM,
    JOINT_DIM,
    LEFT_EE,
    LEFT_GRIPPER,
    LEFT_POSITION,
    LEFT_ROTATION,
    RIGHT_GRIPPER,
    RIGHT_POSITION,
    RIGHT_ROTATION,
    WIRE_STATE_DIM,
    build_wire_state,
    checked_vector,
    extract_ee_actions,
    fake_obs,
    freeze_ee_arm,
    prepare_policy_request,
    process_ee_action,
    retime_ee_action_window,
    resolve_gripper_thresholds,
    select_action_window,
    state_ee_to_action_command,
    validate_ee_args,
)
from inference_common import (
    create_robot_environment,
    create_websocket_policy_from_args,
    format_binary_gripper_transitions,
    print_policy_request,
)
from policy_prompts import select_and_confirm_prompt
from rollout_recorder import RolloutManager
from rollout_schemas import EE_SYNC_ROLLOUT_SCHEMA


WORKSPACE_DIR = Path(__file__).resolve().parent


@dataclasses.dataclass
class SyncRolloutArgs(EEArgs):
    """EE 同步推理参数；rollout 默认关闭，由 launcher 按任务开启。"""

    rollout_output_dir: str | None = None
    rollout_fps: float = 20.0
    rollout_enabled: bool = False
    # Only the synchronous EE entrypoint consumes these fields.  RTC keeps its
    # original fixed-H policy timeline and never inherits this retiming mode.
    retime_action_chunk: bool = False
    max_retimed_chunk_steps: int = 500
    # ``none`` keeps both arms active.  ``left``/``right`` freezes the selected
    # arm's complete EE7 target at the episode-start pose and gripper command.
    freeze_arm: str = "none"


def main(args: SyncRolloutArgs) -> None:
    import tqdm

    validate_ee_args(args)
    action_log = ActionLogSession(
        mode="ee", entrypoint="inference_pi05_g1_desk_ee"
    )
    if not math.isfinite(args.rollout_fps) or args.rollout_fps <= 0:
        action_log.close("validation_error")
        raise ValueError("rollout_fps must be finite and positive")
    if (
        isinstance(args.max_retimed_chunk_steps, bool)
        or args.max_retimed_chunk_steps <= 0
    ):
        action_log.close("validation_error")
        raise ValueError("max_retimed_chunk_steps must be a positive integer")
    resolved_freeze_arm = str(args.freeze_arm).strip().casefold()
    if resolved_freeze_arm not in {"none", "left", "right"}:
        action_log.close("validation_error")
        raise ValueError("freeze_arm must be one of: none, left, right")
    if args.freeze_left_arm:
        if resolved_freeze_arm == "right":
            action_log.close("validation_error")
            raise ValueError(
                "freeze_left_arm conflicts with freeze_arm='right'; use only "
                "freeze_arm for the single-arm test"
            )
        resolved_freeze_arm = "left"
    if args.rollout_output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.rollout_output_dir = str(
            WORKSPACE_DIR / "rollout_datasets" / f"ee_rollout_sync_{stamp}"
        )
    action_log.update_config(args)
    try:
        rollout = RolloutManager(
            EE_SYNC_ROLLOUT_SCHEMA,
            Path(args.rollout_output_dir).resolve(),
            args.rollout_fps,
            args.rollout_enabled,
            0,
        )
    except BaseException:
        action_log.close("rollout_startup_error")
        raise
    close_threshold, open_threshold = resolve_gripper_thresholds(
        gripper_threshold=args.gripper_threshold,
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
    )
    control_period = 1.0 / args.control_frequency
    trajectory_time = args.trajectory_time or control_period * 1.1

    print("=" * 72)
    print("pi0.5 G1 desk EE inference client")
    print(f"  server: {args.remote_host}:{args.remote_port}")
    print(f"  task_name: {args.task_name}")
    print(f"  model action horizon: {args.action_horizon}")
    print(f"  selected model rows per chunk: {args.execution_horizon}")
    print(f"  skipped chunk prefix: {args.chunk_skip}")
    print(f"  control frequency: {args.control_frequency:g} Hz")
    print(f"  trajectory time: {trajectory_time:.4f} s")
    print("  request state: [joint16, EE14] = 30D")
    print("  response action: [L_xyz, L_axisangle, Lg, R_xyz, R_axisangle, Rg] = 14D")
    print(f"  binarize gripper action: {args.binarize_gripper}")
    if args.binarize_gripper:
        if args.gripper_threshold is not None:
            print(f"  gripper legacy single threshold: >{close_threshold:g}")
        else:
            print(
                "  gripper hysteresis: "
                f"close>={close_threshold:g}, open<={open_threshold:g}"
            )
    print(f"  frozen EE arm at episode start: {resolved_freeze_arm}")
    print(f"  synchronous chunk retiming: {args.retime_action_chunk}")
    if args.retime_action_chunk:
        print(
            "  retiming limits: "
            f"position={args.max_position_delta} m/tick, "
            f"rotation={args.max_rotation_delta} rad/tick, "
            f"max_chunk_steps={args.max_retimed_chunk_steps}"
        )
        print(
            "  execution_horizon counts model rows; max_timesteps counts "
            "retimed robot commands"
        )
    print(
        f"  rollout: enabled={rollout.enabled}, fps={rollout.fps:g}, "
        f"dir={rollout.output_root}"
    )
    if rollout.enabled:
        print("  rollout mode: ee_sync_inference_rollout (RTC prefix length=0)")
    print("=" * 72)

    env = None
    policy = None
    try:
        print("Connecting policy server (waiting for tunnel/model if needed)...")
        policy = action_log.wrap_policy(create_websocket_policy_from_args(args))
        print("Connected.")

        if not args.use_fake_obs:
            env = create_robot_environment(
                control_mode="camera_position",
                hz=args.control_frequency,
                trajectory_time=trajectory_time,
                freeze_arm=resolved_freeze_arm,
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
            try:
                # Prompt confirmation happens before reset or recorder startup.
                if env is not None:
                    print(f"Resetting robot to initial pose: {args.task_name}")
                    obs = env.reset(task_name=args.task_name)
                    print("Initial-pose reset complete.")
                else:
                    obs = fake_obs()
                previous_command = state_ee_to_action_command(obs["state/pose"])
                frozen_arm_target = previous_command.copy()
                action_queue: list[np.ndarray] = []
                model_action_queue: list[np.ndarray] = []
                raw_action_queue: list[np.ndarray] = []
                printed_first_request = False
                obs_provider = (
                    env.get_obs_with_last_action if env is not None else fake_obs
                )
                rollout.begin_episode(prompt, obs_provider)

                for step_index in tqdm.tqdm(
                    range(args.max_timesteps), desc="running", unit="step"
                ):
                    step_start = time.perf_counter()
                    if not action_queue:
                        obs = env.get_obs() if env is not None else fake_obs()
                        if not args.retime_action_chunk:
                            feedback_command = state_ee_to_action_command(
                                obs["state/pose"]
                            )
                            # The legacy limiter starts each new chunk from
                            # fresh feedback, but its gripper hysteresis must
                            # retain the last command latch: feedback trails a
                            # command by several frames and could otherwise
                            # undo a just-issued transition.
                            feedback_command[LEFT_GRIPPER] = previous_command[
                                LEFT_GRIPPER
                            ]
                            feedback_command[RIGHT_GRIPPER] = previous_command[
                                RIGHT_GRIPPER
                            ]
                            previous_command = feedback_command
                        # Retiming deliberately keeps previous_command as the
                        # last successfully issued command.  The observation
                        # sent to the policy is still fresh, while this command
                        # baseline makes the rate bound continuous across chunk
                        # boundaries instead of reintroducing a feedback-lag
                        # jump at the first substep of every new chunk.
                        request = prepare_policy_request(
                            obs,
                            prompt,
                            resize_images=args.resize_images,
                            image_size=args.image_size,
                        )
                        if not printed_first_request:
                            print_policy_request(request)
                            printed_first_request = True

                        infer_start = time.perf_counter()
                        response = policy.infer(request)
                        infer_ms = (time.perf_counter() - infer_start) * 1000.0
                        actions = extract_ee_actions(response, max_horizon=args.action_horizon)
                        raw_actions = actions.copy()
                        if resolved_freeze_arm != "none":
                            actions = freeze_ee_arm(
                                actions,
                                frozen_arm_target,
                                arm=resolved_freeze_arm,
                            )
                        selected_actions = select_action_window(
                            actions,
                            skip=args.chunk_skip,
                            count=args.execution_horizon,
                        )
                        selected_raw_actions = select_action_window(
                            raw_actions,
                            skip=args.chunk_skip,
                            count=args.execution_horizon,
                        )
                        if args.retime_action_chunk:
                            retimed = retime_ee_action_window(
                                np.stack(selected_actions, axis=0),
                                previous_command,
                                max_position_delta=args.max_position_delta,
                                max_rotation_delta=args.max_rotation_delta,
                                binarize_gripper=args.binarize_gripper,
                                gripper_threshold=args.gripper_threshold,
                                gripper_close_threshold=(
                                    args.gripper_close_threshold
                                ),
                                gripper_open_threshold=args.gripper_open_threshold,
                                max_output_steps=args.max_retimed_chunk_steps,
                            )
                            action_queue = [
                                command.copy() for command in retimed.commands
                            ]
                            model_action_queue = [
                                selected_actions[int(source_index)].copy()
                                for source_index in retimed.source_indices
                            ]
                            raw_action_queue = [
                                selected_raw_actions[int(source_index)].copy()
                                for source_index in retimed.source_indices
                            ]
                            max_segment_steps = int(retimed.segment_steps.max())
                            tqdm.tqdm.write(
                                "[EE retime] "
                                f"model_rows={len(selected_actions)} -> "
                                f"commands={len(action_queue)} "
                                f"({len(action_queue) / len(selected_actions):.2f}x), "
                                f"max_segment_steps={max_segment_steps}, "
                                f"duration={len(action_queue) / args.control_frequency:.2f}s"
                            )
                        else:
                            action_queue = selected_actions
                            model_action_queue = [
                                action.copy() for action in selected_actions
                            ]
                            raw_action_queue = selected_raw_actions
                        if args.debug:
                            tqdm.tqdm.write(
                                f"infer_ms={infer_ms:.1f} chunk={len(actions)} "
                                f"skip={args.chunk_skip} "
                                f"execute_commands={len(action_queue)}"
                            )

                    raw_server_action = raw_action_queue.pop(0)
                    model_action = model_action_queue.pop(0)
                    execution_target = action_queue.pop(0)
                    previous_before = previous_command.copy()
                    if args.retime_action_chunk:
                        # The complete window was already rate-constrained on
                        # SO(3). Re-running the old rotvec-Euclidean limiter
                        # here would distort SLERP and recreate trajectory lag.
                        action = checked_vector(
                            execution_target,
                            dim=EE_DIM,
                            name="retimed EE command",
                        )
                    else:
                        action = process_ee_action(
                            execution_target,
                            previous_command,
                            max_position_delta=args.max_position_delta,
                            max_rotation_delta=args.max_rotation_delta,
                            binarize_gripper=args.binarize_gripper,
                            gripper_threshold=args.gripper_threshold,
                            gripper_close_threshold=args.gripper_close_threshold,
                            gripper_open_threshold=args.gripper_open_threshold,
                        )
                    if args.binarize_gripper:
                        for message in format_binary_gripper_transitions(
                            model_action,
                            previous_before,
                            action,
                            grippers=(
                                ("left", LEFT_GRIPPER),
                                ("right", RIGHT_GRIPPER),
                            ),
                            prefix="EE gripper",
                            step=step_index + 1,
                        ):
                            tqdm.tqdm.write(message)
                    if env is not None:
                        env.step(action, update_obs=False)
                    # Only publish commands that were successfully sent (or
                    # intentionally simulated in fake-observation mode).
                    rollout.set_last_action(action.copy())
                    action_log.log_action(
                        model_action,
                        action,
                        raw_model_action=raw_server_action,
                        episode_step=step_index + 1,
                        source=(
                            "ee_sync_retimed_command"
                            if args.retime_action_chunk
                            else "ee_sync_effective_target"
                        ),
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
                    rollout.end_episode()
                finally:
                    action_log.end_episode(episode_reason)

            answer = input("\nContinue with another task? (y/n): ").strip().lower()
            if answer not in {"y", "yes"}:
                break
    finally:
        try:
            # Retry a transient recorder commit failure before releasing the
            # policy and hardware resources. A completed manager is a no-op.
            rollout.end_episode()
        finally:
            action_log.close()
            if policy is not None:
                policy.close()
            if env is not None:
                env.close()


if __name__ == "__main__":
    import tyro

    main(tyro.cli(SyncRolloutArgs))


__all__ = ["SyncRolloutArgs", "main"]
