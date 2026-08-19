"""pi0.5 G1 桌面任务的同步关节空间推理入口。

文件职责：解析 ``policy_joint.JointArgs``、连接策略服务器、同步获取动作
chunk，并在逐关节限幅及夹爪二值化后发送给 ``RobotEnv``。当前机器人端与
OpenPI checkpoint 已统一使用 16 维顺序
``[left_arm7, left_hand1, right_arm7, right_hand1]``，这里不会再执行旧版
``[L7,R7,Lh,Rh]`` 重排。Prompt 来自 ``policy_prompts.py``，Shell/CLI
提供值时优先，否则由用户选择；确认发生在 reset 和首个动作之前。程序先用
公共客户端等待本地端口转发及远端 checkpoint 就绪，再初始化机器人；连接
超时、退避与关闭均由 ``JointArgs.ws_*`` 控制，重连重发默认关闭。
"""

from __future__ import annotations

import faulthandler
import time

import numpy as np

from action_logging import ActionLogSession
from inference_common import (
    create_robot_environment,
    create_websocket_policy_from_args,
    format_binary_gripper_transitions,
    print_policy_request,
)
from policy_joint import (
    JOINT_DIM,
    MODEL_LEFT_GRIPPER,
    MODEL_RIGHT_GRIPPER,
    OPENPI_LEFT_GRIPPER,
    OPENPI_RIGHT_GRIPPER,
    ROBOT_LEFT_GRIPPER,
    ROBOT_RIGHT_GRIPPER,
    JointArgs,
    checked_joint_vector,
    extract_openpi_actions,
    extract_robot_actions,
    fake_obs,
    openpi_chunk_to_robot_chunk,
    openpi_joint_to_robot_joint,
    prepare_policy_request,
    process_robot_action,
    robot_chunk_to_openpi_chunk,
    robot_joint_to_openpi_joint,
    select_action_window,
    validate_joint_args,
)
from policy_prompts import select_and_confirm_prompt


def main(args: JointArgs) -> None:
    import tqdm

    validate_joint_args(args)
    action_log = ActionLogSession(
        mode="joint", entrypoint="inference_pi05_g1_desk_joint"
    )
    action_log.update_config(args)
    faulthandler.enable()
    control_period = 1.0 / args.control_frequency
    trajectory_time = args.trajectory_time or control_period * 1.1

    print("=" * 72)
    print("pi0.5 G1 desk joint inference client")
    print(f"  server: {args.remote_host}:{args.remote_port}")
    print(f"  task_name: {args.task_name}")
    print(f"  model action horizon: {args.action_horizon}")
    print(f"  executed horizon: {args.execution_horizon}")
    print(f"  skipped chunk prefix: {args.chunk_skip}")
    print(f"  control frequency: {args.control_frequency:g} Hz")
    print(f"  max joint delta/tick: {args.max_joint_delta}")
    print(f"  gripper threshold (0=open, 1=closed): >{args.gripper_threshold:g}")
    print("  joint layout: [left7, left_hand, right7, right_hand] = 16D")
    print("=" * 72)

    env = None
    policy = None
    try:
        print("Connecting policy server (waiting for tunnel/model if needed)...")
        policy = action_log.wrap_policy(create_websocket_policy_from_args(args))
        print("Connected.")

        if not args.use_fake_obs:
            env = create_robot_environment(
                control_mode="joint_position",
                hz=args.control_frequency,
                use_smooth_trajectory=args.use_smooth_trajectory,
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

            # Confirmation precedes reset and the first policy/robot action.
            if env is not None:
                print(f"Resetting robot to initial pose: {args.task_name}")
                obs = env.reset(task_name=args.task_name)
                print("Initial-pose reset complete.")
            else:
                obs = fake_obs()

            action_queue: list[np.ndarray] = []
            current_command = checked_joint_vector(
                obs["state/joint"], name="initial robot joint"
            )
            printed_first_request = False

            try:
                for step_index in tqdm.tqdm(
                    range(args.max_timesteps), desc="running", unit="step"
                ):
                    step_start = time.perf_counter()
                    if not action_queue:
                        obs = env.get_obs() if env is not None else fake_obs()
                        current_command = checked_joint_vector(
                            obs["state/joint"], name="current robot joint"
                        )
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
                        robot_actions = extract_robot_actions(
                            response, max_horizon=args.action_horizon
                        )
                        action_queue = select_action_window(
                            robot_actions,
                            skip=args.chunk_skip,
                            count=args.execution_horizon,
                        )
                        if args.debug:
                            tqdm.tqdm.write(
                                f"infer_ms={infer_ms:.1f} chunk={len(robot_actions)} "
                                f"execute={len(action_queue)}"
                            )

                    raw_action = action_queue.pop(0)
                    previous_before = current_command.copy()
                    command = process_robot_action(
                        raw_action,
                        current_command,
                        gripper_threshold=args.gripper_threshold,
                        max_joint_delta=args.max_joint_delta,
                    )
                    for message in format_binary_gripper_transitions(
                        raw_action,
                        previous_before,
                        command,
                        grippers=(
                            ("left", ROBOT_LEFT_GRIPPER),
                            ("right", ROBOT_RIGHT_GRIPPER),
                        ),
                        prefix="joint gripper",
                        step=step_index + 1,
                    ):
                        tqdm.tqdm.write(message)
                    if env is not None:
                        env.step(command, update_obs=False)
                    action_log.log_action(
                        raw_action,
                        command,
                        raw_model_action=raw_action,
                        episode_step=step_index + 1,
                        source="joint_sync_effective_target",
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

    main(tyro.cli(JointArgs))


__all__ = [
    "JOINT_DIM",
    "MODEL_LEFT_GRIPPER",
    "MODEL_RIGHT_GRIPPER",
    "OPENPI_LEFT_GRIPPER",
    "OPENPI_RIGHT_GRIPPER",
    "ROBOT_LEFT_GRIPPER",
    "ROBOT_RIGHT_GRIPPER",
    "JointArgs",
    "checked_joint_vector",
    "extract_openpi_actions",
    "extract_robot_actions",
    "fake_obs",
    "main",
    "openpi_chunk_to_robot_chunk",
    "openpi_joint_to_robot_joint",
    "prepare_policy_request",
    "process_robot_action",
    "robot_chunk_to_openpi_chunk",
    "robot_joint_to_openpi_joint",
    "select_action_window",
    "validate_joint_args",
]
