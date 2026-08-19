"""把 EE/joint 运行时观测与已执行命令映射为 rollout 帧的策略 schema。

EE schema 保存 30 维 ``[joint16, ee14]`` 状态，以及 ``[joint16_feedback, executed_ee14]`` 动作。
joint schema 的 state/action 均为 16 维 ``[L7, Lhand, R7, Rhand]``，不进行索引重排。
写入的动作始终是经过限幅和夹爪处理、实际交给 RobotEnv 的命令，而非原始模型目标。
本模块依赖 NumPy、两个纯 policy 模块和 ``RolloutSchema``，不执行硬件、网络或文件 I/O。
同步 EE、RTC EE 与 joint rollout 入口把对应 schema 交给通用 recorder 使用；
同步/RTC EE 虽共用维度和编码，但 processing_mode 分开，禁止混写数据集。
编码阶段再次执行严格维度与有限值校验，并为 state/action 生成一致的 modality 区间。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

import policy_ee
import policy_joint
from rollout_recorder import RolloutSchema


def _ee_frame(
    obs: dict[str, Any], executed_ee_action: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    joint = policy_ee.checked_vector(
        obs["state/joint"], dim=policy_ee.JOINT_DIM, name="joint feedback"
    )
    pose = policy_ee.checked_vector(
        obs["state/pose"], dim=policy_ee.EE_DIM, name="EE pose feedback"
    )
    command = policy_ee.checked_vector(
        executed_ee_action, dim=policy_ee.EE_DIM, name="executed EE command"
    )
    state = np.concatenate([joint, pose]).astype(np.float32, copy=False)
    # The EE training transform ignores action[:16]. Synchronized feedback in
    # this prefix preserves the original 30-D source-dataset schema.
    action = np.concatenate([joint, command]).astype(np.float32, copy=False)
    return state, action


def _joint_frame(
    obs: dict[str, Any], executed_robot_action: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    state = policy_joint.checked_joint_vector(
        obs["state/joint"], name="joint feedback"
    )
    action = policy_joint.checked_joint_vector(
        executed_robot_action, name="executed joint command"
    )
    # LeRobot and the current OpenPI joint checkpoint both use this robot order.
    return state, action


EE_MODALITY = {
    "left_arm_joint": {"start": 0, "end": 7},
    "left_joint_gripper": {"start": 7, "end": 8},
    "right_arm_joint": {"start": 8, "end": 15},
    "right_joint_gripper": {"start": 15, "end": 16},
    "left_ee_position": {"start": 16, "end": 19},
    "left_ee_axis_angle": {"start": 19, "end": 22},
    "left_ee_gripper": {"start": 22, "end": 23},
    "right_ee_position": {"start": 23, "end": 26},
    "right_ee_axis_angle": {"start": 26, "end": 29},
    "right_ee_gripper": {"start": 29, "end": 30},
}

JOINT_MODALITY = {
    "left_arm_joint": {"start": 0, "end": 7},
    "left_joint_gripper": {"start": 7, "end": 8},
    "right_arm_joint": {"start": 8, "end": 15},
    "right_joint_gripper": {"start": 15, "end": 16},
}


EE_ROLLOUT_SCHEMA = RolloutSchema(
    name="ee",
    control_mode="camera_position",
    state_dim=30,
    action_dim=30,
    command_dim=policy_ee.EE_DIM,
    state_layout="[joint16, ee14]",
    action_layout="[joint16_feedback, executed_ee14]",
    action_semantics=(
        "action[16:30] is the actual EE command sent after motion limits; "
        "action[0:16] is synchronized joint feedback and is ignored by the "
        "EE training transform"
    ),
    processing_mode="ee_rtc_inference_rollout",
    modality_mapping=EE_MODALITY,
    encode_frame=_ee_frame,
)

# 同步 EE 保存完全相同的30维 state/action，但执行时序不是 RTC。独立的
# processing_mode 会让 RolloutManager 在误用同一输出目录时提前拒绝混写。
EE_SYNC_ROLLOUT_SCHEMA = dataclasses.replace(
    EE_ROLLOUT_SCHEMA,
    processing_mode="ee_sync_inference_rollout",
)

JOINT_ROLLOUT_SCHEMA = RolloutSchema(
    name="joint",
    control_mode="joint_position",
    state_dim=policy_joint.JOINT_DIM,
    action_dim=policy_joint.JOINT_DIM,
    command_dim=policy_joint.JOINT_DIM,
    state_layout="robot [left_arm7, left_gripper, right_arm7, right_gripper]",
    action_layout=(
        "executed robot [left_arm7, left_gripper, right_arm7, right_gripper]"
    ),
    action_semantics=(
        "action is the actual post-limit, post-binarization joint command sent "
        "to RobotEnv; the current OpenPI checkpoint uses the identical order"
    ),
    processing_mode="joint_rtc_inference_rollout",
    modality_mapping=JOINT_MODALITY,
    encode_frame=_joint_frame,
)
