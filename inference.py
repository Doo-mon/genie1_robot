"""
Agibot 机器人通用推理脚本（固定数据格式）。

文件说明：
    这是较早的统一推理入口，同时支持 ``joint_position`` 与
    ``camera_position`` 两种控制模式，并保留插值、外插、图像保存和轨迹
    可视化等调试能力。Prompt 文本统一来自 ``policy_prompts.py``：Shell/CLI
    提供的 prompt 优先，否则进入交互菜单；任何 prompt 都必须经用户确认后
    才会 reset 和执行。新部署若只使用桌面 EE 或 joint 模型，优先选择对应的
    ``inference_pi05_g1_desk_*`` 薄入口，本文件主要用于兼容原通用流程。
    网络连接同样复用 ``inference_common`` 的可靠客户端：先等待本地 SSH
    端口转发和远端模型，再初始化机器人；握手、metadata、响应超时和运行中
    重连均由 ``Args.ws_*`` 配置，结束时显式关闭 socket 与机器人环境。

================================================================================
数据格式说明
================================================================================

【发送给服务端的数据格式】

┌─────────────────────────────────────────────────────────────────────────────┐
│ 字段名                          │ 类型        │ 维度/说明                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ observation/image               │ np.ndarray  │ (800, 1280, 3) uint8 头部   │
│ observation/wrist_left_image    │ np.ndarray  │ (480, 848, 3) uint8 左腕    │
│ observation/wrist_right_image   │ np.ndarray  │ (480, 848, 3) uint8 右腕    │
│ observation/state               │ np.ndarray  │ (18,) 或 (20,) float32      │
│ prompt                          │ str         │ 任务指令文本                │
└─────────────────────────────────────────────────────────────────────────────┘

【camera_position 模式 - 状态维度 18】

observation/state 维度分解:
┌───────────────────────────────────────────────────────────────────────────┐
│ 索引     │ 内容               │ 维度 │ 说明                              │
├───────────────────────────────────────────────────────────────────────────┤
│ [0:3]    │ 左手位置 xyz       │ 3    │ 相机坐标系下的位置 (米)           │
│ [3:6]    │ 左手轴角           │ 3    │ 旋转向量表示的姿态                │
│ [6]      │ 左手夹爪 state     │ 1    │ 编码位置：0=张开, 120=闭合        │
│ [7:10]   │ 右手位置 xyz       │ 3    │ 相机坐标系下的位置 (米)           │
│ [10:13]  │ 右手轴角           │ 3    │ 旋转向量表示的姿态                │
│ [13]     │ 右手夹爪 state     │ 1    │ 编码位置：0=张开, 120=闭合        │
│ [14:16]  │ 头部               │ 2    │ 头部关节角度                      │
│ [16:18]  │ 腰部               │ 2    │ 腰部关节角度                      │
├───────────────────────────────────────────────────────────────────────────┤
│ 总计     │                    │ 18   │                                   │
└───────────────────────────────────────────────────────────────────────────┘

【joint_position 模式 - 状态维度 20】

observation/state 维度分解:
┌───────────────────────────────────────────────────────────────────────────┐
│ 索引     │ 内容               │ 维度 │ 说明                              │
├───────────────────────────────────────────────────────────────────────────┤
│ [0:7]    │ 左臂关节角         │ 7    │ 7个关节的角度 (弧度)              │
│ [7]      │ 左手夹爪 state     │ 1    │ 编码位置：0=张开, 120=闭合        │
│ [8:15]   │ 右臂关节角         │ 7    │ 7个关节的角度 (弧度)              │
│ [15]     │ 右手夹爪 state     │ 1    │ 编码位置：0=张开, 120=闭合        │
│ [16:18]  │ 头部               │ 2    │ 头部关节角度                      │
│ [18:20]  │ 腰部               │ 2    │ 腰部关节角度                      │
├───────────────────────────────────────────────────────────────────────────┤
│ 总计     │                    │ 20   │                                   │
└───────────────────────────────────────────────────────────────────────────┘

【从服务端接收的动作格式】

服务端返回 response["actions"]，维度与状态相同：
- camera_position 模式: (chunk_size, 18) 
- joint_position 模式: (chunk_size, 20)

action 的夹爪维为连续 0--1，当前数据约定 0=张开、1=闭合；它与
observation state 中的 0--120 编码位置不是同一数值空间。本兼容入口仍会
按阈值把夹爪转成端点。新 EE 薄入口直接运行 Python 时默认保留连续
action，而配套通用 Shell 默认开启双阈值滞回二值化。

================================================================================
使用示例
================================================================================

# camera_position 模式
python workspace/inference.py \\
    --task-name beijing_benchmark_setting \\
    --control-mode camera_position \\
    --remote-host 127.0.0.1 \\
    --remote-port 2379

# joint_position 模式
python workspace/inference.py \\
    --task-name beijing_benchmark_setting \\
    --control-mode joint_position \\
    --remote-host 127.0.0.1 \\
    --remote-port 2379

"""

import dataclasses
import faulthandler
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

# ============ 导入模块 ============
from action_logging import ActionLogSession
from inference_common import (
    create_robot_environment,
    create_websocket_policy_from_args,
)
from inference_utils import (
    plot_trajectory_xyz,
    save_first_request_images,
    print_first_request_data,
)
from policy_prompts import select_and_confirm_prompt
import policy_ee
import policy_joint

# 状态和动作维度
CAMERA_POSITION_DIM = 18  # 左手7 + 右手7 + 头2 + 腰2
JOINT_POSITION_DIM = 20   # 左手8 + 右手8 + 头2 + 腰2

# 夹爪索引
CAMERA_LEFT_GRIPPER_IDX = 6
CAMERA_RIGHT_GRIPPER_IDX = 13
JOINT_LEFT_GRIPPER_IDX = 7
JOINT_RIGHT_GRIPPER_IDX = 15

CAMERA_ROBOT_COMMAND_DIM = 14
JOINT_ROBOT_COMMAND_DIM = 16


# ============ 数据处理函数（内联，不依赖 inference_utils） ============

def get_fake_obs(control_mode: str) -> Dict[str, Any]:
    """
    生成假的观测数据，用于测试。
    
    Args:
        control_mode: 'camera_position' 或 'joint_position'
    
    Returns:
        obs: 符合格式要求的观测字典
        
    【返回数据结构】
    
    camera_position 模式:
    {
        "image/head": np.ndarray (800, 1280, 3) uint8,
        "image/left_wrist": np.ndarray (480, 848, 3) uint8,
        "image/right_wrist": np.ndarray (480, 848, 3) uint8,
        "state/pose": np.ndarray (14,) float32,  # 左手7 + 右手7
        "state/head": np.ndarray (2,) float32,
        "state/waist": np.ndarray (2,) float32,
    }
    
    joint_position 模式:
    {
        "image/head": np.ndarray (800, 1280, 3) uint8,
        "image/left_wrist": np.ndarray (480, 848, 3) uint8,
        "image/right_wrist": np.ndarray (480, 848, 3) uint8,
        "state/joint": np.ndarray (16,) float32,  # 左手8 + 右手8
        "state/head": np.ndarray (2,) float32,
        "state/waist": np.ndarray (2,) float32,
    }
    """
    # 生成假图像（彩色噪声），尺寸与真实机器人一致
    # 头部图像: 800 x 1280
    # 腕部图像: 480 x 848
    fake_head_image = np.random.randint(0, 256, (800, 1280, 3), dtype=np.uint8)
    fake_left_wrist_image = np.random.randint(0, 256, (480, 848, 3), dtype=np.uint8)
    fake_right_wrist_image = np.random.randint(0, 256, (480, 848, 3), dtype=np.uint8)
    
    # 生成假的头部和腰部状态
    fake_head_state = np.zeros(2, dtype=np.float32)
    fake_waist_state = np.zeros(2, dtype=np.float32)
    
    obs = {
        "image/head": fake_head_image,
        "image/left_wrist": fake_left_wrist_image,
        "image/right_wrist": fake_right_wrist_image,
        "state/head": fake_head_state,
        "state/waist": fake_waist_state,
    }
    
    if control_mode == "camera_position":
        # 左手: xyz(3) + 轴角(3) + 夹爪(1) = 7
        # 右手: xyz(3) + 轴角(3) + 夹爪(1) = 7
        # 总计: 14
        fake_pose = np.zeros(14, dtype=np.float32)
        # 设置一些合理的默认位置
        fake_pose[0:3] = [0.3, 0.2, 0.5]   # 左手位置
        fake_pose[3:6] = [0.0, 0.0, 0.0]   # 左手轴角
        fake_pose[6] = 0.0                  # 左手夹爪 state（张开）
        fake_pose[7:10] = [0.3, -0.2, 0.5] # 右手位置
        fake_pose[10:13] = [0.0, 0.0, 0.0] # 右手轴角
        fake_pose[13] = 0.0                 # 右手夹爪 state（张开）
        obs["state/pose"] = fake_pose
    else:
        # 左臂: 7关节 + 夹爪(1) = 8
        # 右臂: 7关节 + 夹爪(1) = 8
        # 总计: 16
        fake_joint = np.zeros(16, dtype=np.float32)
        fake_joint[7] = 0.0   # 左手夹爪 state（张开）
        fake_joint[15] = 0.0  # 右手夹爪 state（张开）
        obs["state/joint"] = fake_joint
    
    return obs


def prepare_observation_for_policy(obs: Dict, control_mode: str) -> Dict:
    """
    将 RobotEnv 的观测格式转换为策略服务器期望的格式。
    
    - 图像：直接使用原始图像，不做 resize
    - 状态：拼接为完整的 18 维 (camera_position) 或 20 维 (joint_position)
    
    Args:
        obs: RobotEnv 返回的观测字典
        control_mode: 'camera_position' 或 'joint_position'
    
    Returns:
        request_data: 策略服务器期望的请求数据
        
    【发送数据示例】
    
    camera_position 模式:
    {
        "observation/image": np.ndarray (800, 1280, 3) uint8,
        "observation/wrist_left_image": np.ndarray (480, 848, 3) uint8,
        "observation/wrist_right_image": np.ndarray (480, 848, 3) uint8,
        "observation/state": np.ndarray (18,) float32,
        "prompt": "<selected prompt>"
    }
    
    joint_position 模式:
    {
        "observation/image": np.ndarray (800, 1280, 3) uint8,
        "observation/wrist_left_image": np.ndarray (480, 848, 3) uint8,
        "observation/wrist_right_image": np.ndarray (480, 848, 3) uint8,
        "observation/state": np.ndarray (20,) float32,
        "prompt": "<selected prompt>"
    }
    """
    # 处理图像：直接使用原图，移除 alpha 通道（如果有）
    head_image = obs["image/head"]
    wrist_left_image = obs["image/left_wrist"]
    wrist_right_image = obs["image/right_wrist"]
    
    if head_image.shape[-1] == 4:
        head_image = head_image[..., :3]
    if wrist_left_image.shape[-1] == 4:
        wrist_left_image = wrist_left_image[..., :3]
    if wrist_right_image.shape[-1] == 4:
        wrist_right_image = wrist_right_image[..., :3]
    
    # 处理状态：拼接为完整维度
    head_state = obs["state/head"]    # (2,)
    waist_state = obs["state/waist"]  # (2,)
    
    if control_mode == "camera_position":
        base_state = obs["state/pose"]  # (14,)
    else:
        base_state = obs["state/joint"]  # (16,)
    
    # 拼接: [base_state, head_state, waist_state]
    full_state = np.concatenate([base_state, head_state, waist_state]).astype(np.float32)
    
    request_data = {
        "observation/image": head_image,
        "observation/wrist_left_image": wrist_left_image,
        "observation/wrist_right_image": wrist_right_image,
        "observation/state": full_state,
    }
    
    return request_data


def extract_compat_actions(
    response: Dict[str, Any],
    control_mode: str,
) -> np.ndarray:
    """Validate the legacy server's strict Hx18/Hx20 action response."""

    if "actions" not in response:
        raise KeyError(
            f"Policy response has no 'actions'; keys={sorted(response.keys())}"
        )
    expected_dim = (
        CAMERA_POSITION_DIM
        if control_mode == "camera_position"
        else JOINT_POSITION_DIM
    )
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError(f"Expected action batch size 1, got {actions.shape}")
        actions = actions[0]
    elif actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != expected_dim:
        raise ValueError(
            f"Expected {control_mode} actions shape (H, {expected_dim}), "
            f"got {actions.shape}"
        )
    if not len(actions):
        raise ValueError("Policy returned an empty action chunk")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy actions contain NaN or Inf")
    return actions.copy()


def policy_action_to_robot_command(
    action: np.ndarray,
    control_mode: str,
) -> np.ndarray:
    """Strip the legacy head/waist tail before calling ``RobotEnv.step``.

    The old policy predicts EE14/Joint16 plus head2 and waist2.  RobotEnv has
    no fixed-rate head/waist command path, so it intentionally executes only
    the arm/hand prefix and keeps the current head/waist pose.
    """

    expected_dim = (
        CAMERA_POSITION_DIM
        if control_mode == "camera_position"
        else JOINT_POSITION_DIM
    )
    command_dim = (
        CAMERA_ROBOT_COMMAND_DIM
        if control_mode == "camera_position"
        else JOINT_ROBOT_COMMAND_DIM
    )
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.shape != (expected_dim,):
        raise ValueError(
            f"Expected one {control_mode} action with shape ({expected_dim},), "
            f"got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("Robot command source contains NaN or Inf")
    return value[:command_dim].copy()


def process_single_action(
    action: np.ndarray,
    control_mode: str,
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    """
    处理单个动作：二值化夹爪。
    
    Args:
        action: 原始动作 (18,) 或 (20,)
        control_mode: 'camera_position' 或 'joint_position'
        gripper_threshold: 夹爪二值化阈值
    
    Returns:
        processed_action: 处理后的动作，夹爪已二值化
    """
    result = action.copy()
    
    if control_mode == "camera_position":
        left_idx = CAMERA_LEFT_GRIPPER_IDX
        right_idx = CAMERA_RIGHT_GRIPPER_IDX
    else:
        left_idx = JOINT_LEFT_GRIPPER_IDX
        right_idx = JOINT_RIGHT_GRIPPER_IDX
    
    result[left_idx] = 1.0 if result[left_idx] > gripper_threshold else 0.0
    result[right_idx] = 1.0 if result[right_idx] > gripper_threshold else 0.0
    
    return result


# ============ 插值函数 ============

def _slerp_axis_angle(aa1: np.ndarray, aa2: np.ndarray, t: float) -> np.ndarray:
    """
    对两个轴角表示的旋转进行球面线性插值 (SLERP)
    """
    from scipy.spatial.transform import Rotation, Slerp

    norm1 = np.linalg.norm(aa1)
    norm2 = np.linalg.norm(aa2)
    
    if norm1 < 1e-8 and norm2 < 1e-8:
        return np.zeros(3, dtype=np.float32)
    elif norm1 < 1e-8:
        return (aa2 * t).astype(np.float32)
    elif norm2 < 1e-8:
        return (aa1 * (1 - t)).astype(np.float32)
    
    r1 = Rotation.from_rotvec(aa1)
    r2 = Rotation.from_rotvec(aa2)
    
    slerp = Slerp([0, 1], Rotation.concatenate([r1, r2]))
    r_interp = slerp(t)
    
    return r_interp.as_rotvec().astype(np.float32)


def interpolate_single_action_pair(
    action_prev: np.ndarray,
    action_next: np.ndarray,
    control_mode: str,
    num_interp: int,
) -> np.ndarray:
    """
    在两个动作之间进行插值。
    
    Args:
        action_prev: 前一个动作 (18,) 或 (20,)
        action_next: 后一个动作 (18,) 或 (20,)
        control_mode: 'camera_position' 或 'joint_position'
        num_interp: 插值点数
    
    Returns:
        interpolated: (num_interp, action_dim) 插值后的动作序列
    """
    t_values = np.linspace(0, 1, num_interp + 1)[1:]
    
    interpolated = []
    
    for t in t_values:
        if control_mode == "camera_position":
            action_interp = _interpolate_camera_action(action_prev, action_next, t)
        else:
            action_interp = _interpolate_joint_action(action_prev, action_next, t)
        interpolated.append(action_interp)
    
    return np.array(interpolated, dtype=np.float32)


def _interpolate_joint_action(a1: np.ndarray, a2: np.ndarray, t: float) -> np.ndarray:
    """
    关节位置模式：直接线性插值
    维度: [左臂7, 左夹爪1, 右臂7, 右夹爪1, 头2, 腰2] = 20
    """
    result = a1 * (1 - t) + a2 * t
    
    # 夹爪保持目标值
    result[JOINT_LEFT_GRIPPER_IDX] = a2[JOINT_LEFT_GRIPPER_IDX]
    result[JOINT_RIGHT_GRIPPER_IDX] = a2[JOINT_RIGHT_GRIPPER_IDX]
    
    return result.astype(np.float32)


def _interpolate_camera_action(a1: np.ndarray, a2: np.ndarray, t: float) -> np.ndarray:
    """
    相机坐标系模式：位置线性插值，轴角用 SLERP
    维度: [左xyz3, 左轴角3, 左夹爪1, 右xyz3, 右轴角3, 右夹爪1, 头2, 腰2] = 18
    """
    result = np.zeros(18, dtype=np.float32)
    
    # 左手
    result[0:3] = a1[0:3] * (1 - t) + a2[0:3] * t      # xyz 线性插值
    result[3:6] = _slerp_axis_angle(a1[3:6], a2[3:6], t)  # 轴角 SLERP
    result[6] = a2[6]                                      # 夹爪保持目标值
    
    # 右手
    result[7:10] = a1[7:10] * (1 - t) + a2[7:10] * t   # xyz 线性插值
    result[10:13] = _slerp_axis_angle(a1[10:13], a2[10:13], t)  # 轴角 SLERP
    result[13] = a2[13]                                    # 夹爪保持目标值
    
    # 头和腰：线性插值
    result[14:18] = a1[14:18] * (1 - t) + a2[14:18] * t
    
    return result


def interpolate_action_chunk(
    action_chunk: np.ndarray,
    control_mode: str,
    control_frequency: float,
    execution_frequency: float,
    open_loop_horizon: int,
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    """
    对整个 action chunk 的前 open_loop_horizon 个动作进行插值。
    
    Args:
        action_chunk: (chunk_size, action_dim) 原始动作序列
        control_mode: 'camera_position' 或 'joint_position'
        control_frequency: 模型输出频率 Hz
        execution_frequency: 实际执行频率 Hz
        open_loop_horizon: 开环控制步数
        gripper_threshold: 夹爪二值化阈值
    
    Returns:
        interpolated_actions: 插值后的动作序列
    """
    interp_ratio = execution_frequency / control_frequency
    num_interp_per_pair = int(np.round(interp_ratio))
    
    num_actions = min(open_loop_horizon, len(action_chunk))
    
    # 处理动作：二值化夹爪
    processed_actions = []
    for i in range(num_actions):
        action = process_single_action(action_chunk[i], control_mode, gripper_threshold)
        processed_actions.append(action)
    
    # 插值
    interpolated_actions = [processed_actions[0]]
    
    for i in range(1, num_actions):
        prev_action = processed_actions[i - 1]
        next_action = processed_actions[i]
        
        interp_actions = interpolate_single_action_pair(
            prev_action,
            next_action,
            control_mode,
            num_interp_per_pair
        )
        interpolated_actions.extend(interp_actions)
    
    return np.array(interpolated_actions, dtype=np.float32)


# ============ 外插函数 ============

def extrapolate_linear(
    last_action: np.ndarray,
    second_last_action: np.ndarray,
    control_mode: str,
    num_steps: int = 1
) -> np.ndarray:
    """
    线性外插：根据最后两个动作的变化趋势，生成未来动作
    """
    from scipy.spatial.transform import Rotation

    extrapolated = []
    
    for step in range(1, num_steps + 1):
        if control_mode == "camera_position":
            new_action = np.zeros_like(last_action)
            
            # 位置部分：线性外插
            delta_pos_left = last_action[0:3] - second_last_action[0:3]
            delta_pos_right = last_action[7:10] - second_last_action[7:10]
            new_action[0:3] = last_action[0:3] + delta_pos_left * step
            new_action[7:10] = last_action[7:10] + delta_pos_right * step
            
            # 旋转部分：使用相对旋转外插
            if np.linalg.norm(last_action[3:6]) > 1e-8 and np.linalg.norm(second_last_action[3:6]) > 1e-8:
                r_prev_left = Rotation.from_rotvec(second_last_action[3:6])
                r_curr_left = Rotation.from_rotvec(last_action[3:6])
                r_delta_left = r_curr_left * r_prev_left.inv()
                r_next_left = (r_delta_left ** step) * r_curr_left
                new_action[3:6] = r_next_left.as_rotvec()
            else:
                new_action[3:6] = last_action[3:6]
            
            if np.linalg.norm(last_action[10:13]) > 1e-8 and np.linalg.norm(second_last_action[10:13]) > 1e-8:
                r_prev_right = Rotation.from_rotvec(second_last_action[10:13])
                r_curr_right = Rotation.from_rotvec(last_action[10:13])
                r_delta_right = r_curr_right * r_prev_right.inv()
                r_next_right = (r_delta_right ** step) * r_curr_right
                new_action[10:13] = r_next_right.as_rotvec()
            else:
                new_action[10:13] = last_action[10:13]
            
            # 夹爪保持
            new_action[6] = last_action[6]
            new_action[13] = last_action[13]
            
            # 头腰线性外插
            delta_head_waist = last_action[14:18] - second_last_action[14:18]
            new_action[14:18] = last_action[14:18] + delta_head_waist * step
            
        else:
            # 关节模式：直接线性外插
            delta = last_action - second_last_action
            new_action = last_action + delta * step
            
            # 夹爪保持
            new_action[JOINT_LEFT_GRIPPER_IDX] = last_action[JOINT_LEFT_GRIPPER_IDX]
            new_action[JOINT_RIGHT_GRIPPER_IDX] = last_action[JOINT_RIGHT_GRIPPER_IDX]
        
        extrapolated.append(new_action)
    
    return np.array(extrapolated, dtype=np.float32)




# ============ 参数定义 ============

@dataclasses.dataclass
class Args:
    """推理参数"""
    task_name: str
    """任务名称（必需）"""
    
    control_mode: Literal["joint_position", "camera_position"] = "joint_position"
    """控制模式: 'joint_position' (20维) 或 'camera_position' (18维)"""
    
    remote_host: str = "127.0.0.1"
    """本地端口转发的监听地址；也可传 ws:// 或 wss:// 完整地址"""
    
    remote_port: int = 2379
    """本地转发端口"""

    ws_open_timeout: float = 10.0
    """单次 WebSocket 握手超时"""

    ws_close_timeout: float = 10.0
    """关闭 WebSocket 的最长等待"""

    ws_response_timeout: float = 300.0
    """metadata/推理响应超时；0 表示不限时"""

    ws_retry_initial_delay: float = 1.0
    """首次连接失败后的初始重试间隔"""

    ws_retry_max_delay: float = 10.0
    """指数退避的最大重试间隔"""

    ws_server_wait_timeout: float = 0.0
    """启动时等待隧道和模型的总时限；0 表示持续等待"""

    ws_reconnect_wait_timeout: float = 30.0
    """运行中断线后每次重连的总等待时限；0 表示持续等待"""

    ws_inference_retries: int = 0
    """一次推理因传输失败而允许重连重发的次数；默认 0 表示不重发"""
    
    max_timesteps: int = 600
    """最大执行步数"""
    
    open_loop_horizon: int = 5
    """开环控制步数"""
    
    gripper_threshold: float = 0.5
    """夹爪二值化阈值"""

    max_position_delta: float | None = 0.004
    """camera_position 每步单手位置最大变化（米）"""

    max_rotation_delta: float | None = 0.02
    """camera_position 每步单手轴角最大变化"""

    min_ee_base_z: float | None = None
    """camera_position 在 base_link 坐标系中的共同最小 z（兼容参数）"""

    min_left_ee_base_z: float | None = None
    """camera_position 左末端独立最小 z；优先于共同值"""

    min_right_ee_base_z: float | None = None
    """camera_position 右末端独立最小 z；优先于共同值"""

    max_joint_delta: float | None = 0.15
    """joint_position 每个手臂关节最大单步变化"""
    
    save_images: bool = False
    """是否保存图像"""
    
    control_frequency: float = 30.0
    """模型输出频率 Hz"""
    
    trajectory_time: Optional[float] = None
    """轨迹执行时间（仅 camera_position 模式）"""
    
    use_interpolation: bool = False
    """是否启用内插"""
    
    execution_frequency: float = 100.0
    """实际执行频率 Hz（仅当 use_interpolation=True 时生效）"""
    
    extrapolation_mode: Literal["none", "hold", "linear"] = "none"
    """外插模式: 'none', 'hold', 'linear'"""
    
    extrapolation_skip: int = 4
    """linear模式下新chunk到达后丢弃的动作数"""
    
    max_extrapolation_steps: int = 10
    """最大外插步数"""
    
    debug: bool = False
    """调试模式，启用可视化"""
    
    use_fake_obs: bool = False
    """使用假数据测试（不连接真实机器人）"""

    prompt: Optional[str] = None
    """Shell/CLI 提供的 prompt；仍会在执行前要求确认"""

    prompt_key: Optional[str] = None
    """policy_prompts.py 中与当前 EE/joint profile 对应的 prompt key"""


def validate_args(args: Args) -> None:
    """Reject invalid timing/action settings before connecting or moving."""

    if args.control_mode not in {"joint_position", "camera_position"}:
        raise ValueError(f"Unsupported control_mode: {args.control_mode!r}")
    if not np.isfinite(args.control_frequency) or args.control_frequency <= 0:
        raise ValueError("control_frequency must be finite and positive")
    if not np.isfinite(args.execution_frequency) or args.execution_frequency <= 0:
        raise ValueError("execution_frequency must be finite and positive")
    if args.use_interpolation and round(
        args.execution_frequency / args.control_frequency
    ) < 1:
        raise ValueError(
            "execution_frequency/control_frequency must round to at least 1 "
            "when interpolation is enabled"
        )
    if args.max_timesteps <= 0:
        raise ValueError("max_timesteps must be positive")
    if args.open_loop_horizon <= 0:
        raise ValueError("open_loop_horizon must be positive")
    if not np.isfinite(args.gripper_threshold) or not 0 <= args.gripper_threshold <= 1:
        raise ValueError("gripper_threshold must be finite and within [0, 1]")
    for name in ("max_position_delta", "max_rotation_delta", "max_joint_delta"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive or None")
    floor_names = (
        "min_ee_base_z",
        "min_left_ee_base_z",
        "min_right_ee_base_z",
    )
    configured_floors = [
        (name, getattr(args, name))
        for name in floor_names
        if getattr(args, name) is not None
    ]
    if configured_floors:
        if args.control_mode != "camera_position":
            raise ValueError(
                "EE base z floors are only valid for camera_position mode"
            )
        for name, value in configured_floors:
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite or None")
    if args.trajectory_time is not None and (
        not np.isfinite(args.trajectory_time) or args.trajectory_time <= 0
    ):
        raise ValueError("trajectory_time must be finite and positive")
    if args.extrapolation_skip < 0:
        raise ValueError("extrapolation_skip must be non-negative")
    if args.max_extrapolation_steps < 0:
        raise ValueError("max_extrapolation_steps must be non-negative")
    if not args.task_name.strip():
        raise ValueError("task_name must not be empty")
    if args.prompt is not None and not args.prompt.strip():
        raise ValueError("prompt must not be empty when provided")
    if args.prompt_key is not None and not args.prompt_key.strip():
        raise ValueError("prompt_key must not be empty when provided")


def main(args: Args):
    """主推理循环"""
    import tqdm

    validate_args(args)
    action_log = ActionLogSession(
        mode=("ee" if args.control_mode == "camera_position" else "joint"),
        entrypoint="inference_compat",
    )
    action_log.update_config(args)
    faulthandler.enable()
    
    # 计算控制参数
    control_period = 1.0 / args.control_frequency
    action_dim = CAMERA_POSITION_DIM if args.control_mode == "camera_position" else JOINT_POSITION_DIM
    
    if args.use_interpolation:
        execution_period = 1.0 / args.execution_frequency
        interp_ratio = args.execution_frequency / args.control_frequency
        num_interp_per_pair = int(np.round(interp_ratio))
    else:
        execution_period = control_period
        num_interp_per_pair = 1
    
    trajectory_time = args.trajectory_time if args.trajectory_time else execution_period * 1.1
    
    images_dir = Path("images") if args.save_images else None
    if images_dir:
        images_dir.mkdir(parents=True, exist_ok=True)
    
    # 打印配置
    print("=" * 70)
    print("Agibot 推理服务（简化版 - 固定数据格式）")
    print("=" * 70)
    print(f"  任务名称: {args.task_name}")
    print(f"  控制模式: {args.control_mode}")
    print(f"  状态/动作维度: {action_dim}")
    robot_command_dim = (
        CAMERA_ROBOT_COMMAND_DIM
        if args.control_mode == "camera_position"
        else JOINT_ROBOT_COMMAND_DIM
    )
    print(
        f"  实际机器人命令维度: {robot_command_dim} "
        f"(模型尾部 head2+waist2 保持但不下发)"
    )
    print(f"  策略服务器: {args.remote_host}:{args.remote_port}")
    print(f"  模型输出频率: {args.control_frequency} Hz (周期 {control_period*1000:.1f} ms)")
    print(f"  开环 horizon: {args.open_loop_horizon}")
    if args.control_mode == "camera_position":
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
        print(
            "  单步安全限幅: "
            f"position={args.max_position_delta}, rotation={args.max_rotation_delta}"
        )
        print(
            "  base_link EE z 最低位置: "
            f"left={'disabled' if left_z_floor is None else f'{left_z_floor:.4f} m'}, "
            f"right={'disabled' if right_z_floor is None else f'{right_z_floor:.4f} m'} "
            "(clamp before SDK)"
        )
    else:
        print(f"  单步关节限幅: {args.max_joint_delta}")
    
    if args.use_interpolation:
        print(f"  【插值已启用】")
        print(f"    执行频率: {args.execution_frequency} Hz (周期 {execution_period*1000:.1f} ms)")
        print(f"    插值比例: {interp_ratio:.2f}x (每对动作间 {num_interp_per_pair} 个点)")
    else:
        print(f"  【插值未启用】")
        print(f"    执行频率: {args.control_frequency} Hz")
    
    print(f"  【外插模式】: {args.extrapolation_mode}")
    if args.extrapolation_mode == 'linear':
        print(f"    - 丢弃动作数: {args.extrapolation_skip}")
        print(f"    - 最大外插步数: {args.max_extrapolation_steps}")
    elif args.extrapolation_mode == 'hold':
        print(f"    - 最大保持步数: {args.max_extrapolation_steps}")
    
    if args.control_mode == "camera_position":
        print(f"  轨迹时间: {trajectory_time*1000:.1f} ms")
    
    if args.use_fake_obs:
        print(f"  【测试模式】使用假数据")
    
    print("=" * 70)

    # 先等待本地端口转发和远端模型，再初始化机器人；服务端启动较慢时
    # 不会提前占用硬件。连接失败会由公共客户端自动退避重试。
    print("\n[1/2] 连接策略服务器（必要时等待隧道/模型就绪）...")
    try:
        policy_client = action_log.wrap_policy(
            create_websocket_policy_from_args(args)
        )
    except BaseException:
        action_log.close("policy_connection_error")
        raise
    print("✓ 连接成功")

    try:
        if not args.use_fake_obs:
            print("[2/2] 初始化环境...")
            env = create_robot_environment(
                control_mode=args.control_mode,
                hz=(
                    args.execution_frequency
                    if args.use_interpolation
                    else args.control_frequency
                ),
                trajectory_time=trajectory_time,
                min_ee_base_z=(
                    args.min_ee_base_z
                    if args.control_mode == "camera_position"
                    else None
                ),
                min_left_ee_base_z=(
                    args.min_left_ee_base_z
                    if args.control_mode == "camera_position"
                    else None
                ),
                min_right_ee_base_z=(
                    args.min_right_ee_base_z
                    if args.control_mode == "camera_position"
                    else None
                ),
                # Only this compatibility entry builds the legacy 20-D joint
                # observation.  Thin joint policies remain 16-D and avoid two
                # unnecessary hardware polls on every RTC observation.
                include_auxiliary_state=(args.control_mode == "joint_position"),
            )
            print("✓ 环境就绪\n")
        else:
            print("[2/2] 测试模式，跳过环境初始化\n")
            env = None
    except BaseException:
        policy_client.close()
        action_log.close("environment_startup_error")
        raise

    episode_idx = 0

    try:
        while True:
            prompt_profile = (
                "ee" if args.control_mode == "camera_position" else "joint"
            )
            instruction = select_and_confirm_prompt(
                prompt_profile,
                supplied_prompt=args.prompt,
                prompt_key=args.prompt_key,
            )
            if instruction == "q":
                break
            action_log.begin_episode(instruction)
            
            # Prompt confirmation happens before reset and execution.
            if env is not None:
                obs = env.reset(task_name=args.task_name)
            else:
                obs = get_fake_obs(args.control_mode)

            if args.control_mode == "camera_position":
                previous_robot_command = policy_ee.state_ee_to_action_command(
                    obs["state/pose"]
                )
            else:
                previous_robot_command = policy_joint.checked_joint_vector(
                    obs["state/joint"], name="initial compatibility joint command"
                )

            pred_action_chunk = None
            chunk_idx = 0
            first_request_data = None
            
            interpolated_actions = None
            action_exec_idx = 0
            total_interpolated_actions = 0
            
            # 外插相关
            extrapolation_count = 0
            is_extrapolating = False
            
            # 可视化数据
            traj_timestamps = []
            traj_left_xyz = []
            traj_right_xyz = []
            raw_timestamps = []
            raw_left_xyz = []
            raw_right_xyz = []
            episode_start_time = time.perf_counter()

            bar = tqdm.tqdm(range(args.max_timesteps), desc="推理中", unit="step")

            episode_reason = "completed"
            try:
                for t_step in bar:
                    step_start = time.perf_counter()
                    
                    # 判断是否需要推理
                    need_infer = (
                        interpolated_actions is None or 
                        action_exec_idx >= total_interpolated_actions
                    )
                    
                    if need_infer:
                        # 外插处理
                        if args.extrapolation_mode != 'none' and interpolated_actions is not None:
                            if extrapolation_count < args.max_extrapolation_steps:
                                is_extrapolating = True
                                extrapolation_count += 1
                                
                                if args.extrapolation_mode == 'hold':
                                    last_action = interpolated_actions[-1].copy()
                                    interpolated_actions = np.vstack([interpolated_actions, last_action[np.newaxis, :]])
                                    total_interpolated_actions += 1
                                    
                                    if args.debug:
                                        print(f"[外插-保持] 步数: {extrapolation_count}/{args.max_extrapolation_steps}")
                                
                                elif args.extrapolation_mode == 'linear':
                                    if len(interpolated_actions) >= 2:
                                        last_action = interpolated_actions[-1]
                                        second_last_action = interpolated_actions[-2]
                                        
                                        new_actions = extrapolate_linear(
                                            last_action,
                                            second_last_action,
                                            args.control_mode,
                                            num_steps=1
                                        )
                                        
                                        interpolated_actions = np.vstack([interpolated_actions, new_actions])
                                        total_interpolated_actions += 1
                                        
                                        if args.debug:
                                            print(f"[外插-线性] 步数: {extrapolation_count}/{args.max_extrapolation_steps}")
                                    else:
                                        last_action = interpolated_actions[-1].copy()
                                        interpolated_actions = np.vstack([interpolated_actions, last_action[np.newaxis, :]])
                                        total_interpolated_actions += 1
                                
                                need_infer = False
                            else:
                                print(f"[警告] 达到最大外插步数 {args.max_extrapolation_steps}，等待推理...")
                                need_infer = True
                        
                        # 执行推理
                        if need_infer:
                            chunk_idx += 1
                            
                            # 1. 获取观测
                            t0 = time.perf_counter()
                            if env is not None:
                                obs = env.get_obs()
                            else:
                                obs = get_fake_obs(args.control_mode)
                            t_get_obs = time.perf_counter() - t0
                            print(f"[性能] 获取观测 耗时: {t_get_obs*1000:.2f} ms")

                            # 2. 打包观测
                            t0 = time.perf_counter()
                            request_data = prepare_observation_for_policy(obs, args.control_mode)
                            request_data["prompt"] = instruction
                            t_pack = time.perf_counter() - t0
                            print(f"[性能] 打包观测 耗时: {t_pack*1000:.2f} ms")
                            
                            # 存储第一帧
                            if args.save_images and first_request_data is None:
                                first_request_data = request_data.copy()
                            
                            print_first_request_data(request_data, chunk_idx)

                            # 3. 推理
                            t0 = time.perf_counter()
                            response = policy_client.infer(request_data)
                            t_infer = time.perf_counter() - t0
                            print(f"[性能] 策略推理 耗时: {t_infer*1000:.2f} ms")
                            
                            # 4. 提取并严格校验服务端 Hx18/Hx20 动作。
                            pred_action_chunk = extract_compat_actions(
                                response, args.control_mode
                            )
                            
                            # 记录原始输出（可视化）
                            if args.debug and args.control_mode == "camera_position":
                                current_offset = time.perf_counter() - episode_start_time
                                for i in range(min(args.open_loop_horizon, len(pred_action_chunk))):
                                    raw_timestamps.append(current_offset + i * (1.0 / args.control_frequency))
                                    raw_left_xyz.append(pred_action_chunk[i, 0:3].copy())
                                    raw_right_xyz.append(pred_action_chunk[i, 7:10].copy())

                            # 5. 插值处理
                            t0 = time.perf_counter()
                            if args.use_interpolation:
                                interpolated_actions = interpolate_action_chunk(
                                    pred_action_chunk,
                                    args.control_mode,
                                    args.control_frequency,
                                    args.execution_frequency,
                                    args.open_loop_horizon,
                                    args.gripper_threshold,
                                )
                                total_interpolated_actions = len(interpolated_actions)
                                print(f"[插值] 原始 {args.open_loop_horizon} 个动作 -> 插值后 {total_interpolated_actions} 个动作")
                            else:
                                # 不插值：直接处理
                                processed = []
                                for i in range(min(args.open_loop_horizon, len(pred_action_chunk))):
                                    action = process_single_action(
                                        pred_action_chunk[i], 
                                        args.control_mode, 
                                        args.gripper_threshold
                                    )
                                    processed.append(action)
                                interpolated_actions = np.array(processed, dtype=np.float32)
                                total_interpolated_actions = len(interpolated_actions)
                            
                            t_interp = time.perf_counter() - t0
                            print(f"[性能] 插值处理 耗时: {t_interp*1000:.2f} ms")
                            
                            # 线性外插：丢弃前N个动作
                            if is_extrapolating and args.extrapolation_mode == 'linear' and args.extrapolation_skip > 0:
                                skip_count = min(args.extrapolation_skip, total_interpolated_actions - 1)
                                if skip_count > 0:
                                    interpolated_actions = interpolated_actions[skip_count:]
                                    total_interpolated_actions = len(interpolated_actions)
                                    print(f"[外插-丢弃] 丢弃前 {skip_count} 个动作，剩余 {total_interpolated_actions} 个")
                            
                            # 重置外插计数
                            extrapolation_count = 0
                            is_extrapolating = False
                            action_exec_idx = 0

                            if args.debug:
                                print(f"\n[DEBUG] Chunk #{chunk_idx} | 原始 shape: {pred_action_chunk.shape}")
                                print(f"[DEBUG] 插值后动作数: {total_interpolated_actions}")

                    # 6. 获取当前动作
                    action = interpolated_actions[action_exec_idx]
                    
                    # 7. 执行动作
                    t0 = time.perf_counter()
                    command_source = policy_action_to_robot_command(
                        action, args.control_mode
                    )
                    if args.control_mode == "camera_position":
                        robot_command = policy_ee.process_ee_action(
                            command_source,
                            previous_robot_command,
                            max_position_delta=args.max_position_delta,
                            max_rotation_delta=args.max_rotation_delta,
                            # process_single_action already applied this legacy
                            # entry's threshold; retain those exact endpoints.
                            binarize_gripper=False,
                        )
                    else:
                        robot_command = policy_joint.process_robot_action(
                            command_source,
                            previous_robot_command,
                            gripper_threshold=args.gripper_threshold,
                            max_joint_delta=args.max_joint_delta,
                        )
                    previous_robot_command = robot_command.copy()
                    if env is not None:
                        env.step(robot_command, update_obs=False)
                    action_log.log_action(
                        command_source,
                        robot_command,
                        episode_step=t_step + 1,
                        source=f"compat_{args.control_mode}_effective_target",
                        environment=env,
                    )
                    t_step_time = time.perf_counter() - t0
                    
                    # 记录执行轨迹（可视化）
                    if args.debug and args.control_mode == "camera_position":
                        traj_timestamps.append(time.perf_counter() - episode_start_time)
                        traj_left_xyz.append(robot_command[0:3].copy())
                        traj_right_xyz.append(robot_command[7:10].copy())

                    action_exec_idx += 1

                    # 计算总耗时
                    step_elapsed = time.perf_counter() - step_start
                    
                    if need_infer and not is_extrapolating:
                        print(f"[性能] 执行动作 耗时: {t_step_time*1000:.2f} ms")
                        print(f"[性能] 本步总耗时: {step_elapsed*1000:.2f} ms")
                        print("-" * 50)

                    sleep_time = execution_period - step_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n[中断]")
                episode_reason = "keyboard_interrupt"
            except Exception as e:
                episode_reason = "fatal_error"
                print(f"\n[错误] {e}")
                import traceback
                traceback.print_exc()
                # Hardware, protocol, or safety-validation failures are fatal
                # for this compatibility process.  Do not offer another
                # episode on a possibly invalid robot/policy state.
                raise
            finally:
                action_log.end_episode(episode_reason)
                # 保存图像
                if args.save_images and first_request_data is not None and images_dir:
                    print(f"\n[保存图像] Episode {episode_idx}...")
                    save_first_request_images(first_request_data, images_dir, episode_idx)
                    print("✓ 图像已保存")
                
                # 绘制轨迹图
                if args.debug and args.control_mode == "camera_position" and len(traj_timestamps) > 0:
                    print(f"\n[绘制轨迹] Episode {episode_idx}...")
                    plot_trajectory_xyz(
                        timestamps=traj_timestamps,
                        left_xyz=traj_left_xyz,
                        right_xyz=traj_right_xyz,
                        raw_times=raw_timestamps if raw_timestamps else None,
                        raw_left=raw_left_xyz if raw_left_xyz else None,
                        raw_right=raw_right_xyz if raw_right_xyz else None,
                        episode_idx=episode_idx,
                        output_dir=Path("./images"),
                    )
                    print("✓ 轨迹图已保存")

            episode_idx += 1

            try:
                if input("\n继续? (y/n): ").lower() != "y":
                    break
            except (KeyboardInterrupt, EOFError):
                print("\n[退出]")
                break

    except KeyboardInterrupt:
        print("\n[退出] 收到中断信号")
    finally:
        print("\n正在关闭...")
        action_log.close()
        policy_client.close()
        if env is not None:
            env.close()
        print("✓ 完成")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
