"""Agibot 真实机器人环境的 Gym 风格封装。

``joint_position`` 接收 16 维 ``[左7, 左夹, 右7, 右夹]``，``camera_position`` 接收 14 维双手位姿。
``get_obs`` 输出三路图像和关节状态；相机坐标模式还计算末端位姿、头部与腰部状态。
旧 20 维 joint 入口可通过 ``include_auxiliary_state`` 按需补充头腰，新 16 维入口不额外读取。
未注入 ``robot_node`` 时才加载 RobotNode，未注入 retargeter 的相机模式才加载外部坐标转换模块。
推理入口通常经 ``inference_common.create_robot_environment`` 构造本类，也可注入测试替身隔离硬件。
``reset`` 可调用任务初始姿态表，``step`` 直接下发关节或末端命令，循环频率主要由上层控制。
真实节点构造可能等待传感器；错误维度、失效标定或过快调用都可能导致执行失败或运动风险。
"""

import math
import time
import threading
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np


def _finite_vector(value: Any, *, size: int, name: str) -> np.ndarray:
    """Return one finite hardware state vector or fail instead of using stale data."""

    if value is None:
        raise RuntimeError(f"无法获取{name}，拒绝使用陈旧控制状态")
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (size,):
        raise RuntimeError(f"{name}维度异常：期望 ({size},)，得到 {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise RuntimeError(f"{name}包含 NaN 或 Inf")
    return vector


class RobotEnv:
    def __init__(
        self,
        control_mode: Literal["joint_position", "camera_position"] = "joint_position",
        hz: float = 30.0,
        use_smooth_trajectory: bool = False,
        trajectory_time: Optional[float] = None,
        retargeter_urdf_path: Optional[str] = None,
        retargeter_camera_extrinsic_path: Optional[str] = None,
        robot_node: Optional[Any] = None,
        retargeter: Optional[Any] = None,
        include_auxiliary_state: bool = False,
        freeze_arm: Literal["none", "left", "right"] = "none",
    ):
        """
        Args:
            control_mode: 控制模式
            hz: 基础频率（用于计算默认参数）
            use_smooth_trajectory: 是否使用平滑轨迹（仅 joint_position 模式）
            trajectory_time: camera_position 模式下轨迹执行时间（秒），默认 1/hz * 2
            retargeter_urdf_path: URDF 文件路径
            retargeter_camera_extrinsic_path: 相机外参文件路径
            robot_node: 可选的已构造机器人节点，主要用于测试或自定义后端
            retargeter: 可选的已构造坐标转换器
            include_auxiliary_state: joint 模式是否额外读取 2 维头部和
                2 维腰部状态；仅旧 20 维兼容策略需要
            freeze_arm: camera 模式的硬件隔离开关；在 reset 后锁住所选
                手臂的 6 维 ABS_POSE 目标和夹爪命令
        """
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError(f"hz must be finite and positive, got {hz}")
        if control_mode not in {"joint_position", "camera_position"}:
            raise ValueError(f"未知控制模式: {control_mode!r}")
        normalized_freeze_arm = str(freeze_arm).strip().casefold()
        if normalized_freeze_arm not in {"none", "left", "right"}:
            raise ValueError("freeze_arm must be one of: none, left, right")
        if normalized_freeze_arm != "none" and control_mode != "camera_position":
            raise ValueError("freeze_arm is only supported in camera_position mode")
        if trajectory_time is not None and (
            not math.isfinite(trajectory_time) or trajectory_time <= 0
        ):
            raise ValueError("trajectory_time must be finite and positive")
        self.control_mode = control_mode
        self.hz = hz
        self.use_smooth_trajectory = use_smooth_trajectory
        self.include_auxiliary_state = bool(include_auxiliary_state)
        self.freeze_arm = normalized_freeze_arm
        
        # 轨迹执行时间（camera_position 模式使用）
        self.trajectory_time = trajectory_time if trajectory_time is not None else (1.0 / hz) * 2.0

        if robot_node is None:
            from robot_interface_wrapper import RobotNode

            robot_node = RobotNode(hz=hz)
        self.robot_node = robot_node
        self.retargeter = retargeter
        # RTC inference/execution and rollout capture use separate threads.
        # Serialize all SDK and retargeter access because the hardware backend
        # does not advertise thread safety.  RLock permits step/reset to call
        # get_obs while already holding the same safety boundary.
        self._io_lock = threading.RLock()
        self.step_count = 0
        self.episode_count = 0
        self._last_obs = None
        self._last_executed_action: np.ndarray | None = None
        self._action_generation = 0
        self._sensor_fallback_count = 0
        self._max_sensor_fallbacks = max(1, int(round(hz)))
        self._frozen_arm_pose_anchor: np.ndarray | None = None
        self._frozen_gripper_anchor: float | None = None

        if self.control_mode == "camera_position" and self.retargeter is None:
            self._init_retargeter(retargeter_urdf_path, retargeter_camera_extrinsic_path)

        print(f"[RobotEnv] 模式: {control_mode}, hz: {hz}, traj_time: {self.trajectory_time:.3f}s")
        print("[RobotEnv] 硬件读写已启用串行锁（RTC/rollout 线程安全模式）")
        if self.freeze_arm != "none":
            print(
                f"[RobotEnv] 单臂硬件隔离已启用: freeze_arm={self.freeze_arm}; "
                "将在 reset 后锁定该臂 ABS_POSE 与夹爪"
            )

    def _init_retargeter(self, urdf_path: Optional[str], extrinsic_path: Optional[str]):
        try:
            from utils.agibot import g1_pos_transform
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "camera_position mode requires utils.agibot.g1_pos_transform. "
                "Add the real_robot_g1 project root to PYTHONPATH before starting."
            ) from error

        assets_dir = Path(g1_pos_transform.__file__).resolve().parent / "G1"
        if urdf_path is None:
            urdf_path = str(assets_dir / "model.urdf")
        if extrinsic_path is None:
            extrinsic_path = str(assets_dir / "head_extrinsic_ours.json")

        BodyRetargeter = g1_pos_transform.BodyRetargeter
        self.retargeter = BodyRetargeter(urdf_path=urdf_path, camera_extrinsic_path=extrinsic_path)
        print(f"[RobotEnv] 坐标转换器已加载")
    
    def get_obs(self) -> Dict[str, np.ndarray]:
        with self._io_lock:
            img_head = self.robot_node.get_img_head()
            img_left = self.robot_node.get_img_left_wrist()
            img_right = self.robot_node.get_img_right_wrist()
            # Control state is never reused: a stale joint/EE state can shift
            # the RTC timeline and produce an inaccurate physical command.
            joint_state = _finite_vector(
                self.robot_node.get_joint_state(), size=16, name="关节状态"
            )

            images = {
                "image/head": img_head,
                "image/left_wrist": img_left,
                "image/right_wrist": img_right,
            }
            missing_images = [key for key, value in images.items() if value is None]
            if missing_images:
                if self._last_obs is None:
                    raise RuntimeError(
                        "图像传感器读取失败且没有可复用的上一帧："
                        + ", ".join(missing_images)
                    )
                self._sensor_fallback_count += 1
                if self._sensor_fallback_count > self._max_sensor_fallbacks:
                    raise RuntimeError(
                        "图像传感器连续读取失败超过约1秒，拒绝继续使用陈旧图像"
                    )
                if (
                    self._sensor_fallback_count == 1
                    or self._sensor_fallback_count % 30 == 0
                ):
                    print(
                        "[RobotEnv] 部分图像读取失败，仅复用对应上一帧 "
                        f"(keys={missing_images}, "
                        f"连续次数={self._sensor_fallback_count})"
                    )
                for key in missing_images:
                    images[key] = self._last_obs[key]
            elif self._sensor_fallback_count:
                print(
                    "[RobotEnv] 图像读取已恢复 "
                    f"(此前连续复用={self._sensor_fallback_count})"
                )
                self._sensor_fallback_count = 0
        
            obs = {
                key: np.asarray(value, dtype=np.uint8).copy()
                for key, value in images.items()
            }
            obs["state/joint"] = joint_state.copy()
        
            if self.control_mode == "camera_position":
                obs.update(self._compute_pose_state(joint_state))
            elif self.include_auxiliary_state:
                # The legacy 20-D joint policy appends head2 + waist2 to the
                # 16-D arm/hand state even though RobotEnv executes only the
                # first 16 command dimensions.
                obs["state/head"] = _finite_vector(
                    self.robot_node.get_head_joint_states(),
                    size=2,
                    name="头部关节状态",
                )
                obs["state/waist"] = _finite_vector(
                    self.robot_node.get_waist_joint_states(),
                    size=2,
                    name="腰部关节状态",
                )

            # Keep a private cache: policy/recorder code may transform its
            # returned arrays and must not corrupt the next image fallback.
            self._last_obs = {key: value.copy() for key, value in obs.items()}
            return obs

    def _compute_pose_state(self, joint_state: np.ndarray) -> Dict[str, np.ndarray]:
        head_states = _finite_vector(
            self.robot_node.get_head_joint_states(), size=2, name="头部关节状态"
        )
        waist_states = _finite_vector(
            self.robot_node.get_waist_joint_states(), size=2, name="腰部关节状态"
        )
        # Derive both arms from the exact same joint16 sample returned in this
        # observation.  A second SDK poll here would let joint16 and FK pose
        # describe different instants while the robot is moving.
        arm_states = np.concatenate(
            [joint_state[:7], joint_state[8:15]], axis=0
        ).astype(np.float32, copy=False)
        
        state_vec = np.zeros((1, 53), dtype=np.float32)
        state_vec[0, 28:35] = arm_states[:7]
        state_vec[0, 35:42] = arm_states[7:]
        state_vec[0, 42:43] = joint_state[7]
        state_vec[0, 43:44] = joint_state[15]
        state_vec[0, 51:53] = waist_states
        state_vec[0, 26:28] = head_states

        (left_pos, left_aa), (right_pos, right_aa) = self.retargeter.process_kinematics(state_vec)
            
        pose = np.concatenate([
            left_pos[0], left_aa[0], [joint_state[7]],
            right_pos[0], right_aa[0], [joint_state[15]]
        ])

        return {
            "state/pose": pose.astype(np.float32),
            "state/head": head_states,
            "state/waist": waist_states,
        }

    def step(self, action: np.ndarray, update_obs: bool = True) -> Tuple[Optional[Dict], float, bool, Dict]:
        """
        执行动作。
        
        注意：此函数不包含 sleep，频率控制由调用方负责。
        """
        with self._io_lock:
            action = np.asarray(action, dtype=np.float32)
            if not np.all(np.isfinite(action)):
                raise ValueError("机器人动作包含 NaN 或 Inf，已拒绝下发")

            if self.control_mode == "joint_position":
                self._step_joint(action)
            elif self.control_mode == "camera_position":
                self._step_cartesian(action)
            else:
                raise ValueError(f"未知控制模式: {self.control_mode}")

            # Publish the command for rollout sampling only after the hardware
            # call returns successfully, while still holding the same I/O lock
            # used by get_obs_with_last_action().
            self._last_executed_action = action.copy()
            self._action_generation += 1
            self.step_count += 1
            obs = self.get_obs() if update_obs else None
        
        return obs, 0.0, False, {
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "control_mode": self.control_mode,
        }

    def _step_joint(self, action: np.ndarray):
        """关节位置控制（无 sleep）"""
        if action.shape != (16,):
            raise ValueError(f"joint_position 动作期望 (16,), 得到 {action.shape}")

        if self.use_smooth_trajectory:
            self.robot_node.publish_joint_command(action)
        else:
            self.robot_node.publish_joint_command_direct(action)

    def _step_cartesian(self, action: np.ndarray):
        """笛卡尔位置控制（无 sleep）"""
        if action.shape != (14,):
            raise ValueError(f"camera_position 动作期望 (14,), 得到 {action.shape}")

        hand_left, hand_right = action[6], action[13]
        head_states = _finite_vector(
            self.robot_node.get_head_joint_states(), size=2, name="头部关节状态"
        )
        waist_states = _finite_vector(
            self.robot_node.get_waist_joint_states(), size=2, name="腰部关节状态"
        )

        action_vec = np.zeros((1, 53), dtype=np.float32)
        action_vec[0, 51:53] = waist_states
        action_vec[0, 26:28] = head_states

        left_pos = action[:3].reshape(1, 3)
        left_aa = action[3:6].reshape(1, 3)
        right_pos = action[7:10].reshape(1, 3)
        right_aa = action[10:13].reshape(1, 3)

        (left_pos_base, left_euler), (right_pos_base, right_euler) = \
            self.retargeter.inverse_kinematics_from_camera_axisangle(
                left_pos, left_aa, right_pos, right_aa, action_vec
            )

        abs_action = _finite_vector(
            np.concatenate([
                left_pos_base[0], left_euler[0],
                right_pos_base[0], right_euler[0]
            ]),
            size=12,
            name="IK 绝对位姿命令",
        )

        arm_states = _finite_vector(
            self.robot_node.get_arm_joint_states(), size=14, name="手臂关节状态"
        )

        if self.freeze_arm != "none":
            if (
                self._frozen_arm_pose_anchor is None
                or self._frozen_gripper_anchor is None
            ):
                raise RuntimeError(
                    "单臂硬件隔离尚未建立 reset 锚点，已拒绝下发；请先调用 "
                    "RobotEnv.reset()"
                )
            if self.freeze_arm == "left":
                abs_action[:6] = self._frozen_arm_pose_anchor
                hand_left = self._frozen_gripper_anchor
            else:
                abs_action[6:12] = self._frozen_arm_pose_anchor
                hand_right = self._frozen_gripper_anchor

        action_abs = {
            "observation_timestamp": int(time.time() * 1e9),
            "head_joint_states": np.rad2deg(head_states).tolist(),
            "waist_joint_states": waist_states.tolist(),
            "arm_joint_states": arm_states.tolist(),
            "arm_cmd": [abs_action.tolist()],
        }

        hand_action = np.array([hand_left, hand_right], dtype=np.float32)
        self.robot_node.publish_abs_pose_command_and_hand(
            action_abs, hand_action, 
            trajectory_reference_time=self.trajectory_time
        )
    
    def reset(self, **kwargs) -> Dict[str, np.ndarray]:
        with self._io_lock:
            self.step_count = 0
            self.episode_count += 1
            # Never let a failed first read reuse state/images from the
            # previous episode as a new RTC command baseline.
            self._last_obs = None
            self._last_executed_action = None
            self._action_generation = 0
            self._sensor_fallback_count = 0
            self._frozen_arm_pose_anchor = None
            self._frozen_gripper_anchor = None

            if "task_name" in kwargs:
                from init_node_pos import init_node_pos

                init_node_pos(self.robot_node, kwargs["task_name"])
                time.sleep(2.0)
            obs = self.get_obs()
            if self.freeze_arm != "none":
                joint_state = _finite_vector(
                    obs["state/joint"], size=16, name="冻结锚点关节状态"
                )
                arm_pose_pair = self.robot_node.get_arm_pose_euler()
                if arm_pose_pair is None or len(arm_pose_pair) != 2:
                    raise RuntimeError(
                        "无法读取冻结手臂的当前 ABS_POSE，已拒绝开始单臂测试"
                    )
                if self.freeze_arm == "left":
                    frozen_pose = arm_pose_pair[0]
                    gripper_state = float(joint_state[7])
                else:
                    frozen_pose = arm_pose_pair[1]
                    gripper_state = float(joint_state[15])
                self._frozen_arm_pose_anchor = _finite_vector(
                    frozen_pose,
                    size=6,
                    name=f"{self.freeze_arm} arm ABS_POSE freeze anchor",
                )
                if not 0.0 <= gripper_state <= 120.0:
                    raise RuntimeError(
                        "冻结锚点夹爪状态必须位于 [0, 120]，得到 "
                        f"{gripper_state}"
                    )
                self._frozen_gripper_anchor = gripper_state / 120.0
                print(
                    f"[RobotEnv] {self.freeze_arm} arm freeze anchor captured: "
                    "ABS_POSE=6D, "
                    f"gripper_command={self._frozen_gripper_anchor:.4f}"
                )
            return obs
    
    def close(self):
        with self._io_lock:
            self.robot_node.shutdown()
        print("[RobotEnv] 已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # 便捷方法
    def get_joint_state(self) -> np.ndarray:
        with self._io_lock:
            return self.robot_node.get_joint_state()

    def get_obs_with_last_action(
        self,
    ) -> tuple[Dict[str, np.ndarray], np.ndarray | None, int]:
        """Atomically sample an observation and its latest successful command.

        Rollout capture uses this method so a hardware step cannot finish in
        the gap between reading the observation and choosing its action label.
        """

        with self._io_lock:
            obs = self.get_obs()
            action = (
                None
                if self._last_executed_action is None
                else self._last_executed_action.copy()
            )
            return obs, action, self._action_generation
    
    def get_images(self) -> Dict[str, np.ndarray]:
        with self._io_lock:
            return {
                "head": self.robot_node.get_img_head(),
                "left_wrist": self.robot_node.get_img_left_wrist(),
                "right_wrist": self.robot_node.get_img_right_wrist(),
            }
