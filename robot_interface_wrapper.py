"""面向推理控制循环的 RobotNode 轻量兼容封装。

本类继承 ``robot_interface.RobotNode``，仅覆盖直接关节命令以移除原实现末尾的固定 sleep。
输入仍是 16 维 ``[左臂7, 左夹爪, 右臂7, 右夹爪]``，调用后直接发送双臂和夹爪目标。
模块导入只加载基类定义；a2d_sdk 仍在实际构造 RobotNode 时由基类延迟加载。
``RobotEnv`` 在没有注入测试节点时通过末尾的 ``RobotNode`` 别名实例化此封装。
除直接关节方法外，图像、状态、轨迹和关闭接口全部沿用原始节点行为。
该路径跳过插值并立即驱动硬件，且不负责限频或维度校验，调用方必须实施安全限制和周期控制。
"""

import numpy as np


from robot_interface import RobotNode as _OriginalRobotNode


class RobotNodeWrapper(_OriginalRobotNode):
    """
    RobotNode 封装器，移除内部 sleep，频率控制由调用方负责。
    
    覆盖的方法：
        - publish_joint_command_direct: 移除 time.sleep(0.01)
    
    所有其他方法保持原样。
    """
    
    def publish_joint_command_direct(self, action):
        """
        直接发送关节命令，不进行轨迹插值。
        action: 16 维 (左臂7维 + 左夹爪1维 + 右臂7维 + 右夹爪1维)
        
        注意：此函数不包含 sleep，频率控制由调用方负责。
        """
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (16,):
            raise ValueError(f"joint command must have shape (16,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("joint command contains NaN or Inf")
        target_positions = np.concatenate((action[:7], action[8:15]))
        self.robot.move_arm(target_positions)
        self.robot.move_gripper([action[7], action[15]])



RobotNode = RobotNodeWrapper
