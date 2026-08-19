"""G1 真实机器人、相机与轨迹控制的底层节点接口。

输入包括 16 维关节/夹爪命令及绝对或增量位姿字典，输出为最新图像、关节状态和运动状态。
构造 ``RobotNode`` 时才加载 a2d_sdk 并创建 DDS、相机和控制器，普通模块导入不要求机器人环境。
Ruckig 仅在平滑关节轨迹方法中加载，SciPy 仅在欧拉角转轴角的查询方法中加载。
``robot_interface_wrapper`` 继承本类，随后由 ``RobotEnv`` 和各推理入口间接使用。
构造过程会阻塞等待首帧图像与关节数据；超过5秒会低频告警但继续等待，状态轮询也可能等待至超时。
所有 publish/move 方法都会影响真实硬件；本类无锁且部分方法内部 sleep，不宜被无协调地并发调用。
"""

import time
import numpy as np


def _is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

class RobotNode:
    """
    单线程 / 无锁版本 SimNode，接口与 ROS 版本保持一致。
    每次 get_* 都实时拉取最新数据；publish 直接下发。
    """

    def __init__(self, hz: float = 30.0):
        try:
            from a2d_sdk.robot import CosineCamera, RobotController, RobotDds
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "RobotNode requires a2d_sdk. Source the robot SDK environment "
                "before constructing the real-robot interface."
            ) from error

        self.robot   = RobotDds()
        self.camera  = CosineCamera(["head", "hand_left", "hand_right"])
        self.robot_controller = RobotController()
        self.period  = 1.0 / hz
        self._t0     = time.time()
        self._smooth_command_count = 0

        # 先等底层数据就绪
        self._wait_until_ready()

    # -------------------------------------
    def _wait_until_ready(self):
        """阻塞直到首批硬件数据就绪，并对异常长等待给出低频警告。

        正常部署默认会很快获得数据，因此不设置强制终止时间。超过5秒后打印
        第一次 warning，此后每10秒提醒一次仍未就绪的通道，便于区分模型启动
        较慢与相机/DDS连接异常；等待本身继续进行，Ctrl-C仍可人工终止。
        """

        started = time.monotonic()
        next_warning = started + 5.0
        while True:
            img, _ = self.camera.get_latest_image("head")
            pos, _ = self.robot.arm_joint_states()
            grip, _ = self.robot.gripper_states()
            if img is not None and pos is not None and grip is not None:
                break
            now = time.monotonic()
            if now >= next_warning:
                missing = []
                if img is None:
                    missing.append("head_camera")
                if pos is None:
                    missing.append("arm_joint_states")
                if grip is None:
                    missing.append("gripper_states")
                print(
                    "[RobotNode] warning: hardware initialization is still "
                    f"waiting after {now - started:.1f}s; missing="
                    f"{','.join(missing) or 'unknown'}. Will keep waiting; "
                    "press Ctrl-C to stop.",
                    flush=True,
                )
                next_warning = now + 10.0
            time.sleep(0.01)

    def poll_state(self, getter, length, name, timeout=2.0, interval=0.1, verbose=False):
        deadline = time.time() + timeout
        last_vals = None

        while time.time() < deadline:
            vals, _ = getter()
            last_vals = vals
            # 获取长度
            try:
                actual_len = len(vals)
            except Exception:
                actual_len = None

            # 判断每个元素是否可转 float
            all_numeric = (actual_len == length) and all(
                _is_number(v) for v in vals
            )
            if all_numeric:
                if verbose:
                    print(f"✅ {name} 就绪")
                return list(vals)

            time.sleep(interval)

        raise RuntimeError(f"{name} 在 {timeout}s 内未就绪，最后 vals={last_vals!r}")

    # ------------ 原接口 ------------------
    def get_img_head(self):
        img, _ = self.camera.get_latest_image("head")
        return img
        #return img[:, :, ::-1] if img is not None else None

    def get_img_left_wrist(self):
        img, _ = self.camera.get_latest_image("hand_left")
        return img
        #return img[:, :, ::-1] if img is not None else None

    def get_img_right_wrist(self):
        img, _ = self.camera.get_latest_image("hand_right")
        return img
        #return img[:, :, ::-1] if img is not None else None

    def get_joint_state(self):
        """返回 16 维 [左7 左夹 右7 右夹]"""
        pos, _  = self.robot.arm_joint_states()
        grip, _ = self.robot.gripper_states()
        if pos is None or grip is None:
            return None
        #left_g  = max(0.0, min(1.0, 0.8 - grip[0])) if grip[0] is not None else 0.0
        #right_g = max(0.0, min(1.0, 0.8 - grip[1])) if grip[1] is not None else 0.0

        #import pdb; pdb.set_trace()

        left_g = grip[0]
        right_g = grip[1]

        return pos[:7] + [left_g] + pos[7:] + [right_g]

    def get_head_joint_states(self):
        """返回弧度制头部关节角，便于后续数值运算"""
        head_states = self.poll_state(self.robot.head_joint_states, 2, "head_joint_states", verbose=False)
        if head_states is None:
            return None
        return np.deg2rad(head_states).tolist()

    def get_waist_joint_states(self):
        return self.poll_state(self.robot.waist_joint_states, 2, "waist_joint_states", verbose=False)

    def get_arm_joint_states(self):
        return self.poll_state(self.robot.arm_joint_states, 14, "arm_joint_states", verbose=False)

    def publish_joint_command(self, action):
        """action: 16 维"""
        try:
            import ruckig
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Smooth joint trajectories require the ruckig package."
            ) from error

        target_positions = np.concatenate((action[:7], action[8:15]))

        # Get current joint states
        current_positions, _ = self.robot.arm_joint_states()
        if not current_positions:
            raise Exception("Failed to get arm joint states")

        dof = 14  # Degrees of freedom
        interval = 0.01  # Time interval
        rk = ruckig.Ruckig(dof, interval)
        rk_input = ruckig.InputParameter(dof)
        rk_output = ruckig.OutputParameter(dof)

        # Set current state
        rk_input.current_position = current_positions
        rk_input.current_velocity = [0.0] * dof
        rk_input.current_acceleration = [0.0] * dof

        # Set target state
        rk_input.target_position = target_positions
        rk_input.target_velocity = [0.0] * dof
        rk_input.target_acceleration = [0.0] * dof

        # Set motion constraints
        rk_input.max_velocity = [2.0] * dof
        rk_input.max_acceleration = [1.0] * dof
        rk_input.max_jerk = [5.0] * dof

        # Generate trajectory
        #print("Generating trajectory...")
        trajs = []
        result = rk.update(rk_input, rk_output)
        while result == ruckig.Result.Working:
            trajs.append(rk_output.new_position)
            rk_output.pass_to_input(rk_input)
            result = rk.update(rk_input, rk_output)
        if result != ruckig.Result.Finished:
            raise RuntimeError(f"Ruckig trajectory generation failed: {result}")
        self._smooth_command_count += 1
        if (
            self._smooth_command_count == 1
            or self._smooth_command_count % 30 == 0
        ):
            print(
                "[RobotNode] smooth joint trajectory "
                f"command={self._smooth_command_count}, points={len(trajs)}, "
                f"nominal_duration={len(trajs) * interval:.3f}s"
            )
        #print(f"Generated {len(trajs)} trajectory points")
        #print("Executing trajectory...")
        for i, traj in enumerate(trajs):
            #print(f"Executing point {i+1}/{len(trajs)}")
            self.robot.move_arm(traj)
            time.sleep(0.01)
        # sleep 已移除，由 robot_env.py 统一控制

        self.robot.move_gripper([action[7], action[15]])

    def publish_joint_command_direct(self, action):
        """
        直接发送关节命令，不进行轨迹插值
        action: 16 维 (左臂7维 + 左夹爪1维 + 右臂7维 + 右夹爪1维)
        """
        target_positions = np.concatenate((action[:7], action[8:15]))

        # Get current joint states for logging
        # current_positions, _ = self.robot.arm_joint_states()
        # if not current_positions:
        #     raise Exception("Failed to get arm joint states")

        # print(f"Moving from current positions to target...")
        # print(f"Current positions: {current_positions}")
        # print(f"Target positions: {target_positions}")

        # Directly send target position to robot
        self.robot.move_arm(target_positions)
        
        # Move gripper
        self.robot.move_gripper([action[7], action[15]])
        time.sleep(0.01)
        
        # print("Command sent to robot")

    def publish_head_command(self, target_positions):
        print("开始头部运动测试...")
        # 获取当前头部关节状态
        current_positions, _ = self.robot.head_joint_states()
        if not current_positions:
            raise Exception("获取头部关节状态失败")
        print(f"当前头部位置: {current_positions}")
        print(f"目标头部位置: {target_positions}")
        self.robot.move_head(target_positions)
        time.sleep(2)
        print("头部运动完成。\n")

    def publish_waist_command(self, target_positions):
        print("开始腰部运动测试...")
        current_positions, _ = self.robot.waist_joint_states()
        if not current_positions:
            raise Exception("获取腰部关节状态失败")
        print(f"当前腰部位置: {current_positions}")
        print(f"目标腰部位置: {target_positions}")
        self.robot.move_waist(target_positions)
        time.sleep(2)
        print("腰部运动完成。\n")

    def publish_delta_pose_command(self, action: dict):
        """
        调用 SDK 下发相对位姿轨迹控制命令。
        """
        # 定义机器人状态字典
        robot_states = {
            "head":  action["head_joint_states"],
            "waist": action["waist_joint_states"],
            "arm":   action["arm_joint_states"],
        }
        # 定义机器人动作列表
        robot_actions = [
            {
                "left_arm":  {"action_data": delta[:6],  "control_type": "DELTA_POSE"},
                "right_arm": {"action_data": delta[6:12], "control_type": "DELTA_POSE"},
            }
            for delta in action["arm_cmd"]
        ]
        # 调用 SDK 下发相对位姿轨迹控制命令
        self.robot_controller.trajectory_tracking_control(
            infer_timestamp      = action["observation_timestamp"],#必须有的时间
            robot_states         = robot_states,# 必须的参考状态，有了更好
            robot_actions        = robot_actions,
            robot_link           = "base_link",#基础坐标系
            trajectory_reference_time = 1.0,#运行的参考时间，越小运行的越快
        )


    def publish_abs_pose_command(self, action: dict):
        """
        调用 SDK 下发相对位姿轨迹控制命令。
        """
        # 定义机器人状态字典
        robot_states = {
            "head":  action["head_joint_states"],
            "waist": action["waist_joint_states"],
            "arm":   action["arm_joint_states"],
        }
        # 定义机器人动作列表
        robot_actions = [
            {
                "left_arm":  {"action_data": abs_pose[:6],  "control_type": "ABS_POSE"},
                "right_arm": {"action_data": abs_pose[6:12], "control_type": "ABS_POSE"},
            }
            for abs_pose in action["arm_cmd"]
        ]
        # 调用 SDK 下发相对位姿轨迹控制命令
        self.robot_controller.trajectory_tracking_control(
            infer_timestamp      = action["observation_timestamp"],#必须有的时间
            robot_states         = robot_states,# 必须的参考状态，有了更好
            robot_actions        = robot_actions,
            robot_link           = "base_link",#基础坐标系
            trajectory_reference_time = 1.0,#运行的参考时间，越小运行的越快
        )

    def publish_abs_pose_command_and_hand(self, action: dict, hand_action: np.ndarray, trajectory_reference_time: float = None):
        """
        调用 SDK 下发相对位姿轨迹控制命令。
        
        Args:
            action: 动作字典，包含 observation_timestamp, head_joint_states, waist_joint_states, arm_joint_states, arm_cmd
            hand_action: 夹爪动作 [左夹爪, 右夹爪]
            trajectory_reference_time: 轨迹参考时间（秒），如果为 None 则使用默认值 1.0
                                     越小运行越快，通常设置为 action_delay 的倍数（如 2-5倍）
        """
        if trajectory_reference_time is None:
            trajectory_reference_time = 1.0
        
        # 定义机器人状态字典
        robot_states = {
            "head":  action["head_joint_states"],
            "waist": action["waist_joint_states"],
            "arm":   action["arm_joint_states"],
        }
        # 定义机器人动作列表
        robot_actions = [
            {
                "left_arm":  {"action_data": abs_pose[:6],  "control_type": "ABS_POSE"},
                "right_arm": {"action_data": abs_pose[6:12], "control_type": "ABS_POSE"},
            }
            for abs_pose in action["arm_cmd"]
        ]

        # 调用 SDK 下发相对位姿轨迹控制命令
        self.robot_controller.trajectory_tracking_control(
            infer_timestamp      = action["observation_timestamp"],#必须有的时间
            robot_states         = robot_states,# 必须的参考状态，有了更好
            robot_actions        = robot_actions,
            robot_link           = "base_link",#基础坐标系
            trajectory_reference_time = trajectory_reference_time,#运行的参考时间，越小运行的越快
        )

        self.robot.move_gripper([hand_action[0], hand_action[1]])


    def get_clock(self):
        return time.time() - self._t0

    def loop_rate(self):
        time.sleep(self.period)

    def shutdown(self):
        self.robot.shutdown()

    def get_motion_status(self):
        return self.robot_controller.get_motion_status()


    def get_arm_pose_axis_angle(self):

        try:
            from scipy.spatial.transform import Rotation as R
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Axis-angle pose conversion requires scipy."
            ) from error

        motion_status = self.robot_controller.get_motion_status()
        if motion_status is not None:

            left_arm_pose = motion_status['frames']["arm_left_link7"]
            right_arm_pose = motion_status['frames']["arm_right_link7"]


            left_arm_position = np.array([left_arm_pose['position']['x'], left_arm_pose['position']['y'], left_arm_pose['position']['z']])
            right_arm_position = np.array([right_arm_pose['position']['x'], right_arm_pose['position']['y'], right_arm_pose['position']['z']])

            #left_arm_orientation = np.array([left_arm_pose['orientation']['quaternion']['x'], left_arm_pose['orientation']['quaternion']['y'], left_arm_pose['orientation']['quaternion']['z'], left_arm_pose['orientation']['quaternion']['w']])
            #right_arm_orientation = np.array([right_arm_pose['orientation']['quaternion']['x'], right_arm_pose['orientation']['quaternion']['y'], right_arm_pose['orientation']['quaternion']['z'], right_arm_pose['orientation']['quaternion']['w']])


            
            # 提取欧拉角 (roll, pitch, yaw)
            left_euler = np.array([left_arm_pose['orientation']['euler']['roll'], 
                                   left_arm_pose['orientation']['euler']['pitch'], 
                                   left_arm_pose['orientation']['euler']['yaw']])
            right_euler = np.array([right_arm_pose['orientation']['euler']['roll'], 
                                    right_arm_pose['orientation']['euler']['pitch'], 
                                    right_arm_pose['orientation']['euler']['yaw']])
            
            # 将欧拉角转换为轴角表示（axis-angle representation）
            # 欧拉角顺序: roll(X), pitch(Y), yaw(Z) -> 使用 'xyz' 顺序
            left_rotation = R.from_euler('xyz', left_euler, degrees=False)
            right_rotation = R.from_euler('xyz', right_euler, degrees=False)
            
            # 提取轴角表示 (旋转向量，其方向为旋转轴，模长为旋转角度)
            left_arm_orientation_axis_angle = left_rotation.as_rotvec()  # 返回 3维向量
            right_arm_orientation_axis_angle = right_rotation.as_rotvec()  # 返回 3维向量



            left_arm = np.concatenate((left_arm_position, left_arm_orientation_axis_angle))
            right_arm = np.concatenate((right_arm_position, right_arm_orientation_axis_angle))


            return left_arm, right_arm
        return None

    def get_arm_pose_euler(self):

        motion_status = self.robot_controller.get_motion_status()
        if motion_status is not None:

            left_arm_pose = motion_status['frames']["arm_left_link7"]
            right_arm_pose = motion_status['frames']["arm_right_link7"]

            left_arm = left_arm_pose['xyzrpy']
            right_arm = right_arm_pose['xyzrpy']

            return left_arm, right_arm
        return None

    
if __name__=="__main__":

    robot =  RobotNode()
    print(robot.get_arm_joint_states())
    print(robot.get_img_head())
