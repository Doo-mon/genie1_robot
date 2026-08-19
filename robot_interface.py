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


MOTION_MODE_STOP = 0
MOTION_MODE_SERVO = 1
MOTION_MODE_PLANNING = 2


class MotionControlError(RuntimeError):
    """Raised when the SDK motion controller reports a non-zero error."""

    def __init__(self, *, code, message, mode, status):
        self.code = code
        self.message = str(message or "")
        self.mode = mode
        self.status = status
        super().__init__(
            "motion controller error: "
            f"code={self.code}, message={self.message!r}, mode={self.mode}"
        )


def _as_finite_vector(value, *, length: int, name: str) -> np.ndarray:
    """Normalize one SDK vector returned as list, tuple, or NumPy array."""

    if value is None:
        raise ValueError(f"{name} is None")
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} cannot be converted to a numeric vector") from error
    if vector.shape != (length,):
        raise ValueError(
            f"{name} must contain {length} values, got shape {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or Inf")
    return vector


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

        self._camera_names = ("head", "hand_left", "hand_right")
        self.robot   = RobotDds()
        self.camera  = CosineCamera(list(self._camera_names))
        self.robot_controller = RobotController()
        self.period  = 1.0 / hz
        self._t0     = time.time()
        self._smooth_command_count = 0
        self._robot_closed = False
        self._camera_closed = False
        self._servo_mode_confirmed = False

        # 先等底层数据就绪。如果构造期间被中断，也要释放已创建的
        # DDS 与相机资源，避免半构造对象遗留后台连接。
        try:
            self._wait_until_ready()
        except BaseException:
            try:
                self.shutdown()
            except Exception as cleanup_error:
                print(
                    "[RobotNode] warning: cleanup after initialization failure "
                    f"also failed: {cleanup_error}",
                    flush=True,
                )
            raise

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
            images = {
                name: self.camera.get_latest_image(name)[0]
                for name in self._camera_names
            }
            pos, _ = self.robot.arm_joint_states()
            grip, _ = self.robot.gripper_states()
            if (
                all(image is not None for image in images.values())
                and pos is not None
                and grip is not None
            ):
                break
            now = time.monotonic()
            if now >= next_warning:
                missing = [
                    f"{name}_camera"
                    for name, image in images.items()
                    if image is None
                ]
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
            try:
                vector = _as_finite_vector(vals, length=length, name=name)
            except ValueError:
                vector = None
            if vector is not None:
                if verbose:
                    print(f"✅ {name} 就绪")
                return vector.tolist()

            time.sleep(interval)

        raise RuntimeError(f"{name} 在 {timeout}s 内未就绪，最后 vals={last_vals!r}")

    def _validated_motion_status(self):
        """Return one valid, error-free SDK motion status dictionary."""

        try:
            status = self.robot_controller.get_motion_status()
        except Exception as error:
            self._request_motion_stop(
                "get_motion_status raised " f"{type(error).__name__}: {error}"
            )
            raise RuntimeError(
                "failed to read motion controller status; motion STOP was requested"
            ) from error
        if status is None:
            self._request_motion_stop("get_motion_status returned None")
            raise RuntimeError("motion controller status is unavailable")
        if not isinstance(status, dict):
            self._request_motion_stop("get_motion_status returned a non-dict value")
            raise RuntimeError(
                "motion controller status must be a dict, got "
                f"{type(status).__name__}"
            )

        mode = status.get("mode")
        error = status.get("error")
        if not isinstance(mode, dict) or "value" not in mode:
            self._request_motion_stop("motion status has no valid mode")
            raise RuntimeError(f"motion controller status has invalid mode: {mode!r}")
        if not isinstance(error, dict):
            self._request_motion_stop("motion status has no valid error field")
            raise RuntimeError(f"motion controller status has invalid error: {error!r}")

        mode_value = mode.get("value")
        error_code = error.get("code", 0)
        has_error = bool(error.get("has_error", False))
        try:
            error_code_nonzero = int(error_code) != 0
        except (TypeError, ValueError):
            error_code_nonzero = error_code not in (None, "", 0, False)
        if has_error or error_code_nonzero:
            self._request_motion_stop(
                f"motion error code={error_code}, "
                f"message={error.get('message', '')!r}"
            )
            raise MotionControlError(
                code=error_code,
                message=error.get("message", ""),
                mode=mode_value,
                status=status,
            )
        return status

    def _request_motion_stop(self, reason: str) -> None:
        """Best-effort STOP used only after an invalid/error motion status."""

        try:
            self.robot_controller.set_motion_stop()
        except Exception as stop_error:
            print(
                "[RobotNode] warning: motion status is unsafe and the SDK "
                f"STOP request also failed ({reason}): {stop_error}",
                flush=True,
            )
        else:
            print(
                "[RobotNode] unsafe motion status detected; requested motion "
                f"STOP ({reason})",
                flush=True,
            )

    def ensure_servo_mode(self, timeout: float = 5.0, interval: float = 0.05):
        """Enter Servo once, then reject any unexpected later mode change.

        The first EE command explicitly requests Servo mode and waits for the
        controller to report mode=1.  After that confirmation, STOP or PLANNING
        is never cleared automatically: it may represent an operator stop or a
        fault and therefore requires an explicit restart/recovery decision.
        """

        if timeout <= 0 or interval <= 0:
            raise ValueError("Servo confirmation timeout/interval must be positive")

        status = self._validated_motion_status()
        current_mode = status["mode"]["value"]
        if getattr(self, "_servo_mode_confirmed", False):
            if current_mode != MOTION_MODE_SERVO:
                self._request_motion_stop(
                    "motion controller left Servo mode after execution started: "
                    f"mode={current_mode}"
                )
                raise RuntimeError(
                    "motion controller left Servo mode after execution started: "
                    f"mode={current_mode}; refusing to clear STOP/PLANNING automatically"
                )
            return status

        if current_mode != MOTION_MODE_SERVO:
            print(
                "[RobotNode] requesting motion controller Servo mode (mode=1); "
                f"current_mode={current_mode}",
                flush=True,
            )
            self.robot_controller.set_motion_control_mode(MOTION_MODE_SERVO)

        deadline = time.monotonic() + timeout
        last_status = status
        while True:
            last_status = self._validated_motion_status()
            if last_status["mode"]["value"] == MOTION_MODE_SERVO:
                self._servo_mode_confirmed = True
                print("[RobotNode] motion controller Servo mode confirmed", flush=True)
                return last_status
            if time.monotonic() >= deadline:
                self._request_motion_stop(
                    "Servo mode confirmation timed out: "
                    f"last_mode={last_status['mode']['value']}"
                )
                raise RuntimeError(
                    "motion controller did not enter Servo mode within "
                    f"{timeout:.1f}s; last_mode={last_status['mode']['value']}"
                )
            time.sleep(interval)

    def check_motion_status(self, *, require_servo: bool = False):
        """Fail fast on SDK motion errors and optionally require Servo mode."""

        status = self._validated_motion_status()
        if require_servo and status["mode"]["value"] != MOTION_MODE_SERVO:
            raise RuntimeError(
                "motion controller is not in Servo mode: "
                f"mode={status['mode']['value']}"
            )
        return status

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

        try:
            arm = _as_finite_vector(pos, length=14, name="arm_joint_states")
            gripper = _as_finite_vector(grip, length=2, name="gripper_states")
        except ValueError as error:
            raise RuntimeError(f"机器人关节反馈格式异常: {error}") from error

        # Do not use ``+`` here: NumPy arrays would perform element-wise
        # addition instead of list concatenation and silently corrupt 16-D order.
        return np.concatenate(
            [arm[:7], gripper[:1], arm[7:], gripper[1:2]]
        ).tolist()

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

        action = _as_finite_vector(action, length=16, name="joint command")
        self.check_motion_status()
        target_positions = np.concatenate((action[:7], action[8:15]))

        # Get current joint states
        current_positions, _ = self.robot.arm_joint_states()
        try:
            current_positions = _as_finite_vector(
                current_positions, length=14, name="arm_joint_states"
            )
        except ValueError as error:
            raise RuntimeError(f"Failed to get arm joint states: {error}") from error

        dof = 14  # Degrees of freedom
        interval = 0.01  # Time interval
        rk = ruckig.Ruckig(dof, interval)
        rk_input = ruckig.InputParameter(dof)
        rk_output = ruckig.OutputParameter(dof)

        # Set current state
        rk_input.current_position = current_positions.tolist()
        rk_input.current_velocity = [0.0] * dof
        rk_input.current_acceleration = [0.0] * dof

        # Set target state
        rk_input.target_position = target_positions.tolist()
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
            if i % 10 == 0:
                self.check_motion_status()
            #print(f"Executing point {i+1}/{len(trajs)}")
            self.robot.move_arm(traj)
            time.sleep(0.01)
        # sleep 已移除，由 robot_env.py 统一控制

        self.robot.move_gripper([action[7], action[15]])
        self.check_motion_status()

    def publish_joint_command_direct(self, action):
        """
        直接发送关节命令，不进行轨迹插值
        action: 16 维 (左臂7维 + 左夹爪1维 + 右臂7维 + 右夹爪1维)
        """
        action = _as_finite_vector(action, length=16, name="joint command")
        self.check_motion_status()
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
        self.check_motion_status()
        time.sleep(0.01)
        
        # print("Command sent to robot")

    def publish_head_command(self, target_positions):
        print("开始头部运动测试...")
        # 获取当前头部关节状态
        current_positions, _ = self.robot.head_joint_states()
        try:
            current_positions = _as_finite_vector(
                current_positions, length=2, name="head_joint_states"
            )
            target_positions = _as_finite_vector(
                target_positions, length=2, name="head command"
            )
        except ValueError as error:
            raise RuntimeError(f"头部关节状态或命令异常: {error}") from error
        print(f"当前头部位置: {current_positions.tolist()}")
        print(f"目标头部位置: {target_positions.tolist()}")
        self.robot.move_head(target_positions.tolist())
        time.sleep(2)
        print("头部运动完成。\n")

    def publish_waist_command(self, target_positions):
        print("开始腰部运动测试...")
        current_positions, _ = self.robot.waist_joint_states()
        try:
            current_positions = _as_finite_vector(
                current_positions, length=2, name="waist_joint_states"
            )
            target_positions = _as_finite_vector(
                target_positions, length=2, name="waist command"
            )
        except ValueError as error:
            raise RuntimeError(f"腰部关节状态或命令异常: {error}") from error
        print(f"当前腰部位置: {current_positions.tolist()}")
        print(f"目标腰部位置: {target_positions.tolist()}")
        self.robot.move_waist(target_positions.tolist())
        time.sleep(2)
        print("腰部运动完成。\n")

    def publish_delta_pose_command(self, action: dict):
        """
        调用 SDK 下发相对位姿轨迹控制命令。
        """
        self.ensure_servo_mode()

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
        self.check_motion_status(require_servo=True)


    def publish_abs_pose_command(self, action: dict):
        """
        调用 SDK 下发相对位姿轨迹控制命令。
        """
        self.ensure_servo_mode()

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
        self.check_motion_status(require_servo=True)

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

        self.ensure_servo_mode()
        
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

        self.check_motion_status(require_servo=True)

        self.robot.move_gripper([hand_action[0], hand_action[1]])


    def get_clock(self):
        return time.time() - self._t0

    def loop_rate(self):
        time.sleep(self.period)

    def shutdown(self):
        """Release both DDS and camera resources; safe to call repeatedly."""

        errors = []
        if not getattr(self, "_robot_closed", False):
            try:
                self.robot.shutdown()
            except Exception as error:
                errors.append(f"RobotDds.shutdown: {error}")
            else:
                self._robot_closed = True

        if not getattr(self, "_camera_closed", False):
            try:
                self.camera.close()
            except Exception as error:
                errors.append(f"CosineCamera.close: {error}")
            else:
                self._camera_closed = True

        if errors:
            raise RuntimeError("; ".join(errors))

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
