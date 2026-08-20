#!/usr/bin/env python3
"""只读监测 G1 左右末端的 base_link 高度与夹爪反馈。

启动命令：``python3 monitor_ee_base_z.py``。
脚本不会切换运动模式、不会 reset、不会发送手臂或夹爪命令；它只读取
``RobotController.get_motion_status()`` 中左右 ``link7`` 的 base_link 位姿，
以及 ``RobotDds.gripper_states()`` 返回的夹爪原始反馈。手动移动机械臂时，
终端会实时显示左右 XYZ、当前 z、本次运行观测到的最小 z 和夹爪值。

要标定 ``MIN_LEFT_EE_BASE_Z`` / ``MIN_RIGHT_EE_BASE_Z``，请分别把左右
夹爪/工具最低点移动到仍允许的最低安全位置，记录对应 link7 z，并额外保留
安全余量。旧 ``MIN_EE_BASE_Z`` 仍可作为左右共同回退值。
Ctrl-C 只停止本脚本、释放 DDS，并打印本次统计，不会主动关闭调用它的终端。
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from typing import Any


LEFT_FRAME = "arm_left_link7"
RIGHT_FRAME = "arm_right_link7"


def _finite_vector(value: Any, *, size: int, name: str) -> tuple[float, ...]:
    """将 SDK 的 list/tuple/NumPy 返回值统一为有限浮点向量。"""

    if value is None:
        raise RuntimeError(f"{name} is None")
    try:
        if isinstance(value, (str, bytes)):
            raise TypeError("text is not a numeric vector")
        if hasattr(value, "reshape") and hasattr(value, "tolist"):
            raw_values = value.reshape(-1).tolist()
        else:
            raw_values = list(value)
        vector = tuple(float(item) for item in raw_values)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} cannot be converted to a numeric vector") from error
    if len(vector) != size:
        raise RuntimeError(f"{name} must contain {size} values, got {len(vector)}")
    if not all(math.isfinite(item) for item in vector):
        raise RuntimeError(f"{name} contains NaN or Inf")
    return vector


def _frame_xyz(frames: dict[str, Any], frame_name: str) -> tuple[float, ...]:
    """读取一个 motion-status frame 的 base_link XYZ。"""

    if frame_name not in frames:
        raise RuntimeError(f"motion status is missing frame {frame_name!r}")
    frame = frames[frame_name]
    if not isinstance(frame, dict):
        raise RuntimeError(f"frame {frame_name!r} must be a dict")

    # 当前 SDK 提供 xyzrpy；保留 position 形式作为兼容读取路径。
    xyzrpy = frame.get("xyzrpy")
    if xyzrpy is not None:
        return _finite_vector(xyzrpy, size=6, name=f"{frame_name}.xyzrpy")[:3]

    position = frame.get("position")
    if not isinstance(position, dict):
        raise RuntimeError(
            f"frame {frame_name!r} has neither xyzrpy nor a position dict"
        )
    try:
        xyz = [position[axis] for axis in ("x", "y", "z")]
    except KeyError as error:
        raise RuntimeError(
            f"frame {frame_name!r}.position is missing {error.args[0]!r}"
        ) from error
    return _finite_vector(xyz, size=3, name=f"{frame_name}.position")


def _read_sample(
    robot: Any, controller: Any
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """原子意义上尽量接近地读取左右位姿和两路夹爪反馈。"""

    status = controller.get_motion_status()
    if not isinstance(status, dict):
        raise RuntimeError(
            "get_motion_status() must return a dict, got "
            f"{type(status).__name__}"
        )
    frames = status.get("frames")
    if not isinstance(frames, dict):
        raise RuntimeError("motion status has no valid frames dict")

    left_xyz = _frame_xyz(frames, LEFT_FRAME)
    right_xyz = _frame_xyz(frames, RIGHT_FRAME)

    gripper_result = robot.gripper_states()
    if not isinstance(gripper_result, (tuple, list)) or not gripper_result:
        raise RuntimeError(
            "gripper_states() must return (values, metadata/timestamp)"
        )
    grippers = _finite_vector(
        gripper_result[0], size=2, name="gripper_states"
    )
    return left_xyz, right_xyz, grippers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only monitor for left/right link7 base_link XYZ and raw "
            "gripper feedback. No robot command is sent."
        )
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=10.0,
        help="sampling/printing frequency in Hz (default: 10)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not math.isfinite(args.frequency) or args.frequency <= 0:
        raise ValueError("--frequency must be finite and positive")

    try:
        from a2d_sdk.robot import RobotController, RobotDds
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "找不到 a2d_sdk；请先 source 真机 SDK 环境，再启动本脚本"
        ) from error

    robot = None
    controller = None
    left_min = math.inf
    right_min = math.inf
    valid_samples = 0
    period = 1.0 / args.frequency
    next_tick = time.monotonic()
    last_error_text = ""
    last_error_time = -math.inf

    print("=" * 88)
    print("G1 EE base_link 高度/夹爪只读监测")
    print(f"采样频率: {args.frequency:g} Hz")
    print("不会发送任何动作。请手动移动机械臂；按 Ctrl-C 停止并查看最小值。")
    print("夹爪值为 SDK 原始反馈；z 为 arm_*_link7 原点，不是夹爪最低点。")
    print("=" * 88)

    try:
        robot = RobotDds()
        controller = RobotController()
        while True:
            try:
                left_xyz, right_xyz, grippers = _read_sample(robot, controller)
                left_min = min(left_min, float(left_xyz[2]))
                right_min = min(right_min, float(right_xyz[2]))
                valid_samples += 1
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                line = (
                    f"[{timestamp}] "
                    f"L xyz=({left_xyz[0]:+.4f},{left_xyz[1]:+.4f},{left_xyz[2]:+.4f})m "
                    f"min_z={left_min:+.4f}m grip={grippers[0]:.4f} | "
                    f"R xyz=({right_xyz[0]:+.4f},{right_xyz[1]:+.4f},{right_xyz[2]:+.4f})m "
                    f"min_z={right_min:+.4f}m grip={grippers[1]:.4f}"
                )
                print("\r" + line.ljust(190), end="", flush=True)
            except Exception as error:
                # 读取短暂抖动不退出；同一错误每秒最多另起一行提示一次。
                now = time.monotonic()
                error_text = f"{type(error).__name__}: {error}"
                if error_text != last_error_text or now - last_error_time >= 1.0:
                    print(f"\n[monitor] warning: {error_text}", flush=True)
                    last_error_text = error_text
                    last_error_time = now

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\n[monitor] 已收到 Ctrl-C，停止只读监测。")
    finally:
        if robot is not None:
            try:
                robot.shutdown()
            except Exception as error:
                print(f"[monitor] warning: RobotDds.shutdown failed: {error}")

    if valid_samples:
        print(
            "[monitor] 本次观测最小 link7 base_link z: "
            f"left={left_min:.4f} m, right={right_min:.4f} m, "
            f"valid_samples={valid_samples}"
        )
        print(
            "[monitor] MIN_LEFT_EE_BASE_Z / MIN_RIGHT_EE_BASE_Z 应使用左右手动"
            "放到安全最低位置时的各自读数，并结合工具长度保留余量；不要直接"
            "使用运动过程中偶然出现的最低值。"
        )
    else:
        print("[monitor] 没有获得有效样本，请检查 DDS 和 motion status。")


if __name__ == "__main__":
    main()
