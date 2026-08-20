#!/usr/bin/env bash
# 启动命令：
#   bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_freeze_right.sh
#
# 5659 普通同步 EE 的“冻结右臂、只执行左臂”部署入口，不使用 RTC。
# 本文件始终在 Python 参数末尾追加 --freeze-arm=right，调用者不能误把右臂解冻。
# 冻结同时发生在 policy action 与 RobotEnv 最终 ABS_POSE 下发层；右臂
# 位姿和夹爪保持 episode reset 后的锚点。若锚点 z 低于右侧保护值，安全
# 高度优先，右臂 z 会固定在保护值；左臂正常执行模型动作。
#
# 同步模式每次推理选取 EXECUTION_HORIZON 个模型 row，再按位置/旋转上限
# 重采样为可变长 command 队列。它没有 RTC prefix/refill，也不会在执行旧
# chunk 时并行请求新 chunk；每个同步队列执行完后才发起下一次推理。
#
# 下列部署参数均可在单次命令前覆盖，且只进入子进程、不污染父终端：
#
#   MAX_TIMESTEPS              最大实际执行 command 数，默认 4000
#   EXECUTION_HORIZON          每次选择的模型 row 数，默认 50
#   MAX_POSITION_DELTA         每 tick 单手位置变化上限，默认 0.002 m
#   MAX_ROTATION_DELTA         每 tick 单手 SO(3) 旋转变化上限，默认 0.01 rad
#   MAX_RETIMED_CHUNK_STEPS    单个 chunk 最多展开的 command 数，默认 500
#   MIN_LEFT_EE_BASE_Z         左末端 base_link z 下限，默认 0.585 m
#   MIN_RIGHT_EE_BASE_Z        右末端 base_link z 下限，默认 0.575 m
#   CONTROL_FREQUENCY          控制频率，默认 30 Hz
#   TRAJECTORY_TIME            SDK 单点轨迹时间，默认 0.03 s
#   ROLLOUT_FPS                rollout 采集频率，默认 30 Hz
#
# 4000 是长任务部署上限，不代表首次测试应直接运行 4000 步。首次建议：
#
#   MAX_TIMESTEPS=100 \
#     bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_freeze_right.sh
#
# 显式覆盖主要运动和高度参数：
#
#   MAX_TIMESTEPS=4000 EXECUTION_HORIZON=50 \
#   MAX_POSITION_DELTA=0.002 MAX_ROTATION_DELTA=0.01 \
#   MIN_LEFT_EE_BASE_Z=0.585 MIN_RIGHT_EE_BASE_Z=0.575 \
#     bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_freeze_right.sh
#
# rollout 默认开启；设置 ROLLOUT_ENABLED=false 可只执行、不保存视频/Parquet。
# ACTION_LOGGING、PLOT_ACTIONS、ROLLOUT_OUTPUT_DIR、PROMPT/KEY 和 WS_* 继续由
# 下层通用 launcher 读取。
# set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for arg in "$@"; do
  case "${arg}" in
    --freeze-arm|--freeze-arm=*|--freeze-left-arm|--no-freeze-left-arm)
      echo "This launcher always freezes the right arm; do not pass ${arg}" >&2
      exit 2
      ;;
  esac
done

ACTION_HORIZON="${ACTION_HORIZON:-50}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-50}"
CHUNK_SKIP="${CHUNK_SKIP:-0}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30}"

MAX_TIMESTEPS="${MAX_TIMESTEPS:-4000}"
MAX_POSITION_DELTA="${MAX_POSITION_DELTA:-0.002}"
MAX_ROTATION_DELTA="${MAX_ROTATION_DELTA:-0.01}"
TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.03}"

RETIME_ACTION_CHUNK="${RETIME_ACTION_CHUNK:-true}"
MAX_RETIMED_CHUNK_STEPS="${MAX_RETIMED_CHUNK_STEPS:-500}"

MIN_LEFT_EE_BASE_Z="${MIN_LEFT_EE_BASE_Z:-0.585}"
MIN_RIGHT_EE_BASE_Z="${MIN_RIGHT_EE_BASE_Z:-0.575}"

ROLLOUT_FPS="${ROLLOUT_FPS:-30}"
ROLLOUT_ENABLED="${ROLLOUT_ENABLED:-true}"

GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.90}"
GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.50}"

exec env \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  EXECUTION_HORIZON="${EXECUTION_HORIZON}" \
  CHUNK_SKIP="${CHUNK_SKIP}" \
  CONTROL_FREQUENCY="${CONTROL_FREQUENCY}" \
  MAX_TIMESTEPS="${MAX_TIMESTEPS}" \
  MAX_POSITION_DELTA="${MAX_POSITION_DELTA}" \
  MAX_ROTATION_DELTA="${MAX_ROTATION_DELTA}" \
  TRAJECTORY_TIME="${TRAJECTORY_TIME}" \
  RETIME_ACTION_CHUNK="${RETIME_ACTION_CHUNK}" \
  MAX_RETIMED_CHUNK_STEPS="${MAX_RETIMED_CHUNK_STEPS}" \
  MIN_LEFT_EE_BASE_Z="${MIN_LEFT_EE_BASE_Z}" \
  MIN_RIGHT_EE_BASE_Z="${MIN_RIGHT_EE_BASE_Z}" \
  ROLLOUT_FPS="${ROLLOUT_FPS}" \
  ROLLOUT_ENABLED="${ROLLOUT_ENABLED}" \
  GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD}" \
  GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD}" \
  bash "${SCRIPT_DIR}/run_inference_pi05_g1_desk_ee_5659_camera_h50.sh" \
  "$@" \
  --freeze-arm=right
