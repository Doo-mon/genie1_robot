#!/usr/bin/env bash
# 启动命令：
#   bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_freeze_right_rollout_rtc.sh
#
# 5659 EE RTC + rollout 的“冻结右臂、只执行左臂”部署入口。
# 本文件始终在 Python 参数末尾追加 --freeze-arm=right，调用者不能误把右臂解冻。
# 冻结同时发生在 RTC policy timeline 与 RobotEnv 最终 ABS_POSE 下发层；右臂
# 位姿和夹爪保持 episode reset 后的锚点。若锚点 z 低于右侧保护值，安全
# 高度优先，右臂 z 固定在保护值；左臂继续走 RTC 队列。
#
# 下列部署参数均可在单次命令前覆盖，且只进入子进程、不污染父终端：
#
#   MAX_TIMESTEPS              最大实际执行 command 数，默认 4000
#   MAX_POSITION_DELTA         每 tick 单手位置变化上限，默认 0.002 m
#   MAX_ROTATION_DELTA         每 tick 单手 SO(3) 旋转变化上限，默认 0.01 rad
#   MIN_LEFT_EE_BASE_Z         左末端 base_link z 下限，默认 0.585 m
#   MIN_RIGHT_EE_BASE_Z        右末端 base_link z 下限，默认 0.575 m
#   RTC_PREFIX_LEN             RTC 硬锁 prefix，轻量交接默认 24
#   RTC_REFILL_AT              剩余队列补充点，轻量交接默认 26，必须小于 H50
#   CONTROL_FREQUENCY          控制频率，默认 30 Hz
#   TRAJECTORY_TIME            SDK 单点轨迹时间，默认 0.03 s
#   ROLLOUT_FPS                rollout 采集频率，默认 30 Hz
#
# 4000 是长任务部署上限。首次真机建议只执行 100 条命令：
#
#   MAX_TIMESTEPS=100 \
#     bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_freeze_right_rollout_rtc.sh
#
# 显式覆盖主要运动、RTC 和高度参数：
#
#   MAX_TIMESTEPS=4000 RTC_PREFIX_LEN=24 RTC_REFILL_AT=26 \
#   MAX_POSITION_DELTA=0.002 MAX_ROTATION_DELTA=0.01 \
#   MIN_LEFT_EE_BASE_Z=0.585 MIN_RIGHT_EE_BASE_Z=0.575 \
#     bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_freeze_right_rollout_rtc.sh
#
# 夹爪到位同步参数也可按环境变量覆盖。默认启用：左臂物理反馈未到达夹爪
# 切换帧时保持旧夹爪，稳定到位后才夹紧/松开，并重新规划剩余动作。
# 设置 ROLLOUT_ENABLED=false 可关闭视频/Parquet，但仍使用 RTC。
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
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30}"
# 轻量交接 RTC：H50 先正常执行24行，剩26行时才发起下一次推理，并把
# 其中24行作为 committed prefix。理想情况下当前 chunk 的前48/50行保持
# 第一次观测生成的轨迹，只有最后约2行用于衔接下一块。24步在30 Hz下还可
# 容纳约0.8秒的推理交接；更慢的响应宁可被拒绝，也不大幅改写当前 chunk。
RTC_PREFIX_LEN="${RTC_PREFIX_LEN:-24}"
RTC_REFILL_AT="${RTC_REFILL_AT:-26}"

MAX_TIMESTEPS="${MAX_TIMESTEPS:-4000}"
MAX_POSITION_DELTA="${MAX_POSITION_DELTA:-0.002}"
MAX_ROTATION_DELTA="${MAX_ROTATION_DELTA:-0.01}"
TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.03}"

MIN_LEFT_EE_BASE_Z="${MIN_LEFT_EE_BASE_Z:-0.585}"
MIN_RIGHT_EE_BASE_Z="${MIN_RIGHT_EE_BASE_Z:-0.575}"

ROLLOUT_FPS="${ROLLOUT_FPS:-30}"
ROLLOUT_ENABLED="${ROLLOUT_ENABLED:-true}"

GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.90}"
GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.50}"
SYNC_GRIPPER_TO_ARM="${SYNC_GRIPPER_TO_ARM:-true}"
GRIPPER_SYNC_TIMEOUT="${GRIPPER_SYNC_TIMEOUT:-5.0}"
GRIPPER_SYNC_MAX_TIMEOUT="${GRIPPER_SYNC_MAX_TIMEOUT:-30.0}"
GRIPPER_SYNC_POSITION_TOLERANCE="${GRIPPER_SYNC_POSITION_TOLERANCE:-0.005}"
GRIPPER_SYNC_ROTATION_TOLERANCE="${GRIPPER_SYNC_ROTATION_TOLERANCE:-0.05}"
GRIPPER_SYNC_STABLE_FRAMES="${GRIPPER_SYNC_STABLE_FRAMES:-3}"
GRIPPER_SETTLE_TIME="${GRIPPER_SETTLE_TIME:-0.20}"
REPLAN_AFTER_GRIPPER_SYNC="${REPLAN_AFTER_GRIPPER_SYNC:-true}"

EXTRA_ARGS=(
  --freeze-arm=right
  --gripper-sync-timeout="${GRIPPER_SYNC_TIMEOUT}"
  --gripper-sync-max-timeout="${GRIPPER_SYNC_MAX_TIMEOUT}"
  --gripper-sync-position-tolerance="${GRIPPER_SYNC_POSITION_TOLERANCE}"
  --gripper-sync-rotation-tolerance="${GRIPPER_SYNC_ROTATION_TOLERANCE}"
  --gripper-sync-stable-frames="${GRIPPER_SYNC_STABLE_FRAMES}"
  --gripper-settle-time="${GRIPPER_SETTLE_TIME}"
)

case "${ROLLOUT_ENABLED}" in
  1|true|TRUE|yes|YES) EXTRA_ARGS+=(--rollout-enabled) ;;
  0|false|FALSE|no|NO) EXTRA_ARGS+=(--no-rollout-enabled) ;;
  *) echo "ROLLOUT_ENABLED must be true or false" >&2; exit 2 ;;
esac

case "${SYNC_GRIPPER_TO_ARM}" in
  1|true|TRUE|yes|YES) EXTRA_ARGS+=(--sync-gripper-to-arm) ;;
  0|false|FALSE|no|NO) EXTRA_ARGS+=(--no-sync-gripper-to-arm) ;;
  *) echo "SYNC_GRIPPER_TO_ARM must be true or false" >&2; exit 2 ;;
esac

case "${REPLAN_AFTER_GRIPPER_SYNC}" in
  1|true|TRUE|yes|YES) EXTRA_ARGS+=(--replan-after-gripper-sync) ;;
  0|false|FALSE|no|NO) EXTRA_ARGS+=(--no-replan-after-gripper-sync) ;;
  *) echo "REPLAN_AFTER_GRIPPER_SYNC must be true or false" >&2; exit 2 ;;
esac

exec env \
  ACTION_HORIZON="${ACTION_HORIZON}" \
  CONTROL_FREQUENCY="${CONTROL_FREQUENCY}" \
  RTC_PREFIX_LEN="${RTC_PREFIX_LEN}" \
  RTC_REFILL_AT="${RTC_REFILL_AT}" \
  MAX_TIMESTEPS="${MAX_TIMESTEPS}" \
  MAX_POSITION_DELTA="${MAX_POSITION_DELTA}" \
  MAX_ROTATION_DELTA="${MAX_ROTATION_DELTA}" \
  TRAJECTORY_TIME="${TRAJECTORY_TIME}" \
  MIN_LEFT_EE_BASE_Z="${MIN_LEFT_EE_BASE_Z}" \
  MIN_RIGHT_EE_BASE_Z="${MIN_RIGHT_EE_BASE_Z}" \
  ROLLOUT_FPS="${ROLLOUT_FPS}" \
  GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD}" \
  GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD}" \
  bash "${SCRIPT_DIR}/run_inference_pi05_g1_desk_ee_5659_camera_h50_rollout_rtc.sh" \
  "$@" \
  "${EXTRA_ARGS[@]}"
