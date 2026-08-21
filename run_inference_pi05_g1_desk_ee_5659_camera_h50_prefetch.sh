#!/usr/bin/env bash
# 启动命令：
#   bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_prefetch.sh
#
# 5659 双臂 H50 的同步重采样 + 异步预取 preset，不使用 RTC 客户端时间线。
# 当前 H50 重采样队列完整执行；剩余24条实际 command 时后台请求未来尾部。
# 这主要用于对照普通同步5659，减少chunk边界等待而不改变当前chunk动作。
#
# 默认30 Hz、0.002 m/tick、0.01 rad/tick、trajectory_time=0.03 s，rollout
# 开启且30 Hz。PREFETCH_PREFIX_LEN只控制何时发网络请求，不改变动作幅度。
# 服务端必须启用 RTC prefix guidance。首次真机建议：
#
#   MAX_TIMESTEPS=100 ROLLOUT_ENABLED=false \
#     bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_prefetch.sh
#
# ACTION_LOGGING默认关闭；只在定位问题时单次开启，完成后及时关闭。
# set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec env \
  TASK_NAME="${TASK_NAME:-5659}" \
  ACTION_HORIZON="${ACTION_HORIZON:-50}" \
  EXECUTION_HORIZON="${EXECUTION_HORIZON:-50}" \
  CHUNK_SKIP="${CHUNK_SKIP:-0}" \
  CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30}" \
  TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.03}" \
  MAX_TIMESTEPS="${MAX_TIMESTEPS:-1500}" \
  MAX_POSITION_DELTA="${MAX_POSITION_DELTA:-0.002}" \
  MAX_ROTATION_DELTA="${MAX_ROTATION_DELTA:-0.01}" \
  MAX_RETIMED_CHUNK_STEPS="${MAX_RETIMED_CHUNK_STEPS:-500}" \
  PREFETCH_PREFIX_LEN="${PREFETCH_PREFIX_LEN:-24}" \
  PREFETCH_WAIT_POLL="${PREFETCH_WAIT_POLL:-0.10}" \
  BINARIZE_GRIPPER="${BINARIZE_GRIPPER:-true}" \
  GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.90}" \
  GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.50}" \
  ROLLOUT_ENABLED="${ROLLOUT_ENABLED:-true}" \
  ROLLOUT_FPS="${ROLLOUT_FPS:-30}" \
  PROMPT_KEY="${PROMPT_KEY:-named:5659-all}" \
  bash "${SCRIPT_DIR}/run_inference_pi05_g1_desk_ee_prefetch.sh" "$@"
