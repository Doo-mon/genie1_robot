#!/usr/bin/env bash
# 启动命令：
#   bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_freeze_right_prefetch.sh
#
# 5659 的独立同步预取实验入口：冻结右臂，只执行左臂；不使用 RTC。当前
# H50 先完整重采样并执行，剩余24条 command 时后台预取下一段。预取只减少
# chunk 间网络等待，不允许新响应改变当前队列，因此便于和准确的同步 H50
# 基线直接比较。
#
# 默认运动上限沿用当前平滑同步版本：30 Hz、0.002 m/tick、0.01 rad/tick、
# SDK trajectory_time=0.03 s。左右 base_link z 最低值默认为0.585/0.575 m。
# 右臂在 policy EE7 与最终 SDK ABS_POSE 两层冻结；高度保护优先于冻结锚点。
#
# 首次真机建议限制实际 command 数：
#
#   MAX_TIMESTEPS=100 ROLLOUT_ENABLED=false \
#     bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_freeze_right_prefetch.sh
#
# 若仍出现边界等待，可根据日志中的 infer_ms 微调 PREFETCH_PREFIX_LEN，但必须
# 小于50。增大它会更早预取，同时减少每次响应可提供的未来 row；不应把它当作
# 动作幅度参数。动作日志默认关闭，需要诊断时单次设置 ACTION_LOGGING=true。
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

exec env \
  TASK_NAME="${TASK_NAME:-5659}" \
  ACTION_HORIZON="${ACTION_HORIZON:-50}" \
  EXECUTION_HORIZON="${EXECUTION_HORIZON:-50}" \
  CHUNK_SKIP="${CHUNK_SKIP:-0}" \
  CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30}" \
  TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.03}" \
  MAX_TIMESTEPS="${MAX_TIMESTEPS:-4000}" \
  MAX_POSITION_DELTA="${MAX_POSITION_DELTA:-0.002}" \
  MAX_ROTATION_DELTA="${MAX_ROTATION_DELTA:-0.01}" \
  MAX_RETIMED_CHUNK_STEPS="${MAX_RETIMED_CHUNK_STEPS:-500}" \
  PREFETCH_PREFIX_LEN="${PREFETCH_PREFIX_LEN:-24}" \
  PREFETCH_WAIT_POLL="${PREFETCH_WAIT_POLL:-0.10}" \
  MIN_LEFT_EE_BASE_Z="${MIN_LEFT_EE_BASE_Z:-0.585}" \
  MIN_RIGHT_EE_BASE_Z="${MIN_RIGHT_EE_BASE_Z:-0.575}" \
  BINARIZE_GRIPPER="${BINARIZE_GRIPPER:-true}" \
  GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.90}" \
  GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.50}" \
  ROLLOUT_ENABLED="${ROLLOUT_ENABLED:-true}" \
  ROLLOUT_FPS="${ROLLOUT_FPS:-30}" \
  PROMPT_KEY="${PROMPT_KEY:-named:5659-all}" \
  bash "${SCRIPT_DIR}/run_inference_pi05_g1_desk_ee_5659_camera_h50_prefetch.sh" \
  "$@" \
  --freeze-arm=right
