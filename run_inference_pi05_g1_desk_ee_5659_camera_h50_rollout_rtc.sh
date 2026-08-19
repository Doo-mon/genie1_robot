#!/usr/bin/env bash
# 5659 camera 任务 preset，不直接调用 Python，而是转交给
# run_inference_pi05_g1_desk_ee_rollout_rtc.sh，最终运行通用 EE RTC rollout 客户端。
# horizon、任务名、频率、RTC、安全阈值、rollout 帧率和默认 PROMPT_KEY 均允许
# 调用方用同名变量覆盖，否则采用本 preset 的 camera/5659 配置；"$@" 原样下传。
# H50 的 RTC 默认提前在剩余 40 步时补充，并硬锁 34 步；在 30 Hz 下可覆盖
# 约 1.13 秒推理延迟，避免短 prefix 造成响应反复被拒绝和队列停走。
# 30 Hz 控制周期约 0.033 秒，TRAJECTORY_TIME 默认设为 0.03 秒，避免上一条
# SDK 单点轨迹还没完成就被下一条命令替换。
# 夹爪默认启用 0.90/0.10 双阈值滞回（0=张开、1=闭合），避免过早切换。
# RTC 还默认把二值夹爪切换与对应手臂目标同步：手臂受单步限幅尚未到达
# 切换帧位姿时，保持旧夹爪并暂停消费该模型 row；到达后才夹紧或松开。
# 该门控只作用于夹爪切换帧，不会把整个 H50 chunk 做可变长度重采样。
# REMOTE_HOST、REMOTE_PORT 与 WS_* 网络参数会从调用环境原样保留，由通用
# launcher 转成 Python 参数；同样慢启动等待，但已发送请求默认不重发。
# 父环境中的 PROMPT 仍优先于 PROMPT_KEY，默认 key 为 named:5659-all；任何来源的
# prompt 都会在 reset、执行和录制之前要求人工确认。
# 所有 preset 值通过 exec env 仅注入下层 shell/Python 子进程，不使用 export，
# 因而不会污染调用本 preset 的父终端环境。
# ACTION_LOG_ROOT/PLOT_ACTIONS 同样由下层通用 launcher 读取，整个 preset 链
# 只建立一个运行日志目录，并在 Python 退出后统一绘图。
# 如需关闭动作/策略 JSONL 和退出后的曲线绘制，可使用：
#   ACTION_LOGGING=false PLOT_ACTIONS=false bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_rollout_rtc.sh
# 上述方式仍保留较小的 console.log；如需同时关闭 rollout 视频和 Parquet 保存：
#   ACTION_LOGGING=false PLOT_ACTIONS=false bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_rollout_rtc.sh --no-rollout-enabled
# set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# These assignments are scoped to this one launch and do not modify the parent shell.
exec env \
  ACTION_HORIZON="${ACTION_HORIZON:-50}" \
  TASK_NAME="${TASK_NAME:-5659}" \
  CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30}" \
  RTC_PREFIX_LEN="${RTC_PREFIX_LEN:-34}" \
  RTC_REFILL_AT="${RTC_REFILL_AT:-40}" \
  BINARIZE_GRIPPER="${BINARIZE_GRIPPER:-true}" \
  MAX_POSITION_DELTA="${MAX_POSITION_DELTA:-0.002}" \
  MAX_ROTATION_DELTA="${MAX_ROTATION_DELTA:-0.01}" \
  TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.03}" \
  MAX_TIMESTEPS="${MAX_TIMESTEPS:-600}" \
  GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.90}" \
  GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.50}" \
  ROLLOUT_FPS="${ROLLOUT_FPS:-30}" \
  PROMPT_KEY="${PROMPT_KEY:-named:5659-all}" \
  bash "${SCRIPT_DIR}/run_inference_pi05_g1_desk_ee_rollout_rtc.sh" "$@"
