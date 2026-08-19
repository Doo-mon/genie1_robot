#!/usr/bin/env bash
# 启动命令：bash ./run_inference_pi05_g1_desk_ee_3017_3040_h50_rollout_rtc.sh
#
# 3017/3040 任务 preset，不直接调用 Python，而是转交给
# run_inference_pi05_g1_desk_ee_rollout_rtc.sh，最终运行通用 EE RTC rollout 客户端。
# action horizon、RTC prefix/refill 和默认 PROMPT_KEY 由调用方同名变量优先，
# 未设置时采用 50/34/40、breads 初始姿态与 named:bread-all；"$@" 原样下传。
# 34 步 prefix 在 30 Hz 下覆盖约 1.13 秒，40 步 refill 为慢推理预留队列余量。
# 如需使用另一套面包机位，可显式传 TASK_NAME=breads_ours 覆盖。
# 夹爪默认启用 0.90/0.50 双阈值滞回（0=张开、1=闭合），同名环境变量可覆盖。
# REMOTE_HOST、REMOTE_PORT 和全部 WS_* 网络参数不在这里重设，会保留调用者值；
# 同样支持慢启动等待，但默认不重发，只有显式配置正数重试次数才断线重连。
# 若父环境提供 PROMPT，通用入口仍按 PROMPT 优先于 PROMPT_KEY 处理；最终 prompt
# 在机器人动作与录制前仍需确认。
# 这里使用 exec env 设置命令级变量，它们只对下层 shell/Python 子进程生效，
# 不会 export 或污染调用本 preset 的父终端环境。
# ACTION_LOG_ROOT/PLOT_ACTIONS 也沿 exec-env 链继承，由下层通用 launcher
# 只创建一个运行目录并统一完成 console 捕获与退出绘图。
# set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# These assignments are scoped to this one launch and do not modify the parent shell.
exec env \
  ACTION_HORIZON="${ACTION_HORIZON:-50}" \
  TASK_NAME="${TASK_NAME:-breads}" \
  RTC_PREFIX_LEN="${RTC_PREFIX_LEN:-34}" \
  RTC_REFILL_AT="${RTC_REFILL_AT:-40}" \
  BINARIZE_GRIPPER="${BINARIZE_GRIPPER:-true}" \
  GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.90}" \
  GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.50}" \
  PROMPT_KEY="${PROMPT_KEY:-named:bread-all}" \
  bash "${SCRIPT_DIR}/run_inference_pi05_g1_desk_ee_rollout_rtc.sh" "$@"
