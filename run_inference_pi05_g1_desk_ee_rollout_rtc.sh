#!/usr/bin/env bash
# 启动命令：bash ./run_inference_pi05_g1_desk_ee_rollout_rtc.sh
#
# 通用异步 EE RTC + rollout 入口，直接调用 inference_pi05_g1_desk_ee_rollout_rtc.py。
# 网络、任务、RTC、可选二值化、控制安全参数及 rollout 配置来自同名环境变量；
# prefix 夹爪范围固定为 0--1，不提供 Shell 切换参数。
# 夹爪约定 0=张开、1=闭合，默认启用 close=0.90/open=0.50 双阈值滞回；
# rollout 保存经过该处理后实际发送的命令，旧单阈值只作为显式兼容覆盖。
# MIN_LEFT_EE_BASE_Z / MIN_RIGHT_EE_BASE_Z 分别设置左右最低 z（米）；
# MIN_EE_BASE_Z 是兼容共同值，单臂值优先，越界时钳位后继续执行。
# WS_* 控制慢启动等待、握手/响应超时和退避；默认不重发已经发送的请求，
# 仅 WS_INFERENCE_RETRIES>0 才重连重发。"$@" 可继续覆盖 Python 参数。
# PROMPT（兼容 FIXED_PROMPT）优先于 PROMPT_KEY；都为空时显示交互菜单，
# 选定的 prompt 必须确认后才会 reset、warmup、执行或开始录制。
# 本文件只把配置传给随后运行的 Python 子进程，正常执行不会污染父终端变量。
# 每次运行会在 ACTION_LOG_ROOT（默认本目录/logs）建立时间戳目录并保存 console.log；
# 动作 JSONL/真实 feedback 默认关闭；ACTION_LOGGING=true 时才记录，且
# PLOT_ACTIONS 默认 true，会在退出后绘图且不覆盖 Python 原退出码。
# set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/inference_log_helpers.sh" || exit $?
REAL_ROBOT_ROOT="${REAL_ROBOT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export PYTHONPATH="${REAL_ROBOT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
REMOTE_HOST="${REMOTE_HOST:-127.0.0.1}"
REMOTE_PORT="${REMOTE_PORT:-2379}"
WS_OPEN_TIMEOUT="${WS_OPEN_TIMEOUT:-10}"
WS_CLOSE_TIMEOUT="${WS_CLOSE_TIMEOUT:-10}"
WS_RESPONSE_TIMEOUT="${WS_RESPONSE_TIMEOUT:-300}"
WS_RETRY_INITIAL_DELAY="${WS_RETRY_INITIAL_DELAY:-1}"
WS_RETRY_MAX_DELAY="${WS_RETRY_MAX_DELAY:-10}"
WS_SERVER_WAIT_TIMEOUT="${WS_SERVER_WAIT_TIMEOUT:-0}"
WS_RECONNECT_WAIT_TIMEOUT="${WS_RECONNECT_WAIT_TIMEOUT:-30}"
WS_INFERENCE_RETRIES="${WS_INFERENCE_RETRIES:-0}"
TASK_NAME="${TASK_NAME:-perdon}"
ACTION_HORIZON="${ACTION_HORIZON:-20}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-100}"
RTC_PREFIX_LEN="${RTC_PREFIX_LEN:-8}"
RTC_REFILL_AT="${RTC_REFILL_AT:-12}"
BINARIZE_GRIPPER="${BINARIZE_GRIPPER:-true}"
MAX_POSITION_DELTA="${MAX_POSITION_DELTA:-0.0005}"
MAX_ROTATION_DELTA="${MAX_ROTATION_DELTA:-0.0025}"
MIN_EE_BASE_Z="${MIN_EE_BASE_Z:-}"
MIN_LEFT_EE_BASE_Z="${MIN_LEFT_EE_BASE_Z:-}"
MIN_RIGHT_EE_BASE_Z="${MIN_RIGHT_EE_BASE_Z:-}"
# 30 Hz 的控制周期约为 0.033 s；默认轨迹时间略短于一个周期，避免
# 每条单点轨迹执行到一半就被下一条命令打断。
TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.03}"
GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.90}"
GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.50}"
ROLLOUT_FPS="${ROLLOUT_FPS:-30}"
ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT_DIR:-}"
PROMPT="${PROMPT:-${FIXED_PROMPT:-}}"
PROMPT_KEY="${PROMPT_KEY:-}"

EXTRA_ARGS=()
case "${BINARIZE_GRIPPER}" in
  1|true|TRUE|yes|YES) EXTRA_ARGS+=(--binarize-gripper) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "BINARIZE_GRIPPER must be true or false" >&2; exit 2 ;;
esac
if [[ -n "${GRIPPER_THRESHOLD:-}" ]]; then
  # Legacy compatibility: a single threshold intentionally disables hysteresis.
  EXTRA_ARGS+=(--gripper-threshold="${GRIPPER_THRESHOLD}")
fi
if [[ -n "${MIN_EE_BASE_Z}" ]]; then
  EXTRA_ARGS+=(--min-ee-base-z="${MIN_EE_BASE_Z}")
fi
if [[ -n "${MIN_LEFT_EE_BASE_Z}" ]]; then
  EXTRA_ARGS+=(--min-left-ee-base-z="${MIN_LEFT_EE_BASE_Z}")
fi
if [[ -n "${MIN_RIGHT_EE_BASE_Z}" ]]; then
  EXTRA_ARGS+=(--min-right-ee-base-z="${MIN_RIGHT_EE_BASE_Z}")
fi
if [[ -n "${ROLLOUT_OUTPUT_DIR}" ]]; then
  EXTRA_ARGS+=(--rollout-output-dir="${ROLLOUT_OUTPUT_DIR}")
fi
if [[ -n "${PROMPT}" ]]; then
  EXTRA_ARGS+=(--prompt="${PROMPT}")
elif [[ -n "${PROMPT_KEY}" ]]; then
  EXTRA_ARGS+=(--prompt-key="${PROMPT_KEY}")
fi

inference_log_init \
  "${SCRIPT_DIR}" \
  "inference_pi05_g1_desk_ee_rollout_rtc" \
  "${PYTHON_BIN}" || exit $?

inference_log_run "${PYTHON_BIN}" "${SCRIPT_DIR}/inference_pi05_g1_desk_ee_rollout_rtc.py" \
  --remote-host="${REMOTE_HOST}" \
  --remote-port="${REMOTE_PORT}" \
  --ws-open-timeout="${WS_OPEN_TIMEOUT}" \
  --ws-close-timeout="${WS_CLOSE_TIMEOUT}" \
  --ws-response-timeout="${WS_RESPONSE_TIMEOUT}" \
  --ws-retry-initial-delay="${WS_RETRY_INITIAL_DELAY}" \
  --ws-retry-max-delay="${WS_RETRY_MAX_DELAY}" \
  --ws-server-wait-timeout="${WS_SERVER_WAIT_TIMEOUT}" \
  --ws-reconnect-wait-timeout="${WS_RECONNECT_WAIT_TIMEOUT}" \
  --ws-inference-retries="${WS_INFERENCE_RETRIES}" \
  --task-name="${TASK_NAME}" \
  --action-horizon="${ACTION_HORIZON}" \
  --control-frequency="${CONTROL_FREQUENCY}" \
  --max-timesteps="${MAX_TIMESTEPS}" \
  --chunk-skip=0 \
  --rtc-prefix-len="${RTC_PREFIX_LEN}" \
  --rtc-refill-at="${RTC_REFILL_AT}" \
  --max-position-delta="${MAX_POSITION_DELTA}" \
  --max-rotation-delta="${MAX_ROTATION_DELTA}" \
  --trajectory-time="${TRAJECTORY_TIME}" \
  --gripper-close-threshold="${GRIPPER_CLOSE_THRESHOLD}" \
  --gripper-open-threshold="${GRIPPER_OPEN_THRESHOLD}" \
  --rollout-fps="${ROLLOUT_FPS}" \
  --debug \
  "${EXTRA_ARGS[@]}" \
  "$@"
run_status=$?
exit "${run_status}"
