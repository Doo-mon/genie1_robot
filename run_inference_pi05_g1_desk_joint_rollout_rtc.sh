#!/usr/bin/env bash
# 启动命令：bash ./run_inference_pi05_g1_desk_joint_rollout_rtc.sh
#
# 通用异步 joint RTC + rollout 入口，直接调用 inference_pi05_g1_desk_joint_rollout_rtc.py。
# 网络、任务、RTC、关节安全参数和 rollout 配置来自同名环境变量。WS_* 控制
# 慢启动等待、握手/响应超时和退避；已发送请求默认不重发，只有
# WS_INFERENCE_RETRIES>0 才重连重发。ROLLOUT_ENABLED=0 会关闭录制。
# PROMPT（兼容 FIXED_PROMPT）优先于 PROMPT_KEY；都为空时由 Python 交互选择，
# 所有 prompt 均在 reset、warmup、RTC 执行和录制开始前要求确认。
# 本 launcher 中的开关和变量只传给随后运行的 Python 子进程，
# 不会写回或污染调用它的父终端环境。
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
TASK_NAME="${TASK_NAME:-beijing_benchmark_setting}"
ACTION_HORIZON="${ACTION_HORIZON:-30}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-100}"
RTC_PREFIX_LEN="${RTC_PREFIX_LEN:-8}"
RTC_REFILL_AT="${RTC_REFILL_AT:-12}"
RTC_PREFIX_SPACE="${RTC_PREFIX_SPACE:-openpi}"
MAX_JOINT_DELTA="${MAX_JOINT_DELTA:-0.15}"
# 30 Hz 的控制周期约为 0.033 s，轨迹时间略短于一个周期。
TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.03}"
GRIPPER_THRESHOLD="${GRIPPER_THRESHOLD:-0.6}"
ROLLOUT_FPS="${ROLLOUT_FPS:-30}"
ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT_DIR:-}"
ROLLOUT_ENABLED="${ROLLOUT_ENABLED:-1}"
PROMPT="${PROMPT:-${FIXED_PROMPT:-}}"
PROMPT_KEY="${PROMPT_KEY:-}"
USE_SMOOTH_TRAJECTORY="${USE_SMOOTH_TRAJECTORY:-0}"
USE_FAKE_OBS="${USE_FAKE_OBS:-0}"
DEBUG="${DEBUG:-1}"

EXTRA_ARGS=()
if [[ -n "${ROLLOUT_OUTPUT_DIR}" ]]; then
  EXTRA_ARGS+=(--rollout-output-dir="${ROLLOUT_OUTPUT_DIR}")
fi
if [[ "${ROLLOUT_ENABLED}" == "0" ]]; then
  EXTRA_ARGS+=(--no-rollout-enabled)
fi
if [[ -n "${PROMPT}" ]]; then
  EXTRA_ARGS+=(--prompt="${PROMPT}")
elif [[ -n "${PROMPT_KEY}" ]]; then
  EXTRA_ARGS+=(--prompt-key="${PROMPT_KEY}")
fi
if [[ "${USE_SMOOTH_TRAJECTORY}" == "1" ]]; then
  EXTRA_ARGS+=(--use-smooth-trajectory)
fi
if [[ "${USE_FAKE_OBS}" == "1" ]]; then
  EXTRA_ARGS+=(--use-fake-obs)
fi
if [[ "${DEBUG}" == "1" ]]; then
  EXTRA_ARGS+=(--debug)
fi

inference_log_init \
  "${SCRIPT_DIR}" \
  "inference_pi05_g1_desk_joint_rollout_rtc" \
  "${PYTHON_BIN}" || exit $?

inference_log_run "${PYTHON_BIN}" "${SCRIPT_DIR}/inference_pi05_g1_desk_joint_rollout_rtc.py" \
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
  --rtc-prefix-space="${RTC_PREFIX_SPACE}" \
  --max-joint-delta="${MAX_JOINT_DELTA}" \
  --trajectory-time="${TRAJECTORY_TIME}" \
  --gripper-threshold="${GRIPPER_THRESHOLD}" \
  --rollout-fps="${ROLLOUT_FPS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
run_status=$?
exit "${run_status}"
