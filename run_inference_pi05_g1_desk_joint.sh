#!/usr/bin/env bash
# 通用同步 joint 推理入口，直接调用 inference_pi05_g1_desk_joint.py。
# 服务地址、任务名、action/execution horizon、控制频率、关节限幅和夹爪阈值
# 均由同名环境变量提供默认覆盖。WS_* 控制慢启动等待、握手/响应超时和退避；
# 已发送请求默认不重发，只有 WS_INFERENCE_RETRIES>0 才断线重连重发。
# PROMPT（兼容 FIXED_PROMPT）优先于 PROMPT_KEY；都未设置时由 Python 交互选择，
# 每个 prompt 都必须明确确认后才允许机器人 reset 和执行。
# 平滑轨迹、假观测和 debug 也由 0/1 环境变量转换为 CLI 开关；这些设置只作用于
# 随后运行的 Python 子进程，不会写回父终端环境。
# 每次运行会在 ACTION_LOG_ROOT（默认本目录/logs）建立时间戳目录并保存 console.log；
# PLOT_ACTIONS 默认 true，退出后绘制动作对比图且不覆盖 Python 原退出码。
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
EXECUTION_HORIZON="${EXECUTION_HORIZON:-1}"
CHUNK_SKIP="${CHUNK_SKIP:-0}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-100}"
MAX_JOINT_DELTA="${MAX_JOINT_DELTA:-0.15}"
# 30 Hz 的控制周期约为 0.033 s，轨迹时间略短于一个周期。
TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.03}"
GRIPPER_THRESHOLD="${GRIPPER_THRESHOLD:-0.6}"
PROMPT="${PROMPT:-${FIXED_PROMPT:-}}"
PROMPT_KEY="${PROMPT_KEY:-}"
USE_SMOOTH_TRAJECTORY="${USE_SMOOTH_TRAJECTORY:-0}"
USE_FAKE_OBS="${USE_FAKE_OBS:-0}"
DEBUG="${DEBUG:-0}"

EXTRA_ARGS=()
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
  "inference_pi05_g1_desk_joint" \
  "${PYTHON_BIN}" || exit $?

inference_log_run "${PYTHON_BIN}" "${SCRIPT_DIR}/inference_pi05_g1_desk_joint.py" \
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
  --execution-horizon="${EXECUTION_HORIZON}" \
  --chunk-skip="${CHUNK_SKIP}" \
  --control-frequency="${CONTROL_FREQUENCY}" \
  --max-timesteps="${MAX_TIMESTEPS}" \
  --max-joint-delta="${MAX_JOINT_DELTA}" \
  --trajectory-time="${TRAJECTORY_TIME}" \
  --gripper-threshold="${GRIPPER_THRESHOLD}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
run_status=$?
exit "${run_status}"
