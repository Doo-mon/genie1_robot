#!/usr/bin/env bash
# 启动命令：
#   bash ./run_inference_pi05_g1_desk_joint_prefetch.sh
#
# joint16 的独立“同步重采样队列 + 异步预取”入口，不使用原 joint RTC。
# 当前H30完整按逐关节 MAX_JOINT_DELTA 展开并执行；剩余 command 不超过
# PREFETCH_PREFIX_LEN 时，以统一的 [L7,Lhand,R7,Rhand] 顺序后台请求下一段。
# 当前队列不会被响应替换，响应前缀被丢弃，只在边界接入未来尾部。
#
# 默认 P=18 在30 Hz下给后台约0.6秒，并留下12个未来模型 row。若模型更慢，
# 边界仍会等待；P必须小于 ACTION_HORIZON。服务端必须通过 RTC_ENABLE=true
# 或等价 ``--rtc`` 启用 prefix guidance，当前 metadata 无法让客户端自动验证。
#
# 本入口明确禁止 USE_SMOOTH_TRAJECTORY：该SDK模式会在一个tick内执行完整
# Ruckig轨迹，破坏固定频率和prefix时间线。rollout、动作日志和绘图分别由
# ROLLOUT_ENABLED、ACTION_LOGGING、PLOT_ACTIONS 控制，默认均不增加动作日志开销。
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
EXECUTION_HORIZON="${EXECUTION_HORIZON:-30}"
CHUNK_SKIP="${CHUNK_SKIP:-0}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30}"
TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.03}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-100}"
MAX_JOINT_DELTA="${MAX_JOINT_DELTA:-0.15}"
MAX_RETIMED_CHUNK_STEPS="${MAX_RETIMED_CHUNK_STEPS:-500}"
PREFETCH_PREFIX_LEN="${PREFETCH_PREFIX_LEN:-18}"
PREFETCH_WAIT_POLL="${PREFETCH_WAIT_POLL:-0.10}"
GRIPPER_THRESHOLD="${GRIPPER_THRESHOLD:-0.6}"

PROMPT="${PROMPT:-${FIXED_PROMPT:-}}"
PROMPT_KEY="${PROMPT_KEY:-}"
USE_FAKE_OBS="${USE_FAKE_OBS:-0}"
DEBUG="${DEBUG:-0}"
ROLLOUT_ENABLED="${ROLLOUT_ENABLED:-false}"
ROLLOUT_FPS="${ROLLOUT_FPS:-30}"
ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT_DIR:-}"

EXTRA_ARGS=()
if [[ -n "${PROMPT}" ]]; then
  EXTRA_ARGS+=(--prompt="${PROMPT}")
elif [[ -n "${PROMPT_KEY}" ]]; then
  EXTRA_ARGS+=(--prompt-key="${PROMPT_KEY}")
fi
if [[ "${USE_FAKE_OBS}" == "1" ]]; then
  EXTRA_ARGS+=(--use-fake-obs)
fi
if [[ "${DEBUG}" == "1" ]]; then
  EXTRA_ARGS+=(--debug)
fi
case "${ROLLOUT_ENABLED}" in
  1|true|TRUE|yes|YES) EXTRA_ARGS+=(--rollout-enabled) ;;
  0|false|FALSE|no|NO) EXTRA_ARGS+=(--no-rollout-enabled) ;;
  *) echo "ROLLOUT_ENABLED must be true or false" >&2; exit 2 ;;
esac
if [[ -n "${ROLLOUT_OUTPUT_DIR}" ]]; then
  EXTRA_ARGS+=(--rollout-output-dir="${ROLLOUT_OUTPUT_DIR}")
fi

inference_log_init \
  "${SCRIPT_DIR}" \
  "inference_pi05_g1_desk_joint_prefetch" \
  "${PYTHON_BIN}" || exit $?

inference_log_run "${PYTHON_BIN}" "${SCRIPT_DIR}/inference_pi05_g1_desk_joint_prefetch.py" \
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
  --trajectory-time="${TRAJECTORY_TIME}" \
  --max-timesteps="${MAX_TIMESTEPS}" \
  --max-joint-delta="${MAX_JOINT_DELTA}" \
  --max-retimed-chunk-steps="${MAX_RETIMED_CHUNK_STEPS}" \
  --prefetch-prefix-len="${PREFETCH_PREFIX_LEN}" \
  --prefetch-wait-poll="${PREFETCH_WAIT_POLL}" \
  --gripper-threshold="${GRIPPER_THRESHOLD}" \
  --rollout-fps="${ROLLOUT_FPS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
run_status=$?
exit "${run_status}"
