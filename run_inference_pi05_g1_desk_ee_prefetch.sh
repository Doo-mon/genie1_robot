#!/usr/bin/env bash
# 启动命令：
#   bash ./run_inference_pi05_g1_desk_ee_prefetch.sh
#
# 通用“同步 EE 重采样队列 + 异步预取下一 chunk”入口。它不是 RTC：当前
# 重采样队列一定完整执行，后台响应只提供当前队列结束后的未来动作。队列剩余
# command 数不超过 PREFETCH_PREFIX_LEN 时，剩余 command 会以 EE14、夹爪
# 0--1 的 prefix 发给服务端。若响应仍迟到，chunk 边界等待，不替换当前轨迹。
# 服务端必须以 RTC_ENABLE=true 或等价 ``--rtc`` 启用 prefix guidance；当前
# server metadata 不携带该开关，客户端只能在启动横幅中提醒，不能自动确认。
#
# ACTION_HORIZON 必须大于 PREFETCH_PREFIX_LEN。H50 建议先使用 24：30 Hz 下
# 给后台约 0.8 秒推理时间，同时保留响应中的 26 行未来动作。该值太小容易仍在
# 边界等待；太大会减少每次响应真正新增的未来 row，不能无限增大。
#
# rollout、动作 JSONL 和绘图仍分别由 ROLLOUT_ENABLED、ACTION_LOGGING 和
# PLOT_ACTIONS 控制。默认动作日志关闭；诊断结束后应及时关闭，避免反馈读取和
# 逐行写盘占用控制周期。所有变量只进入本次子进程，不修改父终端环境。
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
ACTION_HORIZON="${ACTION_HORIZON:-50}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-50}"
CHUNK_SKIP="${CHUNK_SKIP:-0}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30}"
TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.03}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-1500}"

MAX_POSITION_DELTA="${MAX_POSITION_DELTA:-0.002}"
MAX_ROTATION_DELTA="${MAX_ROTATION_DELTA:-0.01}"
MAX_RETIMED_CHUNK_STEPS="${MAX_RETIMED_CHUNK_STEPS:-500}"
PREFETCH_PREFIX_LEN="${PREFETCH_PREFIX_LEN:-24}"
PREFETCH_WAIT_POLL="${PREFETCH_WAIT_POLL:-0.10}"

MIN_EE_BASE_Z="${MIN_EE_BASE_Z:-}"
MIN_LEFT_EE_BASE_Z="${MIN_LEFT_EE_BASE_Z:-}"
MIN_RIGHT_EE_BASE_Z="${MIN_RIGHT_EE_BASE_Z:-}"
BINARIZE_GRIPPER="${BINARIZE_GRIPPER:-true}"
GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.90}"
GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.50}"

PROMPT="${PROMPT:-${FIXED_PROMPT:-}}"
PROMPT_KEY="${PROMPT_KEY:-}"
ROLLOUT_ENABLED="${ROLLOUT_ENABLED:-false}"
ROLLOUT_FPS="${ROLLOUT_FPS:-30}"
ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT_DIR:-}"

EXTRA_ARGS=()
case "${BINARIZE_GRIPPER}" in
  1|true|TRUE|yes|YES) EXTRA_ARGS+=(--binarize-gripper) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "BINARIZE_GRIPPER must be true or false" >&2; exit 2 ;;
esac
case "${ROLLOUT_ENABLED}" in
  1|true|TRUE|yes|YES) EXTRA_ARGS+=(--rollout-enabled) ;;
  0|false|FALSE|no|NO) EXTRA_ARGS+=(--no-rollout-enabled) ;;
  *) echo "ROLLOUT_ENABLED must be true or false" >&2; exit 2 ;;
esac
if [[ -n "${GRIPPER_THRESHOLD:-}" ]]; then
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
if [[ -n "${PROMPT}" ]]; then
  EXTRA_ARGS+=(--prompt="${PROMPT}")
elif [[ -n "${PROMPT_KEY}" ]]; then
  EXTRA_ARGS+=(--prompt-key="${PROMPT_KEY}")
fi
if [[ -n "${ROLLOUT_OUTPUT_DIR}" ]]; then
  EXTRA_ARGS+=(--rollout-output-dir="${ROLLOUT_OUTPUT_DIR}")
fi

inference_log_init \
  "${SCRIPT_DIR}" \
  "inference_pi05_g1_desk_ee_prefetch" \
  "${PYTHON_BIN}" || exit $?

inference_log_run "${PYTHON_BIN}" "${SCRIPT_DIR}/inference_pi05_g1_desk_ee_prefetch.py" \
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
  --max-position-delta="${MAX_POSITION_DELTA}" \
  --max-rotation-delta="${MAX_ROTATION_DELTA}" \
  --max-retimed-chunk-steps="${MAX_RETIMED_CHUNK_STEPS}" \
  --prefetch-prefix-len="${PREFETCH_PREFIX_LEN}" \
  --prefetch-wait-poll="${PREFETCH_WAIT_POLL}" \
  --gripper-close-threshold="${GRIPPER_CLOSE_THRESHOLD}" \
  --gripper-open-threshold="${GRIPPER_OPEN_THRESHOLD}" \
  --rollout-fps="${ROLLOUT_FPS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
run_status=$?
exit "${run_status}"
