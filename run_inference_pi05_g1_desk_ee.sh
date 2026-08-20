#!/usr/bin/env bash
# 启动命令：bash ./run_inference_pi05_g1_desk_ee.sh
#
# 通用同步 EE 推理入口，直接调用 inference_pi05_g1_desk_ee.py。
# 服务地址、任务名、模型/执行 horizon、控制频率和夹爪滞回阈值均可由同名环境变量覆盖。
# ROLLOUT_ENABLED 默认 false；开启后以 ROLLOUT_FPS 独立采集三路视频、
# 30 维状态和成功下发的 EE14 命令，且使用与 RTC 分离的数据集标识。
# RETIME_ACTION_CHUNK 默认 false；开启时会按位置/旋转上限展开同步
# chunk，夹爪仅在对应插值段的最后一步切换。RTC launcher 不使用该开关。
# 夹爪约定 0=张开、1=闭合；默认 close=0.90/open=0.50 且启用二值化，
# 中间区间保持上一命令。旧 GRIPPER_THRESHOLD 仅在显式设置时退回单阈值。
# MIN_LEFT_EE_BASE_Z / MIN_RIGHT_EE_BASE_Z 分别设置左右末端最低 z（米）；
# MIN_EE_BASE_Z 是兼容共同值，单臂值优先。低于阈值时会钳位后继续下发。
# WS_* 控制本地转发首次等待、握手/响应超时和退避；已发送请求默认不重发，
# 仅 WS_INFERENCE_RETRIES>0 才断线重连重发。"$@" 可作为最终参数覆盖。
# PROMPT（兼容 FIXED_PROMPT）优先于 PROMPT_KEY；两者都未设置时由 Python 显示菜单，
# 无论来自环境变量、key 还是交互选择，真正执行前都仍需人工确认。
# 请把本文件作为 launcher 执行；这里设置或导出的变量只服务于随后运行的 Python 子进程，
# 不会写回调用它的父终端环境。
# 每次运行会在 ACTION_LOG_ROOT（默认本目录/logs）建立时间戳目录并保存 console.log；
# 动作 JSONL/真实 feedback 默认关闭；ACTION_LOGGING=true 时才记录，且
# PLOT_ACTIONS 默认 true，会在退出后绘图但不改变 Python 原退出码。
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
EXECUTION_HORIZON="${EXECUTION_HORIZON:-1}"
CHUNK_SKIP="${CHUNK_SKIP:-0}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-20}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-100}"
MAX_POSITION_DELTA="${MAX_POSITION_DELTA:-0.0005}"
MAX_ROTATION_DELTA="${MAX_ROTATION_DELTA:-0.0025}"
MIN_EE_BASE_Z="${MIN_EE_BASE_Z:-}"
MIN_LEFT_EE_BASE_Z="${MIN_LEFT_EE_BASE_Z:-}"
MIN_RIGHT_EE_BASE_Z="${MIN_RIGHT_EE_BASE_Z:-}"
TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.10}"
BINARIZE_GRIPPER="${BINARIZE_GRIPPER:-true}"
GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.90}"
GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.50}"
PROMPT="${PROMPT:-${FIXED_PROMPT:-}}"
PROMPT_KEY="${PROMPT_KEY:-}"
ROLLOUT_ENABLED="${ROLLOUT_ENABLED:-false}"
ROLLOUT_FPS="${ROLLOUT_FPS:-30}"
ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT_DIR:-}"
RETIME_ACTION_CHUNK="${RETIME_ACTION_CHUNK:-false}"
MAX_RETIMED_CHUNK_STEPS="${MAX_RETIMED_CHUNK_STEPS:-500}"

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
case "${RETIME_ACTION_CHUNK}" in
  1|true|TRUE|yes|YES) EXTRA_ARGS+=(--retime-action-chunk) ;;
  0|false|FALSE|no|NO) EXTRA_ARGS+=(--no-retime-action-chunk) ;;
  *) echo "RETIME_ACTION_CHUNK must be true or false" >&2; exit 2 ;;
esac
if [[ -n "${GRIPPER_THRESHOLD:-}" ]]; then
  # Compatibility for old launch commands: one value disables hysteresis.
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
  "inference_pi05_g1_desk_ee" \
  "${PYTHON_BIN}" || exit $?

inference_log_run "${PYTHON_BIN}" "${SCRIPT_DIR}/inference_pi05_g1_desk_ee.py" \
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
  --max-position-delta="${MAX_POSITION_DELTA}" \
  --max-rotation-delta="${MAX_ROTATION_DELTA}" \
  --trajectory-time="${TRAJECTORY_TIME}" \
  --gripper-close-threshold="${GRIPPER_CLOSE_THRESHOLD}" \
  --gripper-open-threshold="${GRIPPER_OPEN_THRESHOLD}" \
  --rollout-fps="${ROLLOUT_FPS}" \
  --max-retimed-chunk-steps="${MAX_RETIMED_CHUNK_STEPS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
run_status=$?
exit "${run_status}"
