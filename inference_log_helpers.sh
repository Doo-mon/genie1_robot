#!/usr/bin/env bash
# 六个通用推理 launcher 共用的日志与退出绘图辅助函数。
#
# 调用方先 source 本文件，再初始化唯一的“时间戳+入口名+PID”运行目录，并通过
# inference_log_run 执行 Python。日志相关 Shell 变量不会 export 到父终端；只有
# INFERENCE_ACTION_LOG_DIR 以单条命令环境变量的形式传给 Python，使动作 JSONL、
# 策略输入/响应和 console.log 落在同一个目录。Python 正常结束、报错或 Ctrl-C
# 导致脚本退出时，EXIT trap 都会尝试从已落盘动作生成逐维曲线，同时严格保留
# 原 Python 退出码；无成功动作或绘图失败不会掩盖真正的推理结果。


inference_log_init() {
  if [[ "$#" -ne 3 ]]; then
    echo "inference_log_init expects: SCRIPT_DIR ENTRY_NAME PYTHON_BIN" >&2
    return 2
  fi

  local script_dir="$1"
  local entry_name="$2"
  local python_bin="$3"
  local timestamp
  local log_root
  local run_base
  local suffix=0

  log_root="${ACTION_LOG_ROOT:-${script_dir}/logs}"
  if ! timestamp="$(date '+%Y%m%d_%H%M%S')"; then
    echo "Failed to generate inference log timestamp" >&2
    return 2
  fi

  if ! mkdir -p -- "${log_root}"; then
    echo "Cannot create inference log root: ${log_root}" >&2
    return 2
  fi

  run_base="${log_root%/}/${timestamp}_${entry_name}_pid$$"
  RUN_LOG_DIR="${run_base}"
  while ! mkdir -- "${RUN_LOG_DIR}" 2>/dev/null; do
    if [[ ! -e "${RUN_LOG_DIR}" ]]; then
      echo "Cannot create inference log directory: ${RUN_LOG_DIR}" >&2
      return 2
    fi
    suffix=$((suffix + 1))
    RUN_LOG_DIR="${run_base}_${suffix}"
  done
  INFERENCE_CONSOLE_LOG="${RUN_LOG_DIR}/console.log"
  INFERENCE_LOG_SCRIPT_DIR="${script_dir}"
  INFERENCE_LOG_PYTHON_BIN="${python_bin}"
  INFERENCE_LOG_FINALIZED=0

  case "${PLOT_ACTIONS:-true}" in
    1|true|TRUE|yes|YES)
      INFERENCE_LOG_PLOT_ACTIONS=1
      ;;
    0|false|FALSE|no|NO)
      INFERENCE_LOG_PLOT_ACTIONS=0
      ;;
    *)
      echo "PLOT_ACTIONS must be true or false" >&2
      return 2
      ;;
  esac

  # Real feedback logging performs an extra synchronous SDK/FK read per
  # command.  Keep it opt-in so diagnostics cannot silently consume the
  # 20/30 Hz control budget.  console.log is always retained.
  case "${ACTION_LOGGING:-false}" in
    1|true|TRUE|yes|YES)
      INFERENCE_LOG_ACTIONS_ENABLED=true
      ;;
    0|false|FALSE|no|NO)
      INFERENCE_LOG_ACTIONS_ENABLED=false
      # With no action JSONL there is nothing useful to plot.
      INFERENCE_LOG_PLOT_ACTIONS=0
      ;;
    *)
      echo "ACTION_LOGGING must be true or false" >&2
      return 2
      ;;
  esac

  if ! : >"${INFERENCE_CONSOLE_LOG}"; then
    echo "Cannot create inference console log: ${INFERENCE_CONSOLE_LOG}" >&2
    return 2
  fi

  printf '[launcher] run_log_dir=%s\n' "${RUN_LOG_DIR}" \
    | tee -a "${INFERENCE_CONSOLE_LOG}"
  trap 'inference_log_exit_trap "$?"' EXIT
}


inference_log_run() {
  if [[ -z "${RUN_LOG_DIR:-}" || -z "${INFERENCE_CONSOLE_LOG:-}" ]]; then
    echo "inference_log_init must succeed before inference_log_run" >&2
    return 2
  fi

  # tee ignores Ctrl-C so Python remains the owner of its existing episode
  # interruption semantics.  The pipeline completes before the EXIT trap plots,
  # ensuring console.log and the Python action writer have both been closed.
  INFERENCE_ACTION_LOG_DIR="${RUN_LOG_DIR}" \
    INFERENCE_ACTION_LOG_ENABLED="${INFERENCE_LOG_ACTIONS_ENABLED}" \
    "$@" 2>&1 \
    | (trap '' INT; tee -a "${INFERENCE_CONSOLE_LOG}")
  local pipeline_statuses=("${PIPESTATUS[@]}")
  local python_status="${pipeline_statuses[0]}"
  local tee_status="${pipeline_statuses[1]}"

  if [[ "${tee_status}" -ne 0 ]]; then
    echo "[launcher] warning: console capture failed with status ${tee_status}" >&2
  fi
  return "${python_status}"
}


inference_log_finalize() {
  if [[ "${INFERENCE_LOG_FINALIZED:-0}" -eq 1 ]]; then
    return 0
  fi
  INFERENCE_LOG_FINALIZED=1

  if [[ "${INFERENCE_LOG_PLOT_ACTIONS:-0}" -ne 1 ]]; then
    return 0
  fi

  printf '[launcher] plotting action log from %s\n' "${RUN_LOG_DIR}" \
    | tee -a "${INFERENCE_CONSOLE_LOG}"
  "${INFERENCE_LOG_PYTHON_BIN}" \
    "${INFERENCE_LOG_SCRIPT_DIR}/plot_action_log.py" \
    "${RUN_LOG_DIR}" 2>&1 \
    | tee -a "${INFERENCE_CONSOLE_LOG}"
  local pipeline_statuses=("${PIPESTATUS[@]}")
  local plot_status="${pipeline_statuses[0]}"
  local tee_status="${pipeline_statuses[1]}"

  if [[ "${plot_status}" -ne 0 ]]; then
    printf '[launcher] warning: action plotting failed with status %s\n' \
      "${plot_status}" | tee -a "${INFERENCE_CONSOLE_LOG}" >&2
  fi
  if [[ "${tee_status}" -ne 0 ]]; then
    echo "[launcher] warning: plot console capture failed with status ${tee_status}" >&2
  fi
  return 0
}


inference_log_exit_trap() {
  local command_status="${1:-0}"
  trap - EXIT
  inference_log_finalize || true
  exit "${command_status}"
}
