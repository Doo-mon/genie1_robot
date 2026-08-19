#!/usr/bin/env bash
# 启动命令：bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50.sh
#
# 5659 camera 任务的同步纯 EE preset：不启用 RTC，默认开启 rollout。
# rollout 保存三路视频、30 维状态和成功下发的 EE14 命令，
# processing_mode 为 ee_sync_inference_rollout，不会与 RTC 数据集混写。
# 本文件设置 5659/H50 默认值，再转交给通用 run_inference_pi05_g1_desk_ee.sh；
# 调用方已经设置的同名变量优先，"$@" 最后传给 Python，可继续显式覆盖参数。
#
# 当前默认：30 Hz、选取 H50 全部模型行，然后在客户端做同步
# chunk 重采样。重采样后任意相邻命令的位置变化不超过 0.002 m，
# SO(3) 旋转变化不超过 0.01 rad；夹爪在对应段的最后一步才切换。
# 30 Hz 下命令位置变化理论上限约 0.06 m/s。TRAJECTORY_TIME=0.03 s，
# MAX_TIMESTEPS=1500 统计重采样后实际下发的命令，不是模型 row 数。
# 单个 chunk 最多展开 500 步；超过时整个 chunk 在下发前拒绝，不会截断终点。
# 夹爪使用 0=张开、1=闭合和 close=0.9/open=0.5 滞回阈值。
#
# 动作 JSONL/绘图默认已关闭；同时关闭 rollout 时只需：
#   ROLLOUT_ENABLED=false \
#     bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50.sh
# 这时只保留较小的 console.log。ROLLOUT_OUTPUT_DIR 可指定数据集目录。
# 所有 preset 值仅注入本次子进程，不会注册或污染调用它的父终端环境。
# set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec env \
  ACTION_HORIZON="${ACTION_HORIZON:-50}" \
  EXECUTION_HORIZON="${EXECUTION_HORIZON:-50}" \
  CHUNK_SKIP="${CHUNK_SKIP:-0}" \
  TASK_NAME="${TASK_NAME:-5659}" \
  CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30}" \
  MAX_POSITION_DELTA="${MAX_POSITION_DELTA:-0.002}" \
  MAX_ROTATION_DELTA="${MAX_ROTATION_DELTA:-0.01}" \
  TRAJECTORY_TIME="${TRAJECTORY_TIME:-0.03}" \
  MAX_TIMESTEPS="${MAX_TIMESTEPS:-1500}" \
  BINARIZE_GRIPPER="${BINARIZE_GRIPPER:-true}" \
  GRIPPER_CLOSE_THRESHOLD="${GRIPPER_CLOSE_THRESHOLD:-0.9}" \
  GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.50}" \
  ROLLOUT_ENABLED="${ROLLOUT_ENABLED:-true}" \
  ROLLOUT_FPS="${ROLLOUT_FPS:-20}" \
  ROLLOUT_OUTPUT_DIR="${ROLLOUT_OUTPUT_DIR:-}" \
  RETIME_ACTION_CHUNK="${RETIME_ACTION_CHUNK:-true}" \
  MAX_RETIMED_CHUNK_STEPS="${MAX_RETIMED_CHUNK_STEPS:-500}" \
  PROMPT_KEY="${PROMPT_KEY:-named:5659-all}" \
  bash "${SCRIPT_DIR}/run_inference_pi05_g1_desk_ee.sh" "$@"
