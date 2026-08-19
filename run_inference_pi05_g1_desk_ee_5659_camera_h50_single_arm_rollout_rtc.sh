#!/usr/bin/env bash
# 启动命令：bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm_rollout_rtc.sh
#
# 5659 EE RTC + rollout 的单臂隔离测试入口。
#
# FREEZE_ARM=left（默认）：EE 与 SDK ABS_POSE 两层冻结左臂，只测试右臂。
# FREEZE_ARM=right：两层冻结右臂，只测试左臂。
# 未冻结手臂沿用 H50 RTC 的并行推理、prefix/hard-lock、单步限幅、动作日志
# 与 rollout。同步版在 chunk 之间等待 infer 的停顿不会出现在正常 RTC 队列中；
# 若服务端连续慢于 H50 可覆盖时间，仍应观察 queue starvation 日志。
#
# 轨迹平滑注意：RTC 默认 30 Hz，控制周期约为 0.033 s；下层 H50 preset
# 现在默认 TRAJECTORY_TIME=0.03，使每条 SDK 单点轨迹尽量在下一条命令前
# 完成。如果改回 0.10，上一段 100 ms 轨迹只执行约三分之一就会被重新规划，
# 可能表现为每个小时间步反复“启动/减速”、机器人晃动。相邻 EE 目标的插值
# 和限幅无法消除 SDK 层的这种重复重规划。
#
# 推荐先做短时单臂对照，让轨迹参考时间不大于控制周期（0.03 已是默认值）：
#   FREEZE_ARM=left TRAJECTORY_TIME=0.03 MAX_TIMESTEPS=30 \
#     bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm_rollout_rtc.sh \
#     --no-rollout-enabled
#
# 若 30 Hz 仍有轻微抖动，可降低到 20 Hz，并保持轨迹时间略短于 0.05 s 周期：
#   FREEZE_ARM=left CONTROL_FREQUENCY=20 TRAJECTORY_TIME=0.045 \
#     MAX_TIMESTEPS=30 \
#     bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm_rollout_rtc.sh \
#     --no-rollout-enabled
#
# 上述 FREEZE_ARM=left 表示固定左臂、只测试右臂；测试左臂时改成 right。
# 如果匹配周期后仍逐点启停，说明 SDK 的单点 ABS_POSE 接口本身按零末速度
# 收尾，需要进一步验证多点轨迹或连续伺服接口，不能只靠增加插值点解决。
#
# 示例：
#   FREEZE_ARM=left bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm_rollout_rtc.sh
#   FREEZE_ARM=right \
#     bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm_rollout_rtc.sh \
#     --no-rollout-enabled
# set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FREEZE_ARM="${FREEZE_ARM:-left}"

case "${FREEZE_ARM}" in
  left|LEFT|Left) FREEZE_ARM="left" ;;
  right|RIGHT|Right) FREEZE_ARM="right" ;;
  *)
    echo "FREEZE_ARM must be left or right for the single-arm RTC test" >&2
    exit 2
    ;;
esac

exec bash \
  "${SCRIPT_DIR}/run_inference_pi05_g1_desk_ee_5659_camera_h50_rollout_rtc.sh" \
  "$@" \
  --freeze-arm="${FREEZE_ARM}"
