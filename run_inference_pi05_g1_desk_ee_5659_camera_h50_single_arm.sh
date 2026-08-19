#!/usr/bin/env bash
# 启动命令：bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm.sh
#
# 5659 同步 EE 单臂隔离测试入口。
#
# FREEZE_ARM=left（默认）：冻结左臂，只测试右臂。
# FREEZE_ARM=right：冻结右臂，只测试左臂。
# 冻结同时作用于模型 EE7 和 IK 后的 6 维 ABS_POSE/夹爪命令，避免 IK
# 分支变化让“已冻结”的手臂仍产生物理运动。
# 冻结目标取自每个 episode reset 后的初始 EE command；未冻结的手臂仍走
# 5659 H50 同步重采样、位置/SO(3) 限幅、夹爪滞回、动作日志和 rollout。
# 本脚本拒绝 none，防止本来准备做单臂测试时意外同时驱动两只手臂。
# 调用方的其余参数会继续传给 5659 通用同步 preset；最终 freeze-arm 由
# FREEZE_ARM 强制设置，不能被前面的 "$@" 意外覆盖。
#
# 示例：
#   FREEZE_ARM=left bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm.sh
#   FREEZE_ARM=right ROLLOUT_ENABLED=false \
#     bash ./run_inference_pi05_g1_desk_ee_5659_camera_h50_single_arm.sh
# set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FREEZE_ARM="${FREEZE_ARM:-left}"

case "${FREEZE_ARM}" in
  left|LEFT|Left) FREEZE_ARM="left" ;;
  right|RIGHT|Right) FREEZE_ARM="right" ;;
  *)
    echo "FREEZE_ARM must be left or right for the single-arm test" >&2
    exit 2
    ;;
esac

exec bash "${SCRIPT_DIR}/run_inference_pi05_g1_desk_ee_5659_camera_h50.sh" \
  "$@" \
  --freeze-arm="${FREEZE_ARM}"
