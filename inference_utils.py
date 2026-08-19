"""Agibot 推理流程的通用日志、数据整理和可视化工具。

输入主要是 RobotEnv 观测字典、策略响应和动作 chunk，输出为策略请求、动作数组或日志/图片文件。
观测处理同时兼容 16 维关节模式和 14 维相机坐标末端模式，并可选拼接头部与腰部状态。
图像规范化委托给 ``inference_common.resize_image``，本模块本身不加载机器人或网络后端。
Pillow 只在保存图片时加载，Matplotlib 只在绘制轨迹时加载，普通数组处理可在无硬件环境导入。
旧推理入口可调用这里的时间线记录、延迟提取和动作处理函数，新策略模块应优先保持严格边界。
部分兼容函数会截断多余动作维度并假定固定字段存在，调用前仍需校验服务端协议和数组形状。
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TextIO

import numpy as np

from inference_common import resize_image


def get_beijing_time() -> datetime.datetime:
    """获取北京时间 (UTC+8)"""
    beijing_tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(beijing_tz)


def create_run_dir(base_dir: str = "records") -> Path:
    """创建本次运行的目录"""
    beijing_time = get_beijing_time()
    timestamp_str = beijing_time.strftime("%Y_%m%d_%H%M")
    run_dir = Path(base_dir) / timestamp_str
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class TimelineLogger:
    """
    完整时间线日志记录器。
    
    记录控制循环中每个环节的精确时间：
    - obs: 获取观测
    - pack: 打包数据  
    - infer: 推理
    - process: 处理动作
    - step: 执行动作（发送命令）
    - sleep: 频率控制等待
    - overhead: 其他开销
    """
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.log_file = run_dir / "logs.txt"
        self._file: Optional[TextIO] = open(self.log_file, "w", encoding="utf-8")
        self._reset_episode()
        
    def _reset_episode(self):
        """重置 episode 数据"""
        self._events: List[Dict] = []
        self._episode_start: Optional[datetime.datetime] = None
        self._chunk_idx = 0
        self._action_idx = 0
        
        # 按类型统计
        self._stats = {
            "obs": {"count": 0, "total_ms": 0.0},
            "pack": {"count": 0, "total_ms": 0.0},
            "infer": {"count": 0, "total_ms": 0.0, "gpu_total_ms": 0.0},
            "process": {"count": 0, "total_ms": 0.0},
            "interp": {"count": 0, "total_ms": 0.0},  # 插值时间
            "step": {"count": 0, "total_ms": 0.0},
            "sleep": {"count": 0, "total_ms": 0.0},
        }
    
    def episode_start(self, episode_idx: int, instruction: str):
        """记录 episode 开始"""
        self._reset_episode()
        self._episode_idx = episode_idx
        self._instruction = instruction
        self._episode_start = get_beijing_time()
    
    def _add_event(self, event_type: str, start: datetime.datetime, end: datetime.datetime, 
                   delta_ms: float, extra: Dict = None):
        """添加事件"""
        event = {
            "type": event_type,
            "start": start,
            "end": end,
            "delta_ms": delta_ms,
        }
        if extra:
            event.update(extra)
        self._events.append(event)
        
        # 更新统计
        if event_type in self._stats:
            self._stats[event_type]["count"] += 1
            self._stats[event_type]["total_ms"] += delta_ms
    
    def log_obs(self, start: datetime.datetime, end: datetime.datetime):
        """记录获取观测"""
        delta_ms = (end - start).total_seconds() * 1000
        self._add_event(
            "obs",
            start,
            end,
            delta_ms,
            {"num": self._stats["obs"]["count"] + 1},
        )
    
    def log_pack(self, start: datetime.datetime, end: datetime.datetime):
        """记录数据打包"""
        delta_ms = (end - start).total_seconds() * 1000
        self._add_event(
            "pack",
            start,
            end,
            delta_ms,
            {"num": self._stats["pack"]["count"] + 1},
        )
    
    def log_infer(self, start: datetime.datetime, end: datetime.datetime, 
                  delta_ms: float, client_ms: float, gpu_ms: Optional[float]):
        """记录推理"""
        self._chunk_idx += 1
        self._stats["infer"]["gpu_total_ms"] += gpu_ms if gpu_ms else 0
        self._add_event("infer", start, end, delta_ms, {
            "num": self._chunk_idx,
            "client_ms": client_ms,
            "gpu_ms": gpu_ms,
        })
    
    def log_process(self, start: datetime.datetime, end: datetime.datetime, delta_ms: float,
                    action_in_chunk: int):
        """记录动作处理"""
        self._add_event("process", start, end, delta_ms, {
            "chunk_idx": self._chunk_idx,
            "action_in_chunk": action_in_chunk,
        })
    
    def log_interp(self, start: datetime.datetime, end: datetime.datetime, delta_ms: float,
                   action_in_chunk: int, method: str = "quaternion"):
        """记录插值/平滑时间"""
        self._add_event("interp", start, end, delta_ms, {
            "chunk_idx": self._chunk_idx,
            "action_in_chunk": action_in_chunk,
            "method": method,
        })
    
    def log_step(self, start: datetime.datetime, end: datetime.datetime, delta_ms: float,
                 action_in_chunk: int):
        """记录动作执行"""
        self._action_idx += 1
        self._add_event("step", start, end, delta_ms, {
            "chunk_idx": self._chunk_idx,
            "action_in_chunk": action_in_chunk,
            "action_num": self._action_idx,
        })
    
    def log_sleep(self, start: datetime.datetime, end: datetime.datetime, delta_ms: float):
        """记录 sleep 时间"""
        self._add_event("sleep", start, end, delta_ms)
    
    def episode_end(self) -> str:
        """生成并保存 episode 日志"""
        episode_end = get_beijing_time()
        total_duration = (episode_end - self._episode_start).total_seconds()
        
        lines = []
        lines.append(f"================ Episode {self._episode_idx:02d} ================")
        lines.append(f"Instruction: {self._instruction}")
        lines.append(f"Global Start : {self._episode_start.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        lines.append(f"Global End   : {episode_end.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        lines.append(f"Total Duration: {total_duration:.3f} s")
        lines.append("")
        lines.append("Timeline (strict order)")
        
        # 时间线
        for i, event in enumerate(self._events, 1):
            start_str = event["start"].strftime("%H:%M:%S.%f")[:-3]
            end_str = event["end"].strftime("%H:%M:%S.%f")[:-3]
            etype = event["type"]
            
            if etype == "obs":
                name = f"obs#{event['num']}"
            elif etype == "pack":
                name = f"pack#{event['num']}"
            elif etype == "infer":
                gpu_str = f"{event['gpu_ms']:6.1f}" if event['gpu_ms'] else "   N/A"
                name = f"infer#{event['num']}"
                extra = f" | client={event['client_ms']:6.1f}ms | gpu={gpu_str}ms"
                lines.append(f"[{i:3d}] {name:42s} {start_str} ~ {end_str}   Δ={event['delta_ms']:7.2f} ms{extra}")
                continue
            elif etype == "process":
                name = f"process chunk#{event['chunk_idx']} idx{event['action_in_chunk']}"
            elif etype == "interp":
                method = event.get('method', 'unknown')
                name = f"interp ({method}) chunk#{event['chunk_idx']} idx{event['action_in_chunk']}"
            elif etype == "step":
                name = f"step chunk#{event['chunk_idx']} idx{event['action_in_chunk']} (action#{event['action_num']})"
            elif etype == "sleep":
                name = "sleep (freq ctrl)"
            else:
                name = etype
            
            lines.append(f"[{i:3d}] {name:42s} {start_str} ~ {end_str}   Δ={event['delta_ms']:7.2f} ms")
        
        # 统计摘要
        lines.append("")
        lines.append("=" * 64)
        lines.append("Episode Summary")
        lines.append("=" * 64)
        
        obs_s = self._stats["obs"]["total_ms"] / 1000
        pack_s = self._stats["pack"]["total_ms"] / 1000
        infer_s = self._stats["infer"]["total_ms"] / 1000
        gpu_s = self._stats["infer"]["gpu_total_ms"] / 1000
        process_s = self._stats["process"]["total_ms"] / 1000
        interp_s = self._stats["interp"]["total_ms"] / 1000
        step_s = self._stats["step"]["total_ms"] / 1000
        sleep_s = self._stats["sleep"]["total_ms"] / 1000
        
        lines.append(f"  获取观测 (obs)      : {self._stats['obs']['count']:4d} 次, 共 {obs_s:8.3f} s")
        lines.append(f"  数据打包 (pack)     : {self._stats['pack']['count']:4d} 次, 共 {pack_s:8.3f} s")
        lines.append(f"  推理 (infer)        : {self._stats['infer']['count']:4d} 次, 共 {infer_s:8.3f} s (GPU: {gpu_s:.3f} s)")
        lines.append(f"  动作处理 (process)  : {self._stats['process']['count']:4d} 次, 共 {process_s:8.3f} s")
        if self._stats["interp"]["count"] > 0:
            lines.append(f"  插值平滑 (interp)   : {self._stats['interp']['count']:4d} 次, 共 {interp_s:8.3f} s")
        lines.append(f"  命令发送 (step)     : {self._stats['step']['count']:4d} 次, 共 {step_s:8.3f} s")
        lines.append(f"  频率控制 (sleep)    : {self._stats['sleep']['count']:4d} 次, 共 {sleep_s:8.3f} s")
        
        recorded_s = obs_s + pack_s + infer_s + process_s + interp_s + step_s + sleep_s
        other_s = total_duration - recorded_s
        lines.append(f"  其他开销 (overhead) : 共 {other_s:8.3f} s")
        lines.append("-" * 64)
        lines.append(f"  总计                : {total_duration:8.3f} s")
        lines.append("")
        
        # 平均耗时
        lines.append("Average per step:")
        if self._action_idx > 0:
            lines.append(f"  每动作平均耗时      : {total_duration * 1000 / self._action_idx:.2f} ms/action")
            lines.append(f"  实际频率            : {self._action_idx / total_duration:.1f} Hz")
        lines.append("")
        
        log_content = "\n".join(lines)
        
        if self._file:
            self._file.write(log_content)
            self._file.flush()
        
        return log_content
    
    def close(self):
        if self._file:
            self._file.close()
            self._file = None


# ============ 辅助函数 ============

def prepare_observation_for_policy(
    obs: dict, 
    control_mode: Literal["joint_position", "camera_position"],
    resize_images: bool = True,
    image_size: int = 224,
    use_head_waist: bool = False,
) -> dict:
    """
    将 RobotEnv 的观测格式转换为策略服务器期望的格式。
    
    Args:
        obs: RobotEnv 返回的观测字典
        control_mode: 控制模式
        resize_images: 是否对图像进行 resize_with_pad 处理（默认 True）
        image_size: resize 目标尺寸（默认 224）
        use_head_waist: 是否在 state 中拼接头部和腰部状态（默认 False）
        
    Returns:
        策略服务器期望的请求数据格式
    """
    # 处理图像
    head_image = obs["image/head"]
    wrist_left_image = obs["image/left_wrist"]
    wrist_right_image = obs["image/right_wrist"]

    head_image = resize_image(head_image, enabled=resize_images, image_size=image_size)
    wrist_left_image = resize_image(
        wrist_left_image, enabled=resize_images, image_size=image_size
    )
    wrist_right_image = resize_image(
        wrist_right_image, enabled=resize_images, image_size=image_size
    )
    
    request_data = {
        "observation/image": head_image,
        "observation/wrist_left_image": wrist_left_image,
        "observation/wrist_right_image": wrist_right_image,
    }
    
    # 处理 state
    if control_mode == "camera_position":
        if "state/pose" not in obs:
            raise ValueError("camera_position 模式需要 state/pose")
        base_state = obs["state/pose"]  # (14,)
    else:
        base_state = obs["state/joint"]  # (16,)
    
    # 是否拼接头和腰
    if use_head_waist:
        if "state/head" not in obs:
            raise ValueError("use_head_waist=True 但观测中缺少 state/head")
        if "state/waist" not in obs:
            raise ValueError("use_head_waist=True 但观测中缺少 state/waist")
        head_state = obs["state/head"]
        waist_state = obs["state/waist"]
        # 拼接顺序：[base_state, head_state, waist_state]
        full_state = np.concatenate([base_state, head_state, waist_state])
        request_data["observation/state"] = full_state
    else:
        request_data["observation/state"] = base_state
    
    return request_data


def extract_gpu_latency_from_response(response: Dict[str, Any]) -> Optional[float]:
    """从策略服务器响应中提取 GPU 延迟信息"""
    try:
        policy_timing = response.get("policy_timing", {}) if isinstance(response, dict) else {}
        if isinstance(policy_timing, dict) and "infer_ms" in policy_timing:
            return float(policy_timing["infer_ms"])
    except (TypeError, ValueError):
        pass
    
    candidate_keys = [
        ("metadata", "gpu_latency_ms"),
        ("metadata", "gpu_ms"),
        ("latency", "gpu_ms"),
        ("profiling", "gpu_ms"),
        (None, "gpu_latency_ms"),
        (None, "gpu_ms"),
    ]
    
    for parent, key in candidate_keys:
        container = response if parent is None else response.get(parent, {})
        if isinstance(container, dict) and key in container:
            try:
                return float(container[key])
            except (TypeError, ValueError):
                continue
    
    return None


def extract_action_chunk(
    response: Dict[str, Any],
    control_mode: str,
    use_head_waist: bool = False,
) -> np.ndarray:
    """
    从策略服务器响应中提取 action chunk，统一处理 batch 维度。
    
    Args:
        response: 策略服务器响应
        control_mode: 控制模式
        use_head_waist: 是否包含头和腰的动作
        
    Returns:
        action_chunk: (chunk_size, action_dim) 的动作序列，无 batch 维度
    """
    actions = response["actions"]
    
    # 统一处理 batch 维度
    if isinstance(actions, np.ndarray):
        if actions.ndim == 3:
            # (batch, chunk_size, action_dim) -> (chunk_size, action_dim)
            actions = actions[0]
        elif actions.ndim == 2:
            # (chunk_size, action_dim) -> 保持不变
            pass
        elif actions.ndim == 1:
            # (action_dim,) -> (1, action_dim)
            actions = actions[np.newaxis, :]
    
    # 确定期望的动作维度
    if use_head_waist:
        # 包含头和腰：base_dim + head(2) + waist(2)
        expected_dim = (14 if control_mode == "camera_position" else 16) + 4
    else:
        expected_dim = 14 if control_mode == "camera_position" else 16
    
    # 如果动作维度超过期望，截取前 expected_dim 维
    if actions.shape[-1] > expected_dim:
        actions = actions[..., :expected_dim]
    
    return actions


def process_action_from_server(
    action_chunk: np.ndarray, 
    control_mode: str,
    action_idx: int,
    gripper_threshold: float = 0.5,
    use_head_waist: bool = False,
) -> np.ndarray:
    """
    处理从策略服务器返回的动作 chunk。
    
    Args:
        action_chunk: 动作序列 (chunk_size, action_dim)
        control_mode: 控制模式
        action_idx: 当前动作索引
        gripper_threshold: 夹爪二值化阈值
        use_head_waist: 是否包含头和腰的动作
        
    Returns:
        处理后的动作数组：
        - 如果 use_head_waist=False: 返回手臂动作 (14 或 16 维)
        - 如果 use_head_waist=True: 返回完整动作 (18 维: 14 arm + 2 head + 2 waist)
    """
    action = action_chunk[action_idx].copy()
    
    if control_mode == "camera_position":
        base_dim = 14
        left_gripper_idx = 6
        right_gripper_idx = 13
    else:  # joint_position
        base_dim = 16
        left_gripper_idx = 7
        right_gripper_idx = 15
    
    # 提取手臂动作
    arm_action = action[:base_dim].copy()
    
    # 二值化夹爪
    arm_action[left_gripper_idx] = 1.0 if arm_action[left_gripper_idx] > gripper_threshold else 0.0
    arm_action[right_gripper_idx] = 1.0 if arm_action[right_gripper_idx] > gripper_threshold else 0.0
    
    # 如果启用头/腰控制，返回完整动作数组
    if use_head_waist:
        if len(action) < base_dim + 4:
            raise ValueError(
                f"use_head_waist=True 但动作维度不足: 期望至少 {base_dim + 4} 维，"
                f"得到 {len(action)} 维"
            )
        head_action = action[base_dim:base_dim+2]
        waist_action = action[base_dim+2:base_dim+4]
        # 拼接: [arm(14), head(2), waist(2)] = 18维
        full_action = np.concatenate([arm_action, head_action, waist_action])
        return full_action
    else:
        # 只返回手臂动作
        return arm_action


def save_images(
    images_list: List[Dict[str, np.ndarray]],
    images_dir: Path,
    episode_idx: int
) -> None:
    """保存 episode 中的所有图像"""
    from PIL import Image

    head_dir = images_dir / "head"
    left_wrist_dir = images_dir / "left_wrist"
    right_wrist_dir = images_dir / "right_wrist"
    
    head_dir.mkdir(parents=True, exist_ok=True)
    left_wrist_dir.mkdir(parents=True, exist_ok=True)
    right_wrist_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[保存图像] 开始保存 {len(images_list)} 帧...")
    
    for step_idx, images in enumerate(images_list):
        for key, img_dir in [("head", head_dir), ("left_wrist", left_wrist_dir), ("right_wrist", right_wrist_dir)]:
            if key in images:
                img = images[key]
                if isinstance(img, np.ndarray):
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        pil_img = Image.fromarray(img, mode='RGB')
                        pil_img.save(img_dir / f"ep{episode_idx:03d}_step{step_idx:05d}.png")
    
    print(f"[保存图像] 完成！已保存到: {images_dir}")


def save_first_request_images(request_data: Dict, images_dir: Path, episode_idx: int) -> None:
    """保存第一次发送给服务端的图像"""
    from PIL import Image

    image_mapping = {
        "observation/image": "head_image",
        "observation/wrist_left_image": "left_image", 
        "observation/wrist_right_image": "right_image",
    }
    
    for key, new_name in image_mapping.items():
        if key in request_data:
            img_array = request_data[key]
            if isinstance(img_array, np.ndarray):
                filename = f"{new_name}_ep{episode_idx:03d}.png"
                filepath = images_dir / filename
                pil_img = Image.fromarray(img_array)
                pil_img.save(filepath)
                print(f"[保存] {new_name} -> {filepath}")


def print_first_request_data(request_data: Dict, chunk_idx: int) -> None:
    """打印第一次发送给服务端的数据及其维度"""
    if chunk_idx != 1:
        return
    
    print("\n" + "=" * 70)
    print("【第一次发送给服务端的数据】")
    print("=" * 70)
    for key, value in request_data.items():
        if isinstance(value, np.ndarray):
            print(f"{key:40s} | shape: {str(value.shape):20s} | dtype: {value.dtype}")
        elif isinstance(value, str):
            print(f"{key:40s} | type: str | length: {len(value)}")
            if len(value) <= 100:
                print(f"{'':40s} | content: {value}")
            else:
                print(f"{'':40s} | content: {value[:100]}...")
        else:
            print(f"{key:40s} | type: {type(value).__name__} | value: {value}")
    print("=" * 70 + "\n")


def plot_trajectory_xyz(
    timestamps: List[float],
    left_xyz: List[np.ndarray],
    right_xyz: List[np.ndarray],
    raw_times: List[float] = None,
    raw_left: List[np.ndarray] = None,
    raw_right: List[np.ndarray] = None,
    episode_idx: int = 0,
    output_dir: Path = Path("./images"),
) -> None:
    """
    绘制轨迹对比图：原始模型输出 vs 插值后执行轨迹
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    matplotlib.use('Agg')
    
    if len(timestamps) == 0:
        print("[可视化] 无轨迹数据，跳过绘图")
        return

    ts = np.array(timestamps)
    l_xyz = np.array(left_xyz)
    r_xyz = np.array(right_xyz)

    fig, axes = plt.subplots(6, 1, figsize=(12, 36), sharex=True)
    fig.suptitle(f'轨迹对比: 原始模型输出 vs 插值执行 (Episode {episode_idx})', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    labels = ['X', 'Y', 'Z']
    exec_colors = ['#FF9999', '#99FF99', '#9999FF']
    raw_colors = ['#8B0000', '#006400', '#00008B']

    for i in range(6):
        ax = axes[i]
        is_right = i >= 3
        idx = i % 3
        data = r_xyz if is_right else l_xyz
        side = "右手" if is_right else "左手"

        ax.plot(ts, data[:, idx], color=exec_colors[idx], 
                label=f'执行轨迹 {side} {labels[idx]}', linewidth=3, alpha=0.8)

        if raw_times and raw_left is not None:
            r_ts = np.array(raw_times)
            r_data = np.array(raw_right if is_right else raw_left)
            mask = r_ts <= ts.max() + 0.5 
            if np.any(mask):
                ax.scatter(r_ts[mask], r_data[mask, idx], color=raw_colors[idx], 
                           label='模型原始输出', s=80, edgecolors='white', zorder=5)

        ax.set_ylabel(f'{side} {labels[idx]} (m)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    axes[-1].set_xlabel('时间 (秒)', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_ep{episode_idx:03d}_{timestamp_str}.png"
    filepath = output_dir / filename
    
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[可视化] 轨迹图已保存: {filepath}")
