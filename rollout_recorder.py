"""EE 与 joint 推理入口共享的 LeRobot v2.1 rollout 录制基础设施。

策略差异通过 ``RolloutSchema`` 注入；本模块统一写入三路 RGB 视频、Parquet 帧和元数据。
采集线程按独立频率读取观测，有限长度队列把视频/磁盘写入与实时控制线程隔离。
模块依赖 NumPy 与标准库；OpenCV、PyArrow 仅在真正创建 recorder 时延迟导入。
它不导入机器人或具体 policy，由两个 rollout 入口经 ``RolloutManager`` 调用。
每帧都会复制并校验 state/action 的维度与有限值，队列满时明确计数并丢帧而不阻塞控制。
结束 episode 时先原子关闭新的入队请求，再停止采集、有界等待写线程并释放视频句柄。
开始和提交前都校验既有数据集的模式、维度与 FPS；后台失败时拒绝发布残缺 episode。
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable

import numpy as np


VIDEO_KEYS = {
    "observation.images.hand_left": "image/left_wrist",
    "observation.images.hand_right": "image/right_wrist",
    "observation.images.head_color": "image/head",
}


@dataclasses.dataclass(frozen=True)
class RolloutSchema:
    """Policy-specific state/action layout used by the generic recorder."""

    name: str
    control_mode: str
    state_dim: int
    action_dim: int
    command_dim: int
    state_layout: str
    action_layout: str
    action_semantics: str
    processing_mode: str
    modality_mapping: dict[str, dict[str, int]]
    encode_frame: Callable[[dict[str, Any], np.ndarray], tuple[np.ndarray, np.ndarray]]


def _load_rollout_dependencies():
    try:
        import cv2
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Rollout recording requires opencv-python and pyarrow. Install both "
            "packages or run with --no-rollout-enabled."
        ) from error
    return cv2, pa, pq


def _rgb_uint8(image: Any) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.ndim != 3:
        raise ValueError(f"Invalid image shape: {frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise ValueError(f"Invalid image channels: {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.nan_to_num(frame.astype(np.float32))
        if frame.size and float(frame.max()) <= 1.0:
            frame *= 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _upsert_jsonl(path: Path, key: str, value: dict[str, Any]) -> None:
    """Atomically insert/replace one JSONL record, removing duplicate keys."""

    records: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]
    identity = value[key]
    output: list[dict[str, Any]] = []
    replaced = False
    for record in records:
        if record.get(key) == identity:
            if not replaced:
                output.append(value)
                replaced = True
            continue
        output.append(record)
    if not replaced:
        output.append(value)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in output:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _schema_contract(schema: RolloutSchema) -> dict[str, Any]:
    """Stable fields that must not change inside one rollout dataset."""

    return {
        "name": schema.name,
        "control_mode": schema.control_mode,
        "state_dim": schema.state_dim,
        "action_dim": schema.action_dim,
        "command_dim": schema.command_dim,
        "state_layout": schema.state_layout,
        "action_layout": schema.action_layout,
        "action_semantics": schema.action_semantics,
        "processing_mode": schema.processing_mode,
    }


def _modality_for_schema(schema: RolloutSchema) -> dict[str, Any]:
    def with_key(original_key: str) -> dict[str, Any]:
        return {
            key: {"original_key": original_key, **span}
            for key, span in schema.modality_mapping.items()
        }

    return {
        "state": with_key("observation.state"),
        "action": with_key("action"),
        "video": {
            "image_hand_left": {
                "original_key": "observation.images.hand_left"
            },
            "image_hand_right": {
                "original_key": "observation.images.hand_right"
            },
            "image_head_color": {
                "original_key": "observation.images.head_color"
            },
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
            "human.subtask_description": {"original_key": "subtask_index"},
        },
    }


def _has_rollout_artifacts(output_root: Path) -> bool:
    """Return whether a root contains dataset files that require metadata."""

    for directory_name in ("data", "videos", "meta"):
        directory = output_root / directory_name
        if directory.exists() and any(path.is_file() for path in directory.rglob("*")):
            return True
    return False


def _validate_existing_dataset_contract(
    schema: RolloutSchema,
    output_root: Path,
    fps: float,
    *,
    allow_uninitialized: bool = False,
    video_shapes: dict[str, tuple[int, int]] | None = None,
    video_codecs: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Validate an existing dataset before creating or committing an episode."""

    info_path = output_root / "meta" / "info.json"
    if not info_path.exists():
        if not allow_uninitialized and _has_rollout_artifacts(output_root):
            raise RuntimeError(
                f"Rollout root {output_root} contains data/video/meta files but "
                "has no meta/info.json; use a clean output directory."
            )
        return None

    try:
        with info_path.open("r", encoding="utf-8") as file:
            info = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read rollout dataset contract: {info_path}") from error
    if not isinstance(info, dict):
        raise RuntimeError(f"Rollout dataset info must be a JSON object: {info_path}")

    if info.get("codebase_version") != "v2.1":
        raise RuntimeError(
            "Rollout dataset codebase_version mismatch: expected 'v2.1', "
            f"got {info.get('codebase_version')!r}"
        )
    if info.get("processing_mode") != schema.processing_mode:
        raise RuntimeError(
            f"Cannot mix {schema.processing_mode!r} rollouts with existing "
            f"dataset mode {info.get('processing_mode')!r}"
        )
    try:
        previous_fps = float(info["fps"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Existing rollout dataset has an invalid fps") from error
    if not math.isfinite(previous_fps) or not math.isclose(
        previous_fps, float(fps), rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError(
            f"Cannot mix rollout fps {fps:g} with existing fps {previous_fps!r}"
        )

    features = info.get("features")
    if not isinstance(features, dict):
        raise RuntimeError("Existing rollout dataset is missing features metadata")
    expected_features = {
        "observation.state": ("float32", [schema.state_dim]),
        "action": ("float32", [schema.action_dim]),
        "timestamp": ("float32", [1]),
        "episode_index": ("int64", [1]),
        "frame_index": ("int64", [1]),
        "index": ("int64", [1]),
        "task_index": ("int64", [1]),
        "subtask_index": ("int64", [1]),
    }
    for key, (expected_dtype, expected_shape) in expected_features.items():
        feature = features.get(key)
        if (
            not isinstance(feature, dict)
            or feature.get("dtype") != expected_dtype
            or feature.get("shape") != expected_shape
        ):
            raise RuntimeError(
                f"Rollout feature {key!r} must be {expected_dtype} with shape "
                f"{expected_shape}, got {feature!r}"
            )
    for video_key in VIDEO_KEYS:
        feature = features.get(video_key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise RuntimeError(
                f"Existing rollout dataset is missing video feature {video_key!r}"
            )
        video_info = feature.get("video_info")
        if not isinstance(video_info, dict):
            raise RuntimeError(
                f"Existing rollout video feature {video_key!r} has no video_info"
            )
        try:
            video_fps = float(video_info["video.fps"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Existing rollout video feature {video_key!r} has invalid fps"
            ) from error
        if not math.isfinite(video_fps) or not math.isclose(
            video_fps, float(fps), rel_tol=1e-6, abs_tol=1e-6
        ):
            raise RuntimeError(
                f"Existing rollout video feature {video_key!r} uses fps "
                f"{video_fps!r}, expected {fps:g}"
            )
        if video_info.get("video.codec") not in {"h264", "mp4v"}:
            raise RuntimeError(
                f"Existing rollout video feature {video_key!r} has unsupported "
                f"codec {video_info.get('video.codec')!r}"
            )
        if video_shapes is not None and video_key in video_shapes:
            height, width = video_shapes[video_key]
            expected_video_shape = [3, height, width]
            if feature.get("shape") != expected_video_shape:
                raise RuntimeError(
                    f"Cannot mix video shape {expected_video_shape} for {video_key!r} "
                    f"with existing shape {feature.get('shape')!r}"
                )
        if video_codecs is not None and video_key in video_codecs:
            existing_codec = (
                video_info.get("video.codec")
            )
            if existing_codec != video_codecs[video_key]:
                raise RuntimeError(
                    f"Cannot mix video codec {video_codecs[video_key]!r} for "
                    f"{video_key!r} with existing codec {existing_codec!r}"
                )

    stored_contract = info.get("rollout_schema")
    if stored_contract is not None and stored_contract != _schema_contract(schema):
        raise RuntimeError("Existing rollout dataset schema fingerprint does not match")

    modality_path = output_root / "meta" / "modality.json"
    try:
        with modality_path.open("r", encoding="utf-8") as file:
            modality = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read rollout modality contract: {modality_path}"
        ) from error
    if modality != _modality_for_schema(schema):
        raise RuntimeError("Existing rollout modality metadata does not match the schema")
    return info


class EpisodeRecorder:
    """Record one prompt episode without blocking the control thread."""

    def __init__(
        self,
        schema: RolloutSchema,
        output_root: Path,
        episode_index: int,
        fps: float,
        rtc_prefix_len: int,
    ) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("rollout fps must be finite and positive")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise ValueError("episode_index must be a non-negative integer")
        if (
            isinstance(rtc_prefix_len, bool)
            or not isinstance(rtc_prefix_len, int)
            or rtc_prefix_len < 0
        ):
            raise ValueError("rtc_prefix_len must be a non-negative integer")
        existing_info = _validate_existing_dataset_contract(
            schema, output_root, fps
        )
        self._cv2, self._pa, self._pq = _load_rollout_dependencies()
        self.schema = schema
        self.output_root = output_root
        self.episode_index = int(episode_index)
        self.chunk_index = self.episode_index // 1000
        self.fps = float(fps)
        self.rtc_prefix_len = int(rtc_prefix_len)
        self.prompt = ""
        self.rows: list[dict[str, Any]] = []
        self.writers: dict[str, Any] = {}
        self.video_shapes: dict[str, tuple[int, int]] = {}
        self.video_codecs: dict[str, str] = {}
        self._expected_video_shapes: dict[str, tuple[int, int]] = {}
        self._expected_video_codecs: dict[str, str] = {}
        if existing_info is not None:
            for video_key in VIDEO_KEYS:
                feature = existing_info["features"][video_key]
                _, height, width = feature["shape"]
                self._expected_video_shapes[video_key] = (height, width)
                self._expected_video_codecs[video_key] = feature["video_info"][
                    "video.codec"
                ]

        self._record_queue: queue.Queue = queue.Queue(maxsize=16)
        self._sentinel = object()
        self._sentinel_enqueued = False
        self._record_error: Exception | None = None
        self._dropped_frames = 0
        self._lifecycle_lock = threading.Lock()
        self._accepting_records = True
        self._finalizing = False
        self._finalized = False
        self._record_worker = threading.Thread(
            target=self._record_worker_loop,
            name=f"{schema.name}-rollout-writer-{episode_index}",
            daemon=True,
        )
        self._record_worker.start()

        self._latest_action_lock = threading.Lock()
        self._latest_action: np.ndarray | None = None
        self._latest_action_generation = 0
        self._capture_stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._obs_provider: Callable[
            [],
            dict[str, Any]
            | tuple[dict[str, Any], np.ndarray | None, int],
        ] | None = None
        self._capture_errors = 0
        self._capture_fatal_error: Exception | None = None
        self._capture_race_drops = 0

    @property
    def _log_prefix(self) -> str:
        return f"[{self.schema.name}-rollout]"

    def set_prompt(self, prompt: str) -> None:
        self.prompt = str(prompt)

    def set_last_action(self, action: np.ndarray) -> None:
        command = np.asarray(action, dtype=np.float32).reshape(-1)
        if command.shape != (self.schema.command_dim,):
            raise ValueError(
                f"{self.schema.name} rollout command must have shape "
                f"({self.schema.command_dim},), got {command.shape}"
            )
        if not np.all(np.isfinite(command)):
            raise ValueError("Rollout command contains NaN or Inf")
        with self._latest_action_lock:
            self._latest_action = command.copy()
            self._latest_action_generation += 1

    def _get_last_action(self) -> tuple[np.ndarray | None, int]:
        with self._latest_action_lock:
            action = (
                None if self._latest_action is None else self._latest_action.copy()
            )
            return action, self._latest_action_generation

    def start_ticker(
        self,
        obs_provider: Callable[
            [],
            dict[str, Any]
            | tuple[dict[str, Any], np.ndarray | None, int],
        ],
    ) -> None:
        self._obs_provider = obs_provider
        self._capture_stop.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"{self.schema.name}-rollout-capture-{self.episode_index}",
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        period = 1.0 / self.fps
        next_tick = time.perf_counter()
        while not self._capture_stop.is_set():
            callback_action, generation = self._get_last_action()
            if self._obs_provider is not None:
                try:
                    sample = self._obs_provider()
                    if isinstance(sample, tuple):
                        if len(sample) != 3:
                            raise ValueError(
                                "Atomic rollout sample must be (obs, action, generation)"
                            )
                        obs, action, _ = sample
                        if action is not None:
                            if not self._record_captured_sample(obs, action):
                                return
                    elif callback_action is not None:
                        obs = sample
                        # Fake/custom providers use the callback fallback.  If
                        # its action changed during capture, drop the mixed pair.
                        with self._latest_action_lock:
                            generation_changed = (
                                generation != self._latest_action_generation
                            )
                        if generation_changed:
                            self._capture_race_drops += 1
                            if (
                                self._capture_race_drops == 1
                                or self._capture_race_drops % 30 == 0
                            ):
                                print(
                                    f"{self._log_prefix} capture crossed an action "
                                    f"boundary; dropped={self._capture_race_drops}"
                                )
                        else:
                            if not self._record_captured_sample(
                                obs, callback_action
                            ):
                                return
                except Exception as error:
                    self._capture_errors += 1
                    self._capture_fatal_error = error
                    self._capture_stop.set()
                    print(
                        f"{self._log_prefix} background capture failed; "
                        f"episode will not be committed: {error}"
                    )
                    return
            next_tick += period
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                self._capture_stop.wait(remaining)
            else:
                next_tick = time.perf_counter()

    def _record_captured_sample(
        self, obs: dict[str, Any], action: np.ndarray
    ) -> bool:
        """Encode/enqueue one sample, making schema errors fatal for the episode."""

        try:
            self.record(obs, action)
        except Exception as error:
            self._capture_fatal_error = error
            self._capture_stop.set()
            print(f"{self._log_prefix} capture encoding failed: {error}")
            return False
        return True

    def _stop_ticker(self) -> None:
        self._capture_stop.set()
        thread = self._capture_thread
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise RuntimeError(
                    f"{self._log_prefix} capture thread did not stop within 5s"
                )
        self._capture_thread = None

    @property
    def background_workers_stopped(self) -> bool:
        capture_stopped = (
            self._capture_thread is None or not self._capture_thread.is_alive()
        )
        return capture_stopped and not self._record_worker.is_alive()

    def _stop_record_worker(self) -> None:
        if not self._sentinel_enqueued and self._record_worker.is_alive():
            try:
                self._record_queue.put(self._sentinel, timeout=5.0)
            except queue.Full as error:
                raise RuntimeError(
                    f"{self._log_prefix} writer queue stayed full for 5s"
                ) from error
            self._sentinel_enqueued = True
        self._record_worker.join(timeout=30.0)
        if self._record_worker.is_alive():
            raise RuntimeError(
                f"{self._log_prefix} writer thread did not stop within 30s"
            )

    def record(self, obs: dict[str, Any], executed_action: np.ndarray) -> None:
        with self._lifecycle_lock:
            if not self._accepting_records:
                return

        state, action = self.schema.encode_frame(obs, executed_action)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if state.shape != (self.schema.state_dim,):
            raise ValueError(
                f"Rollout state must have shape ({self.schema.state_dim},), "
                f"got {state.shape}"
            )
        if action.shape != (self.schema.action_dim,):
            raise ValueError(
                f"Rollout action must have shape ({self.schema.action_dim},), "
                f"got {action.shape}"
            )
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            raise ValueError("Rollout state/action contains NaN or Inf")

        snapshot: dict[str, Any] = {
            "observation.state": state.copy(),
            "action": action.copy(),
        }
        for obs_key in VIDEO_KEYS.values():
            if obs_key not in obs:
                raise KeyError(f"Rollout observation is missing {obs_key!r}")
            frame = _rgb_uint8(obs[obs_key]).copy()
            video_key = next(
                key for key, source_key in VIDEO_KEYS.items() if source_key == obs_key
            )
            expected_shape = self._expected_video_shapes.get(video_key)
            if expected_shape is not None and frame.shape[:2] != expected_shape:
                raise ValueError(
                    f"Rollout frame {video_key!r} must have HxW {expected_shape}, "
                    f"got {frame.shape[:2]}"
                )
            snapshot[obs_key] = frame
        # Re-check under the same lock used by finalize.  A capture that began
        # before shutdown may finish encoding afterwards; it must not enqueue
        # behind the sentinel that terminates the writer.
        with self._lifecycle_lock:
            if not self._accepting_records:
                return
            try:
                self._record_queue.put_nowait(snapshot)
            except queue.Full:
                self._dropped_frames += 1
                if self._dropped_frames == 1 or self._dropped_frames % 30 == 0:
                    print(
                        f"{self._log_prefix} writer queue full; "
                        f"dropped={self._dropped_frames}"
                    )

    def _record_worker_loop(self) -> None:
        while True:
            item = self._record_queue.get()
            try:
                if item is self._sentinel:
                    return
                if self._record_error is None:
                    self._record_sync(item)
            except Exception as error:
                if self._record_error is None:
                    self._record_error = error
                    print(f"{self._log_prefix} background writer failed: {error}")
            finally:
                self._record_queue.task_done()

    def _video_path(self, key: str) -> Path:
        return (
            self.output_root
            / "videos"
            / f"chunk-{self.chunk_index:03d}"
            / key
            / f"episode_{self.episode_index:06d}.mp4"
        )

    def _write_frame(self, key: str, frame_rgb: np.ndarray) -> None:
        writer = self.writers.get(key)
        if writer is None:
            path = self._video_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame_rgb.shape[:2]
            required_codec = self._expected_video_codecs.get(key)
            codecs = (
                ("avc1", "H264")
                if required_codec == "h264"
                else ("mp4v",)
                if required_codec == "mp4v"
                else ("avc1", "H264", "mp4v")
            )
            for codec in codecs:
                candidate = self._cv2.VideoWriter(
                    str(path),
                    self._cv2.VideoWriter_fourcc(*codec),
                    self.fps,
                    (width, height),
                )
                if candidate.isOpened():
                    writer = candidate
                    self.video_shapes[key] = (height, width)
                    self.video_codecs[key] = (
                        "h264" if codec in ("avc1", "H264") else "mp4v"
                    )
                    break
                candidate.release()
            if writer is None:
                raise RuntimeError(f"Cannot create rollout video: {path}")
            self.writers[key] = writer

        expected = self.video_shapes[key]
        if frame_rgb.shape[:2] != expected:
            frame_rgb = self._cv2.resize(
                frame_rgb,
                (expected[1], expected[0]),
                interpolation=self._cv2.INTER_AREA,
            )
        writer.write(self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2BGR))

    def _record_sync(self, snapshot: dict[str, Any]) -> None:
        frame_index = len(self.rows)
        row = {
            "frame_index": frame_index,
            "episode_index": self.episode_index,
            "task_index": 0,
            "timestamp": frame_index / self.fps,
            "subtask_index": 0,
            "observation.state": snapshot["observation.state"].tolist(),
            "action": snapshot["action"].tolist(),
            "index": self.episode_index * 1_000_000 + frame_index,
        }
        for video_key, obs_key in VIDEO_KEYS.items():
            self._write_frame(video_key, snapshot[obs_key])
        self.rows.append(row)

    def _arrow_schema(self):
        """Exact Parquet types matching ``meta/info.json``."""

        return self._pa.schema(
            [
                self._pa.field("frame_index", self._pa.int64()),
                self._pa.field("episode_index", self._pa.int64()),
                self._pa.field("task_index", self._pa.int64()),
                self._pa.field("timestamp", self._pa.float32()),
                self._pa.field("subtask_index", self._pa.int64()),
                self._pa.field(
                    "observation.state",
                    self._pa.list_(self._pa.float32(), self.schema.state_dim),
                ),
                self._pa.field(
                    "action",
                    self._pa.list_(self._pa.float32(), self.schema.action_dim),
                ),
                self._pa.field("index", self._pa.int64()),
            ]
        )

    def finalize(self) -> None:
        with self._lifecycle_lock:
            if self._finalized:
                return
            if self._finalizing:
                raise RuntimeError(f"{self._log_prefix} finalize already in progress")
            self._finalizing = True
            self._accepting_records = False

        try:
            self._finalize_once()
        except BaseException:
            raise
        else:
            with self._lifecycle_lock:
                self._finalized = True
        finally:
            with self._lifecycle_lock:
                self._finalizing = False

    def _finalize_once(self) -> None:
        """Stop workers and publish the episode; safe to retry after a timeout."""

        cleanup_errors: list[BaseException] = []
        try:
            self._stop_ticker()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            self._stop_record_worker()
        except BaseException as error:
            cleanup_errors.append(error)
        if not self._record_worker.is_alive():
            for writer in self.writers.values():
                writer.release()
            self.writers.clear()
        if cleanup_errors:
            # Do not commit metadata while either daemon may still touch the
            # environment, queue, or video writers.
            raise RuntimeError(
                f"{self._log_prefix} failed to stop background workers: "
                + "; ".join(str(error) for error in cleanup_errors)
            ) from cleanup_errors[0]

        if self._record_error is not None:
            raise RuntimeError(
                f"{self._log_prefix} recording incomplete; episode not committed"
            ) from self._record_error
        if self._capture_fatal_error is not None:
            raise RuntimeError(
                f"{self._log_prefix} capture encoding failed; episode not committed"
            ) from self._capture_fatal_error
        if not self.rows:
            print(f"{self._log_prefix} episode has no frames; skipped")
            return

        # A second process may have initialized the root after this episode
        # began.  Re-check before writing tasks, parquet, or episode metadata.
        _validate_existing_dataset_contract(
            self.schema,
            self.output_root,
            self.fps,
            allow_uninitialized=True,
            video_shapes=self.video_shapes,
            video_codecs=self.video_codecs,
        )
        task_index = self._resolve_task_index()
        for row in self.rows:
            row["task_index"] = task_index
        data_path = (
            self.output_root
            / "data"
            / f"chunk-{self.chunk_index:03d}"
            / f"episode_{self.episode_index:06d}.parquet"
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        table = self._pa.Table.from_pylist(
            self.rows,
            schema=self._arrow_schema(),
        )
        self._pq.write_table(table, data_path)
        self._update_metadata(task_index)
        print(
            f"{self._log_prefix} episode {self.episode_index} saved: "
            f"frames={len(self.rows)} dropped={self._dropped_frames} "
            f"capture_errors={self._capture_errors} "
            f"action_boundary_drops={self._capture_race_drops} -> {data_path}"
        )

    def _resolve_task_index(self) -> int:
        meta = self.output_root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        tasks_path = meta / "tasks.jsonl"
        tasks = []
        if tasks_path.exists():
            with tasks_path.open("r", encoding="utf-8") as file:
                tasks = [json.loads(line) for line in file if line.strip()]
        for task in tasks:
            if task.get("task") == self.prompt:
                return int(task["task_index"])
        task_index = len(tasks)
        _append_jsonl(tasks_path, {"task_index": task_index, "task": self.prompt})
        return task_index

    def _update_metadata(self, task_index: int) -> None:
        meta = self.output_root / "meta"
        episodes_path = meta / "episodes.jsonl"
        subtasks_path = meta / "subtasks.jsonl"
        if not subtasks_path.exists():
            subtasks_path.touch()
        episode_record = {
            "chunk_index": self.chunk_index,
            "episode_index": self.episode_index,
            "tasks": [self.prompt],
            "subtask": {
                "task_index": task_index,
                "task": self.prompt,
                "subtask": [
                    {
                        "index": 0,
                        "subtask_index": 0,
                        "start_frame": 0,
                        "end_frame": len(self.rows) - 1,
                        "length": len(self.rows),
                        "task": self.prompt,
                    }
                ],
            },
            "length": len(self.rows),
            "rollout": {
                "control_mode": self.schema.control_mode,
                "state_layout": self.schema.state_layout,
                "action_layout": self.schema.action_layout,
                "action_semantics": self.schema.action_semantics,
                "rtc_prefix_len": self.rtc_prefix_len,
                "dropped_frames": self._dropped_frames,
                "capture_errors": self._capture_errors,
                "action_boundary_drops": self._capture_race_drops,
            },
        }

        # Publish the manifest (info.json) last.  Every earlier operation is
        # idempotent, so a transient filesystem error can safely retry the same
        # recorder without duplicating an episode or inflating totals.
        _atomic_json(meta / "modality.json", self._build_modality())
        _upsert_jsonl(
            episodes_path,
            "episode_index",
            episode_record,
        )

        with episodes_path.open("r", encoding="utf-8") as file:
            episodes = [json.loads(line) for line in file if line.strip()]
        info_path = meta / "info.json"
        info = self._build_info()
        info["total_episodes"] = len(episodes)
        info["total_frames"] = sum(int(item.get("length", 0)) for item in episodes)
        info["total_subtasks"] = sum(
            len(item.get("subtask", {}).get("subtask", []))
            for item in episodes
            if isinstance(item, dict)
        )
        info["total_chunks"] = len(
            {int(item.get("chunk_index", 0)) for item in episodes}
        )
        with (meta / "tasks.jsonl").open("r", encoding="utf-8") as file:
            info["total_tasks"] = sum(1 for line in file if line.strip())
        _atomic_json(info_path, info)

    def _build_info(self) -> dict[str, Any]:
        features: dict[str, dict[str, Any]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": [self.schema.state_dim],
            },
            "action": {"dtype": "float32", "shape": [self.schema.action_dim]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "subtask_index": {"dtype": "int64", "shape": [1]},
        }
        for key, shape in self.video_shapes.items():
            height, width = shape
            features[key] = {
                "dtype": "video",
                "shape": [3, height, width],
                "names": ["channel", "height", "width"],
                "video_info": {
                    "video.fps": self.fps,
                    "video.codec": self.video_codecs.get(key, "mp4v"),
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        return {
            "codebase_version": "v2.1",
            "fps": self.fps,
            "features": features,
            "total_episodes": 1,
            "total_frames": len(self.rows),
            "total_tasks": 1,
            "total_subtasks": 1,
            "chunks_size": 1000,
            "total_chunks": 1,
            "data_path": (
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            ),
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
            "robot_type": "g1",
            "processing_mode": self.schema.processing_mode,
            "rollout_schema": _schema_contract(self.schema),
            "instruction": self.prompt,
        }

    def _build_modality(self) -> dict[str, Any]:
        return _modality_for_schema(self.schema)


class RolloutManager:
    def __init__(
        self,
        schema: RolloutSchema,
        output_root: Path,
        fps: float,
        enabled: bool,
        rtc_prefix_len: int,
    ) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("rollout fps must be finite and positive")
        self.schema = schema
        self.output_root = output_root
        self.fps = float(fps)
        self.enabled = bool(enabled)
        self.rtc_prefix_len = int(rtc_prefix_len)
        self.recorder: EpisodeRecorder | None = None
        if self.enabled:
            # Fail before policy connection, prompt confirmation, or robot
            # reset.  Discovering a missing recorder dependency after reset
            # would move hardware and then produce no dataset.
            _load_rollout_dependencies()
            existing = _validate_existing_dataset_contract(
                self.schema, self.output_root, self.fps
            )
            if existing is not None:
                print(
                    f"[{self.schema.name}-rollout] existing dataset verified: "
                    f"mode={self.schema.processing_mode}, fps={self.fps:g}, "
                    f"episodes={int(existing.get('total_episodes', 0))}"
                )

    def _next_episode_index(self) -> int:
        indices = []
        for path in (self.output_root / "data").glob("chunk-*/episode_*.parquet"):
            try:
                indices.append(int(path.stem.split("_")[-1]))
            except ValueError:
                pass
        return max(indices, default=-1) + 1

    def begin_episode(
        self,
        prompt: str,
        obs_provider: Callable[
            [],
            dict[str, Any]
            | tuple[dict[str, Any], np.ndarray | None, int],
        ],
    ) -> None:
        if not self.enabled:
            return
        self.end_episode()
        _validate_existing_dataset_contract(
            self.schema, self.output_root, self.fps
        )
        self.recorder = EpisodeRecorder(
            self.schema,
            self.output_root,
            self._next_episode_index(),
            self.fps,
            self.rtc_prefix_len,
        )
        self.recorder.set_prompt(prompt)
        self.recorder.start_ticker(obs_provider)
        print(
            f"[{self.schema.name}-rollout] recording episode "
            f"{self.recorder.episode_index} at {self.fps:g} Hz -> "
            f"{self.output_root}"
        )

    def set_last_action(self, action: np.ndarray) -> None:
        if self.recorder is not None:
            self.recorder.set_last_action(action)

    def end_episode(self) -> None:
        recorder = self.recorder
        if recorder is not None:
            # Clear the manager reference only after a complete successful
            # commit.  A transient Parquet/metadata failure leaves the stopped
            # recorder available for the caller's outer-finally retry.
            recorder.finalize()
            self.recorder = None
