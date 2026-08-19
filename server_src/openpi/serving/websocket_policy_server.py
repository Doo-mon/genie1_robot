import asyncio
import http
import logging
import time
import traceback
import os
import cv2
import numpy as np
from pathlib import Path

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        video_root: str = "./review_videos",
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

        self.video_root = Path(video_root)
        self.video_root.mkdir(parents=True, exist_ok=True)

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):

        logger.info(f"Connection from {websocket.remote_address} opened")

        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        # 三路分别缓存
        frames_main = []
        frames_wrist = []
        frames_second = []

        start_ts = time.strftime("%Y%m%d_%H%M%S")

        video_dir = self.video_root
        video_dir.mkdir(parents=True, exist_ok=True)

        main_video_path = video_dir / f"{start_ts}_main.mp4"
        wrist_video_path = video_dir / f"{start_ts}_wrist.mp4"
        second_video_path = video_dir / f"{start_ts}_second.mp4"

        prev_total_time = None

        try:

            while True:

                start_time = time.monotonic()

                obs = msgpack_numpy.unpackb(await websocket.recv())

                # 取三路图像
                img_main = obs.get("observation/image", None)
                img_wrist = obs.get("observation/wrist_left_image", None)
                img_second = obs.get("observation/wrist_right_image", None)

                if isinstance(img_main, np.ndarray):
                    frames_main.append(img_main.copy())

                if isinstance(img_wrist, np.ndarray):
                    frames_wrist.append(img_wrist.copy())

                if isinstance(img_second, np.ndarray):
                    frames_second.append(img_second.copy())

                infer_time = time.monotonic()

                action = self._policy.infer(obs)

                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }

                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = (
                        prev_total_time * 1000
                    )

                await websocket.send(packer.pack(action))

                prev_total_time = time.monotonic() - start_time

        except websockets.ConnectionClosed:
            logger.info("Connection closed")

        except Exception:
            logger.exception("Internal server error")
            try:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
            except Exception:
                pass
            raise

        finally:
            self._save_video(frames_main, main_video_path)
            self._save_video(frames_wrist, wrist_video_path)
            self._save_video(frames_second, second_video_path)

    def _save_video(self, frames: list[np.ndarray], video_path: Path, fps: int = 10):
        if len(frames) == 0:
            print(f"No frames, skip saving: {video_path}")
            return

        first = frames[0]
        if not isinstance(first, np.ndarray) or first.ndim != 3:
            print(f"Invalid first frame, skip saving: {video_path}")
            return

        h, w, c = first.shape
        if c != 3:
            print(f"Expected 3-channel image, got shape={first.shape}, skip: {video_path}")
            return

        print("Saving video:", video_path)

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        for f in frames:
            if not isinstance(f, np.ndarray) or f.ndim != 3 or f.shape[2] != 3:
                continue

            if f.shape[0] != h or f.shape[1] != w:
                f = cv2.resize(f, (w, h))

            # 默认按 RGB 输入转 BGR 再写
            writer.write(f[:, :, ::-1])

        writer.release()

        print("Saved:", video_path)


def _health_check(
    connection: _server.ServerConnection,
    request: _server.Request,
) -> _server.Response | None:

    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")

    return None