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
        video_root: str = "./review_videos",   # ← 新增
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

        # ✅ 创建视频目录
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

        frames = []

        # ✅ 用连接开始时间做名字
        start_ts = time.strftime("%Y%m%d_%H%M%S")

        video_dir = self.video_root
        video_dir.mkdir(parents=True, exist_ok=True)

        video_path = video_dir / f"{start_ts}.mp4"

        prev_total_time = None

        try:

            while True:

                start_time = time.monotonic()

                obs = msgpack_numpy.unpackb(await websocket.recv())

                img = obs["observation/image"]

                if isinstance(img, np.ndarray):
                    frames.append(img.copy())

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

            logger.info(f"Connection closed")

        finally:

            if len(frames) > 0:

                h, w, _ = frames[0].shape

                print("Saving video:", video_path)

                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (w, h),
                )

                for f in frames:
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

# import asyncio
# import http
# import logging
# import time
# import traceback

# from openpi_client import base_policy as _base_policy
# from openpi_client import msgpack_numpy
# import websockets.asyncio.server as _server
# import websockets.frames

# logger = logging.getLogger(__name__)


# class WebsocketPolicyServer:
#     """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

#     Currently only implements the `load` and `infer` methods.
#     """

#     def __init__(
#         self,
#         policy: _base_policy.BasePolicy,
#         host: str = "0.0.0.0",
#         port: int | None = None,
#         metadata: dict | None = None,
#     ) -> None:
#         self._policy = policy
#         self._host = host
#         self._port = port
#         self._metadata = metadata or {}
#         logging.getLogger("websockets.server").setLevel(logging.INFO)

#     def serve_forever(self) -> None:
#         asyncio.run(self.run())

#     async def run(self):
#         async with _server.serve(
#             self._handler,
#             self._host,
#             self._port,
#             compression=None,
#             max_size=None,
#             process_request=_health_check,
#         ) as server:
#             await server.serve_forever()

#     async def _handler(self, websocket: _server.ServerConnection):
#         logger.info(f"Connection from {websocket.remote_address} opened")
#         packer = msgpack_numpy.Packer()

#         await websocket.send(packer.pack(self._metadata))

#         prev_total_time = None
#         while True:
#             try:
#                 start_time = time.monotonic()
#                 obs = msgpack_numpy.unpackb(await websocket.recv())

#                 for k in [
#                     "observation/image",
#                     "observation/wrist_image",
#                     "observation/second_image",
#                     "observation/state",
#                 ]:
#                     v = obs[k]
#                     print(
#                         "[SERVER DEBUG]",
#                         k,
#                         "type=", type(v),
#                         "shape=", getattr(v, "shape", None),
#                         "dtype=", getattr(v, "dtype", None),
#                     )

#                 print("[SERVER DEBUG] prompt:", type(obs["prompt"]), obs["prompt"])

#                 infer_time = time.monotonic()
#                 action = self._policy.infer(obs)
#                 infer_time = time.monotonic() - infer_time

#                 action["server_timing"] = {
#                     "infer_ms": infer_time * 1000,
#                 }
#                 if prev_total_time is not None:
#                     # We can only record the last total time since we also want to include the send time.
#                     action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

#                 await websocket.send(packer.pack(action))
#                 prev_total_time = time.monotonic() - start_time

#             except websockets.ConnectionClosed:
#                 logger.info(f"Connection from {websocket.remote_address} closed")
#                 break
#             except Exception:
#                 await websocket.send(traceback.format_exc())
#                 await websocket.close(
#                     code=websockets.frames.CloseCode.INTERNAL_ERROR,
#                     reason="Internal server error. Traceback included in previous frame.",
#                 )
#                 raise


# def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
#     if request.path == "/healthz":
#         return connection.respond(http.HTTPStatus.OK, "OK\n")
#     # Continue with the normal request handling.
#     return None
