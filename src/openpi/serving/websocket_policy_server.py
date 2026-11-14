import asyncio
from concurrent.futures import ThreadPoolExecutor
import http
import itertools
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from websockets.protocol import State

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        ping_interval: float | None = 20,
        ping_timeout: float | None = 20,
        max_message_size: int = 32 * 1024 * 1024,
    ) -> None:
        if max_message_size <= 0:
            raise ValueError("max_message_size must be positive")
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._max_message_size = max_message_size
        # Policy.infer mutates its RNG and may compile JAX code. Keep all
        # requests on one worker without blocking the network event loop.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-inference")
        self._inference_lock = asyncio.Lock()
        self._request_ids = itertools.count(1)
        self._server = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        try:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=self._max_message_size,
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
                process_request=_health_check,
            ) as server:
                self._server = server
                await server.serve_forever()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _infer(self, payload):
        obs = msgpack_numpy.unpackb(payload)
        if not isinstance(obs, dict):
            raise ValueError("An inference request must be a msgpack dictionary")
        return self._policy.infer(obs)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        prev_total_time = None
        request_id = None
        stage = "metadata"
        try:
            await websocket.send(packer.pack(self._metadata))
            while True:
                stage = "receive"
                start_time = time.monotonic()
                payload = await websocket.recv()
                request_id = next(self._request_ids)
                queued_time = time.monotonic()
                stage = "queued"
                async with self._inference_lock:
                    # A client may disconnect while another request is running.
                    if websocket.state is not State.OPEN:
                        logger.info("Skipping disconnected request=%s peer=%s", request_id, websocket.remote_address)
                        return
                    queue_ms = (time.monotonic() - queued_time) * 1000
                    stage = "inference"
                    logger.info(
                        "Inference started request=%s peer=%s queue_ms=%.1f",
                        request_id, websocket.remote_address, queue_ms,
                    )
                    infer_time = time.monotonic()
                    action = await asyncio.get_running_loop().run_in_executor(self._executor, self._infer, payload)
                    infer_time = time.monotonic() - infer_time
                    logger.info("Inference finished request=%s infer_ms=%.1f", request_id, infer_time * 1000)

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                    "queue_ms": queue_ms,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                stage = "send"
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
        except websockets.ConnectionClosed:
            logger.info(
                "Connection closed peer=%s code=%s reason=%r stage=%s request=%s",
                websocket.remote_address, websocket.close_code, websocket.close_reason, stage, request_id,
            )
        except Exception:
            # Log before sending: the client may already have disconnected,
            # and a failed error response must not hide the original exception.
            error = traceback.format_exc()
            logger.exception(
                "Inference failed peer=%s stage=%s request=%s", websocket.remote_address, stage, request_id,
            )
            try:
                await websocket.send(error)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
            except websockets.ConnectionClosed:
                logger.info(
                    "Client disconnected before error delivery peer=%s request=%s", websocket.remote_address, request_id,
                )


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
