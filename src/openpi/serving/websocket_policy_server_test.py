import asyncio
import contextlib
import logging
import threading
import time

import numpy as np
from openpi_client import msgpack_numpy
import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError

from openpi.serving import websocket_policy_server


@contextlib.asynccontextmanager
async def _running_server(policy, **kwargs):
    server = websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=0, **kwargs)
    task = asyncio.create_task(server.run())
    try:
        async def wait_for_listener():
            while server._server is None:
                if task.done():
                    await task
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_for_listener(), timeout=5)
        port = server._server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}", port
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def test_slow_inference_keeps_heartbeats_and_health_check_responsive():
    class SlowPolicy:
        def __init__(self):
            self.started = threading.Event()
            self.active = 0
            self.max_active = 0

        def infer(self, obs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.started.set()
            # Longer than the ping timeout on both sides of the connection.
            time.sleep(0.6)
            self.active -= 1
            return {"actions": np.full((2, 7), obs["value"])}

    async def scenario():
        policy = SlowPolicy()
        async with _running_server(policy, ping_interval=0.05, ping_timeout=0.2) as (uri, port):
            async with (
                connect(uri, ping_interval=0.05, ping_timeout=0.2) as first,
                connect(uri, ping_interval=0.05, ping_timeout=0.2) as second,
            ):
                assert msgpack_numpy.unpackb(await first.recv()) == {}
                await second.recv()
                await first.send(msgpack_numpy.packb({"value": 1}))
                assert await asyncio.to_thread(policy.started.wait, 2)
                await second.send(msgpack_numpy.packb({"value": 2}))

                pong = await first.ping()
                await asyncio.wait_for(pong, timeout=0.3)
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                await writer.drain()
                response = await asyncio.wait_for(reader.read(), timeout=0.3)
                writer.close()
                await writer.wait_closed()
                assert b"200 OK" in response

                result_one = msgpack_numpy.unpackb(await asyncio.wait_for(first.recv(), timeout=3))
                result_two = msgpack_numpy.unpackb(await asyncio.wait_for(second.recv(), timeout=3))
                np.testing.assert_array_equal(result_one["actions"], 1)
                np.testing.assert_array_equal(result_two["actions"], 2)
                assert result_two["server_timing"]["queue_ms"] > 0
                assert policy.max_active == 1

    asyncio.run(scenario())


def test_error_logged_before_delivering_it_and_next_connection_can_infer(caplog):
    class Policy:
        def infer(self, obs):
            if obs.get("fail"):
                raise ValueError("invalid observation fixture")
            return {"actions": np.zeros((2, 7))}

    async def scenario():
        async with _running_server(Policy()) as (uri, _):
            async with connect(uri) as connection:
                await connection.recv()
                await connection.send(msgpack_numpy.packb({"fail": True}))
                assert "invalid observation fixture" in await connection.recv()
                with pytest.raises(ConnectionClosedError) as error:
                    await connection.recv()
                assert error.value.rcvd.code == 1011
            async with connect(uri) as connection:
                await connection.recv()
                await connection.send(msgpack_numpy.packb({}))
                assert msgpack_numpy.unpackb(await connection.recv())["actions"].shape == (2, 7)

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    assert "Inference failed" in caplog.text
    assert "invalid observation fixture" in caplog.text


def test_close_log_records_peer_reason_and_request_stage(caplog):
    class Policy:
        def infer(self, obs):
            raise AssertionError("No observation should be sent")

    async def scenario():
        async with _running_server(Policy()) as (uri, _):
            async with connect(uri) as connection:
                await connection.recv()
                await connection.close(code=1011, reason="keepalive ping timeout")

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    assert "code=1011" in caplog.text
    assert "keepalive ping timeout" in caplog.text
    assert "stage=receive" in caplog.text


def test_model_error_is_preserved_when_client_already_disconnected(caplog):
    class Policy:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def infer(self, obs):
            self.started.set()
            assert self.release.wait(timeout=3)
            raise ValueError("original failure after disconnect")

    async def scenario():
        policy = Policy()
        async with _running_server(policy) as (uri, _):
            try:
                async with connect(uri) as connection:
                    await connection.recv()
                    await connection.send(msgpack_numpy.packb({}))
                    assert await asyncio.to_thread(policy.started.wait, 2)
                    await connection.close()
                policy.release.set()

                async def wait_for_error_log():
                    while "Client disconnected before error delivery" not in caplog.text:
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(wait_for_error_log(), timeout=3)
            finally:
                policy.release.set()

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    assert "original failure after disconnect" in caplog.text
    assert "Inference failed" in caplog.text


def test_oversized_request_is_rejected_before_inference():
    class Policy:
        def infer(self, obs):
            raise AssertionError("Oversized observation reached the model")

    async def scenario():
        async with _running_server(Policy(), max_message_size=128) as (uri, _):
            async with connect(uri) as connection:
                await connection.recv()
                await connection.send(b"x" * 1024)
                with pytest.raises(ConnectionClosedError) as error:
                    await connection.recv()
                assert error.value.rcvd.code == 1009

    asyncio.run(scenario())
