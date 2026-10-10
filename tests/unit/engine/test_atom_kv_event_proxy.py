###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

from __future__ import annotations

import socket
import time
from unittest.mock import AsyncMock

import msgspec
import pytest
import zmq

from infera.engine.atom.hooks.kv_events import _Publisher
from infera.engine.atom.kv_event_proxy import AtomKvEventProxy
from infera.engine.atom.worker import AtomEngine
from infera.router.kv_event.events import BlockRemoved, KVEventBatch


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_proxy_relays_events_from_multiple_engine_core_publishers() -> None:
    """Every DP rank may publish without racing to bind the worker endpoint."""
    context = zmq.Context.instance()
    external = f"tcp://127.0.0.1:{_free_port()}"
    proxy = AtomKvEventProxy(external)
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"kv-events")
    subscriber.setsockopt(zmq.RCVTIMEO, 100)
    publishers = [
        _Publisher(proxy.ingress_endpoint, connect=True),
        _Publisher(proxy.ingress_endpoint, connect=True),
    ]
    try:
        proxy.start()
        subscriber.connect(external)
        for publisher in publishers:
            publisher.ensure_bound()

        observed: set[int] = set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and observed != {101, 202}:
            publishers[0].block_removed(101)
            publishers[1].block_removed(202)
            try:
                topic, payload = subscriber.recv_multipart()
            except zmq.Again:
                continue
            assert topic == b"kv-events"
            batch = msgspec.msgpack.decode(payload, type=KVEventBatch)
            assert isinstance(batch.events[0], BlockRemoved)
            observed.add(batch.events[0].block_hashes[0])

        assert observed == {101, 202}
    finally:
        for publisher in publishers:
            if publisher._sock is not None:
                publisher._sock.close(linger=0)
        subscriber.close(linger=0)
        proxy.stop()


@pytest.mark.asyncio
async def test_atom_engine_connects_children_to_worker_proxy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeProxy:
        ingress_endpoint = "ipc:///tmp/infera-atom-kv-test.sock"

        def __init__(self, endpoint: str) -> None:
            captured["proxy_endpoint"] = endpoint

        def start(self) -> None:
            captured["proxy_started"] = True

        def stop(self) -> None:
            captured["proxy_stopped"] = True

    class FakeProcess:
        pid = 12345
        returncode = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float) -> None:
            captured["wait_timeout"] = timeout

    def fake_popen(cmd, *, env, start_new_session, stdout, stderr):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["start_new_session"] = start_new_session
        return FakeProcess()

    monkeypatch.setattr("infera.engine.atom.worker.AtomKvEventProxy", FakeProxy)
    monkeypatch.setattr("infera.engine.atom.worker.subprocess.Popen", fake_popen)
    monkeypatch.setattr("infera.engine.atom.worker.os.killpg", lambda *_: None)

    engine = AtomEngine(
        atom_argv=["--tensor-parallel-size", "8"],
        model_name="test/model",
        host="0.0.0.0",
        port=6100,
        advertise_host="10.0.0.1",
        kv_events_endpoint="tcp://10.0.0.1:7000",
        kv_events_bind="tcp://*:7000",
        kv_block_size=16,
    )
    engine._wait_ready = AsyncMock()

    config = await engine.start()
    env = captured["env"]
    assert isinstance(env, dict)
    assert captured["proxy_endpoint"] == "tcp://*:7000"
    assert captured["proxy_started"] is True
    assert env["INFERA_ATOM_KV_EVENTS_ENDPOINT"] == FakeProxy.ingress_endpoint
    assert env["INFERA_ATOM_KV_EVENTS_CONNECT"] == "1"
    assert config.kv_events_endpoint == "tcp://10.0.0.1:7000"

    await engine.stop()
    assert captured["proxy_stopped"] is True
