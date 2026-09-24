###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The readiness port must mean "registered", not "sglang is up"."""

from __future__ import annotations

import asyncio
import socket

import pytest

from infera.engine import readiness
from infera.engine.readiness import (
    DEFAULT_READINESS_PORT,
    READINESS_PORT_ENV,
    close_readiness,
    readiness_port,
    serve_readiness,
    serve_readiness_best_effort,
)


def _port_of(server: asyncio.AbstractServer, family: int = socket.AF_INET) -> int:
    """Port of the listener in ``family``; with port 0 each family gets its own."""
    return next(s for s in server.sockets if s.family == family).getsockname()[1]


async def _probe(port: int, *, timeout: float = 5.0) -> bytes:
    """One kubelet-shaped probe. Raises if the port refuses the connection."""
    reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout)
    try:
        writer.write(b"GET /ready HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        return await asyncio.wait_for(reader.read(1024), timeout)
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_an_open_port_answers_200():
    server = await serve_readiness(0)
    try:
        assert b"200 OK" in await _probe(_port_of(server))
    finally:
        await close_readiness(server)


@pytest.mark.asyncio
async def test_a_closed_port_refuses_the_probe():
    # The whole point: once closed, the kubelet must see NotReady rather than
    # a stale success, or a rollout retires the pod that is still serving.
    server = await serve_readiness(0)
    port = _port_of(server)
    await _probe(port)

    await close_readiness(server)

    with pytest.raises((ConnectionRefusedError, OSError)):
        await _probe(port, timeout=2.0)


@pytest.mark.asyncio
async def test_repeated_probes_are_served():
    # kubelet probes every 15s for the life of the pod; one handler failing to
    # clean up would eventually wedge the listener.
    server = await serve_readiness(0)
    try:
        for _ in range(5):
            assert b"200 OK" in await _probe(_port_of(server))
    finally:
        await close_readiness(server)


@pytest.mark.asyncio
async def test_a_silent_client_does_not_wedge_the_listener():
    # A connection that opens and sends nothing must not block later probes.
    server = await serve_readiness(0)
    port = _port_of(server)
    try:
        _, w = await asyncio.open_connection("127.0.0.1", port)
        try:
            assert b"200 OK" in await _probe(port)
        finally:
            w.close()
    finally:
        await close_readiness(server)


@pytest.mark.asyncio
async def test_closing_twice_and_closing_none_are_safe():
    # Shutdown runs from a finally path that may already have closed it.
    server = await serve_readiness(0)
    await close_readiness(server)
    await close_readiness(server)
    await close_readiness(None)


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("31000", 31000),
        ("", DEFAULT_READINESS_PORT),
        ("not-a-port", DEFAULT_READINESS_PORT),
        ("0", DEFAULT_READINESS_PORT),
        ("-1", DEFAULT_READINESS_PORT),
        ("70000", DEFAULT_READINESS_PORT),
    ],
)
def test_port_resolution_falls_back_rather_than_raising(raw, want):
    # A bad value must not stop the worker from coming up; the default is
    # always a working port.
    assert readiness_port({READINESS_PORT_ENV: raw}) == want


def test_the_default_port_does_not_collide_with_the_engine():
    # Deployments run with hostNetwork, so these share the node's port space.
    assert DEFAULT_READINESS_PORT not in (30000, 30001)


@pytest.mark.asyncio
async def test_a_failed_bind_does_not_stop_the_worker():
    # Two hostNetwork workers on one node share the port space. The one that
    # binds second has already registered and loaded its weights; it keeps
    # serving rather than exiting and repeating the collision on restart.
    held = await serve_readiness(0)
    port = _port_of(held)
    try:
        assert await serve_readiness_best_effort(port) is None
    finally:
        await close_readiness(held)


@pytest.mark.asyncio
async def test_a_successful_bind_returns_the_open_server():
    server = await serve_readiness_best_effort(0)
    try:
        assert server is not None
        assert b"200 OK" in await _probe(_port_of(server))
    finally:
        await close_readiness(server)


@pytest.mark.asyncio
async def test_a_wedged_engine_is_reported_not_ready():
    # This server runs in the supervisor's loop, which stays responsive even
    # when the engine wedges. Without consulting the engine, a hung-but-running
    # engine would keep the pod Ready and a rollout could retire a healthy pod
    # for a stuck one.
    async def wedged():
        await asyncio.sleep(3600)

    server = await serve_readiness(0, engine_alive=wedged)
    try:
        assert b"503" in await _probe(_port_of(server), timeout=10.0)
    finally:
        await close_readiness(server)


@pytest.mark.asyncio
async def test_an_unreachable_engine_is_reported_not_ready():
    async def refused():
        raise ConnectionRefusedError("engine gone")

    server = await serve_readiness(0, engine_alive=refused)
    try:
        assert b"503" in await _probe(_port_of(server))
    finally:
        await close_readiness(server)


@pytest.mark.asyncio
async def test_a_healthy_engine_is_reported_ready():
    async def ok():
        return True

    server = await serve_readiness(0, engine_alive=ok)
    try:
        assert b"200 OK" in await _probe(_port_of(server))
    finally:
        await close_readiness(server)


def test_the_engine_check_fits_inside_the_probe_timeout():
    # sglang's /health often takes over a second on a busy engine. The check
    # must tolerate that, yet answer before the operator's 10s probe timeout
    # so a slow engine reads as 503 rather than as no reply.
    assert 5.0 <= readiness._ENGINE_CHECK_TIMEOUT_S < 10.0


def _ipv6_loopback_usable() -> bool:
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


@pytest.mark.asyncio
@pytest.mark.skipif(not _ipv6_loopback_usable(), reason="no IPv6 loopback")
async def test_the_port_answers_over_ipv6():
    # On an IPv6-primary cluster the kubelet probes the pod's IPv6 address.
    server = await serve_readiness(0)
    try:
        port = _port_of(server, socket.AF_INET6)
        reader, writer = await asyncio.wait_for(asyncio.open_connection("::1", port), 5.0)
        try:
            writer.write(b"GET /ready HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            assert b"200 OK" in await asyncio.wait_for(reader.read(1024), 5.0)
        finally:
            writer.close()
    finally:
        await close_readiness(server)
