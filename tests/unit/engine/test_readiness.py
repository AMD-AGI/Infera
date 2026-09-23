###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The readiness port must mean "registered", not "sglang is up"."""

from __future__ import annotations

import asyncio

import pytest

from infera.engine.readiness import (
    DEFAULT_READINESS_PORT,
    READINESS_PORT_ENV,
    close_readiness,
    readiness_port,
    serve_readiness,
)


def _port_of(server: asyncio.AbstractServer) -> int:
    return server.sockets[0].getsockname()[1]


async def _probe(port: int, *, timeout: float = 5.0) -> bytes:
    """One kubelet-shaped probe. Raises if the port refuses the connection."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection("127.0.0.1", port), timeout
    )
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
