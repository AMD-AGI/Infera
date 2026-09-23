###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A readiness port that answers only while the worker is a routing target.

The engine's own ``/health`` answers as soon as sglang has loaded its weights,
which is several steps before this worker can take traffic: the PD barrier
still has to move a real KV block, and the registration that makes the router
aware of it has not happened yet. A rollout that retires the previous pod on
``/health`` therefore removes the worker that was serving in favour of one that
is not yet reachable.

This port is opened after registration and closed before deregistration, so
"accepting connections" means the same thing the router means by a live
worker.
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

#: Port the readiness endpoint listens on unless overridden. Deployments run
#: with hostNetwork, so this shares the node's port space with the engine
#: (30000) and its bootstrap port (30001) and must not collide with either.
DEFAULT_READINESS_PORT = 30090

#: Environment variable the operator sets to override the port. An env var
#: rather than a flag so the entrypoint, which operators write by hand, does
#: not have to change.
READINESS_PORT_ENV = "INFERA_READINESS_PORT"

_RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\nConnection: close\r\n\r\nready"

# A kubelet probe sends a small request and reads the reply. Bound the read so
# a half-open connection cannot pin the handler forever.
_READ_TIMEOUT_S = 5.0
_MAX_REQUEST_BYTES = 8192


def readiness_port(env: dict[str, str] | None = None) -> int:
    """Resolve the readiness port, falling back to the default when unset."""
    raw = (env if env is not None else os.environ).get(READINESS_PORT_ENV, "")
    try:
        port = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_READINESS_PORT
    return port if 0 < port < 65536 else DEFAULT_READINESS_PORT


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Answer one probe. Never raises: a probe must not kill the server."""
    try:
        try:
            # wait_for, not asyncio.timeout: the latter is 3.11+ and the
            # engine image ships 3.10, where it raises AttributeError -- which
            # left every probe unanswered and the pod permanently NotReady.
            # asyncio.TimeoutError for the same reason: only from 3.11 is it an
            # alias of the builtin, and in 3.10 the builtin is an OSError.
            await asyncio.wait_for(reader.read(_MAX_REQUEST_BYTES), _READ_TIMEOUT_S)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            # Answer anyway. The probe only cares about the status line, and a
            # client that sent nothing readable still gets a truthful "ready".
            pass
        writer.write(_RESPONSE)
        await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


async def serve_readiness(port: int | None = None) -> asyncio.AbstractServer:
    """Start accepting readiness probes on ``port``.

    Binds 0.0.0.0 because the probe arrives from the kubelet on the node, not
    from inside the container.
    """
    bind_port = readiness_port() if port is None else port
    server = await asyncio.start_server(_handle, "0.0.0.0", bind_port)
    logger.info("readiness port open on %d (worker is a routing target)", bind_port)
    return server


async def close_readiness(server: asyncio.AbstractServer | None) -> None:
    """Stop answering probes. Safe to call with None or twice."""
    if server is None:
        return
    server.close()
    try:
        await server.wait_closed()
    except (ConnectionError, OSError):
        pass
    logger.info("readiness port closed (worker is leaving the pool)")
