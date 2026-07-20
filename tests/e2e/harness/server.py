###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""In-process infera server for e2e tests.

Engine-agnostic: the server only knows about etcd + routing policy, never about
a specific engine. It round-robins across the registered workers.

NOTE: ``infera.server.app`` exposes a *module-level* FastAPI singleton and
``init_app`` mutates module globals, so only ONE server can be active per
process at a time. Tests must not stand up two servers concurrently; the
``infera_server`` fixture enforces this.
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager

import uvicorn

from infera.common.discovery import Registry
from infera.router.auto import AutoRouter
from infera.router.policy.round_robin import RoundRobinPolicy
from infera.server.app import init_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@asynccontextmanager
async def running_server(etcd_endpoint: str, etcd_prefix: str):
    """Start a real round-robin infera server bound to 127.0.0.1 on a free port.

    Yields a context dict with ``url``, ``etcd_endpoint`` and ``etcd_prefix`` —
    everything a worker fixture needs to register into the same etcd prefix the
    server watches.
    """
    registry = Registry(endpoint=etcd_endpoint, prefix=etcd_prefix)
    await registry.start()
    router = AutoRouter(registry.pool, RoundRobinPolicy())
    app = init_app(registry, router, kv=None)

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.05)
        yield {
            "url": f"http://127.0.0.1:{port}",
            "etcd_endpoint": etcd_endpoint,
            "etcd_prefix": etcd_prefix,
        }
    finally:
        server.should_exit = True
        await task
        await registry.stop()
        await router.aclose()
