###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Engine-agnostic e2e fixtures.

Anything backend-specific (sglang/vllm/atom worker spawning, argv mapping)
lives in the per-engine subdirectory's own conftest.py, which binds an
:class:`~tests.e2e.harness.EngineAdapter` to the shared harness. The server
here only knows about etcd + round-robin routing.

These tests require a real etcd (the worker registry transport) reachable at
``127.0.0.1:2379`` (run_tests.sh starts one on the host network). The whole e2e
suite fails if that endpoint isn't reachable.
"""

from __future__ import annotations

import contextlib
import time
import uuid

import httpx
import pytest
import pytest_asyncio

from .harness.adapter import emit_reporter_line, set_reporter
from .harness.params import EngineParams


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Announce each parametrized e2e case's full param combo on its own line so
    a running suite is self-describing. Uses the terminal reporter (not print),
    so it shows regardless of capture / -q / -v and carries no worker noise.
    """
    tr = item.config.pluginmanager.get_plugin("terminalreporter")
    capman = item.config.pluginmanager.get_plugin("capturemanager")
    if tr is not None:
        # Let the harness (adapter) emit its launch/setup lines via the same
        # reporter, suspending capture so they show live (not swallowed into the
        # per-test captured output that pytest only replays on failure).
        set_reporter(tr, capman)
    callspec = getattr(item, "callspec", None)
    params = callspec.params.get("params") if callspec else None
    if not isinstance(params, EngineParams):
        return
    if tr is None:
        return
    emit_reporter_line(
        f"[e2e param] {params.id()}  ::  model={params.model} "
        f"tp={params.tensor_parallel_size} ep={params.expert_parallel} "
        f"dp_attn={params.dp_attention} extra_args={list(params.extra_args)}"
    )


def _etcd_reachable(ep: str, *, attempts: int = 3) -> bool:
    """Whether the etcd HTTP/JSON gateway answers. Retries a few times so a
    transient blip (busy etcd, brief network hiccup) doesn't read as down."""
    base = ep if ep.startswith(("http://", "https://")) else f"http://{ep}"
    for i in range(attempts):
        try:
            r = httpx.post(f"{base}/v3/maintenance/status", json={}, timeout=2.0)
            if r.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        if i < attempts - 1:
            time.sleep(1)
    return False


@pytest.fixture(scope="session")
def etcd_endpoint() -> str:
    """The etcd endpoint for the e2e suite (fixed; run_tests.sh starts one on the
    host network). Fails if it isn't reachable — a broken/killed etcd should
    surface as a failure rather than silently masking the suite."""
    ep = "127.0.0.1:2379"
    if not _etcd_reachable(ep):
        pytest.fail(
            f"etcd gateway at {ep} isn't reachable (did etcd crash or get killed "
            "mid-run?). Failing instead of skipping so a broken etcd doesn't "
            "silently mask the e2e suite.",
        )
    return ep


@pytest.fixture
def etcd_prefix() -> str:
    """Per-test isolated prefix so concurrent test runs don't collide."""
    return f"/infera/e2e-{uuid.uuid4().hex[:8]}/"


@pytest_asyncio.fixture
async def infera_server(etcd_endpoint: str, etcd_prefix: str):
    """Factory fixture: start ONE in-process (round-robin) infera server.

    Usage::

        server = await infera_server()

    Returns the server context dict (url, etcd_endpoint, etcd_prefix).
    Only one server per test is supported (``infera.server.app`` is a
    module-level singleton); a second call raises.
    """
    # Imported lazily: it pulls uvicorn + the infera.server stack, which only the
    # PD-mixed tier (running in the engine container) needs. The PD-disaggregated
    # driver host has neither and never uses this fixture.
    from .harness.server import running_server

    stack = contextlib.AsyncExitStack()
    started: list[dict] = []

    async def _make() -> dict:
        if started:
            raise RuntimeError(
                "infera_server() supports one server per test "
                "(infera.server.app is a module-level singleton)"
            )
        ctx = await stack.enter_async_context(running_server(etcd_endpoint, etcd_prefix))
        started.append(ctx)
        return ctx

    try:
        yield _make
    finally:
        await stack.aclose()
