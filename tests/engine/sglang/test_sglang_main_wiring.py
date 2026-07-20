###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/engine/sglang/kvd_wiring.py.

We exercise `wire_infera_kvd_backend` end-to-end against a live kvd
daemon (in-process asyncio fixture, no subprocess) — same pattern as
the other adapter tests. The SGLang import side is gated behind
``register_kvd_backend_with_sglang`` which is itself unit-tested in
test_sglang_kvd_adapter.py; here we test the orchestration:

1. No --infera-kvd-socket → no-op (INFERA_KVD_SOCKET stays unset).
2. With --infera-kvd-socket pointing at a live daemon → env var set,
   server_args mutated to enable hicache + select infera-kvd backend.
3. With --infera-kvd-socket pointing at a missing socket → raises;
   server_args not mutated (operator gets a clean failure).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from infera.engine.sglang.kvd_wiring import wire_infera_kvd_backend
from infera.kvd.server import KvdServer


@pytest.fixture
async def kvd_daemon(tmp_path: Path):
    """Spawn a kvd daemon on a fresh UDS path; yield (server, socket_str)."""
    socket = tmp_path / f"kvd-{uuid.uuid4().hex[:8]}.sock"
    server = KvdServer(socket_path=socket, max_bytes=1 << 20)
    await server.start()
    serve_task = asyncio.create_task(server.serve_forever(), name="kvd-wiring-test")
    await asyncio.sleep(0)
    yield server, str(socket)
    server.shutdown()
    try:
        await asyncio.wait_for(serve_task, timeout=2.0)
    except asyncio.TimeoutError:
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass


def _make_args(socket_path: str | None, *, server_args_extras: dict | None = None):
    """Build a minimal SglangWorkerArgs stand-in. Doesn't import sglang
    (it may not be installed); just shapes the namespace the wiring
    function reads from."""
    server_args = SimpleNamespace(
        host="127.0.0.1",
        port=30000,
        page_size=1,
        # These two fields are what the wiring will toggle. We initialize
        # them as if the operator hadn't set them on the CLI.
        enable_hierarchical_cache=False,
        hicache_storage_backend=None,
    )
    if server_args_extras:
        for k, v in server_args_extras.items():
            setattr(server_args, k, v)
    return SimpleNamespace(
        server_args=server_args,
        infera_kvd_socket=socket_path,
    )


def test_wire_kvd_no_op_when_socket_unset(monkeypatch):
    """No flag → no environment side effects, no server_args mutation."""
    monkeypatch.delenv("INFERA_KVD_SOCKET", raising=False)
    args = _make_args(socket_path=None)
    wire_infera_kvd_backend(args)
    assert "INFERA_KVD_SOCKET" not in os.environ
    assert args.server_args.enable_hierarchical_cache is False
    assert args.server_args.hicache_storage_backend is None


@pytest.mark.asyncio
async def test_wire_kvd_with_live_daemon(kvd_daemon, monkeypatch):
    """Live daemon reachable: env var set, factory registered, hicache
    flags injected. We patch out the SGLang factory call because the
    test env doesn't have sglang installed."""
    _, socket_str = kvd_daemon
    monkeypatch.delenv("INFERA_KVD_SOCKET", raising=False)

    # Patch the SGLang factory registration to a no-op recorder so the
    # test doesn't require sglang to be importable.
    calls = []

    def _fake_register():
        calls.append("registered")

    monkeypatch.setattr(
        "infera.engine.sglang.kvd_adapter.register_kvd_backend_with_sglang",
        _fake_register,
    )

    args = _make_args(socket_path=socket_str)
    # The wiring helper itself does asyncio.run() internally — so we
    # call it from a thread to avoid clashing with this test's loop.
    await asyncio.to_thread(wire_infera_kvd_backend, args)

    assert os.environ.get("INFERA_KVD_SOCKET") == socket_str
    assert args.server_args.enable_hierarchical_cache is True
    assert args.server_args.hicache_storage_backend == "infera-kvd"
    assert calls == ["registered"]


@pytest.mark.asyncio
async def test_wire_kvd_aborts_when_socket_unreachable(tmp_path, monkeypatch):
    """Daemon not running → wiring raises. Server args remain unchanged
    so the operator's `args` dict isn't half-mutated when they catch
    the error."""
    monkeypatch.delenv("INFERA_KVD_SOCKET", raising=False)

    nonexistent = tmp_path / "no-daemon-here.sock"
    args = _make_args(socket_path=str(nonexistent))

    # Any connection-side failure should propagate. We don't pin to a
    # specific class — the asyncio.run wrapper may unwrap into different
    # parent classes across Python versions. The behavior we care about
    # is "the wiring aborts and doesn't half-mutate args".
    from infera.kvd.client import KvdConnectionError

    with pytest.raises((KvdConnectionError, OSError, ConnectionError)):
        await asyncio.to_thread(wire_infera_kvd_backend, args)

    # Even if the env var leaked (it does — we set it before probing),
    # the server_args MUST NOT have been mutated, because half-wiring is
    # worse than no wiring (SGLang would crash trying to resolve the
    # infera-kvd backend name without it being registered).
    assert args.server_args.enable_hierarchical_cache is False
    assert args.server_args.hicache_storage_backend is None


@pytest.mark.asyncio
async def test_wire_kvd_respects_operator_set_flags(kvd_daemon, monkeypatch):
    """If the operator already set --hicache-storage-backend on the
    SGLang CLI, don't clobber it. The wiring only fills in unset fields.
    """
    _, socket_str = kvd_daemon
    monkeypatch.delenv("INFERA_KVD_SOCKET", raising=False)
    monkeypatch.setattr(
        "infera.engine.sglang.kvd_adapter.register_kvd_backend_with_sglang",
        lambda: None,
    )

    args = _make_args(
        socket_path=socket_str,
        server_args_extras={
            "enable_hierarchical_cache": True,  # operator already set
            "hicache_storage_backend": "file",  # operator chose a different backend
        },
    )
    await asyncio.to_thread(wire_infera_kvd_backend, args)

    # Operator's choices preserved.
    assert args.server_args.enable_hierarchical_cache is True
    assert args.server_args.hicache_storage_backend == "file"


@pytest.mark.asyncio
async def test_wire_kvd_lowers_prefetch_threshold_default(kvd_daemon, monkeypatch):
    """PR #9 review fix P1: SGLang default prefetch_threshold (256)
    blocks short cache_control prompts from triggering L3 prefetch.
    Wiring overrides 256 → 64 unless operator chose a different value."""
    _, socket_str = kvd_daemon
    monkeypatch.delenv("INFERA_KVD_SOCKET", raising=False)
    monkeypatch.setattr(
        "infera.engine.sglang.kvd_adapter.register_kvd_backend_with_sglang",
        lambda: None,
    )
    args = _make_args(
        socket_path=socket_str,
        server_args_extras={"hicache_storage_prefetch_threshold": 256},
    )
    await asyncio.to_thread(wire_infera_kvd_backend, args)
    assert args.server_args.hicache_storage_prefetch_threshold == 64


@pytest.mark.asyncio
async def test_wire_kvd_preserves_operator_prefetch_threshold(kvd_daemon, monkeypatch):
    """Operator-set prefetch_threshold (anything != 256/None) is preserved."""
    _, socket_str = kvd_daemon
    monkeypatch.delenv("INFERA_KVD_SOCKET", raising=False)
    monkeypatch.setattr(
        "infera.engine.sglang.kvd_adapter.register_kvd_backend_with_sglang",
        lambda: None,
    )
    args = _make_args(
        socket_path=socket_str,
        server_args_extras={"hicache_storage_prefetch_threshold": 128},  # operator-chosen
    )
    await asyncio.to_thread(wire_infera_kvd_backend, args)
    assert args.server_args.hicache_storage_prefetch_threshold == 128


def test_wire_kvd_tolerates_missing_server_args_fields(monkeypatch):
    """Older SGLang versions may not have the hicache fields. The
    wiring should still register the backend and probe the daemon
    (those are the load-bearing pieces); the missing fields just
    mean the operator has to set them on the CLI themselves."""
    monkeypatch.delenv("INFERA_KVD_SOCKET", raising=False)
    monkeypatch.setattr(
        "infera.engine.sglang.kvd_adapter.register_kvd_backend_with_sglang",
        lambda: None,
    )

    # Build args with NO hicache fields on server_args.
    server_args = SimpleNamespace(host="127.0.0.1", port=30000, page_size=1)
    args = SimpleNamespace(
        server_args=server_args,
        infera_kvd_socket=None,  # so we don't need a real daemon
    )
    # No flag set → no-op path; smoke test that it doesn't blow up on
    # missing attrs (we test the live path with full fields in the
    # other tests).
    wire_infera_kvd_backend(args)
    assert not hasattr(args.server_args, "enable_hierarchical_cache")
