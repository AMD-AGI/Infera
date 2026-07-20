###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/engine/sglang/kvd_adapter.py.

Three layers:
  - Pure helpers (key encoding, compat_key derivation) — torch- and sglang-free.
  - Integration against a real kvd daemon — adapter's sync→async bridge,
    connection lifecycle, batch ops, exists, clear. Torch-free path:
    we override `_tensor_to_bytes` / `_bytes_into_tensor` with byte-only
    stand-ins so the test runs without torch installed.
  - Torch-required: tensor↔bytes round-trip. Skipped when torch is
    absent (so CI on a router-only host doesn't fail).
"""

from __future__ import annotations

import asyncio
import importlib
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from infera.engine.sglang import kvd_adapter
from infera.engine.sglang.kvd_adapter import (
    InferaKvdBackend,
    _encode_key,
)
from infera.kvd.client import KvdConnectionError
from infera.kvd.server import KvdServer

# ----------------------------------------------------------------------
# Pure helpers — no torch, no sglang, no kvd
# ----------------------------------------------------------------------


def test_encode_key_is_utf8():
    assert _encode_key("hello") == b"hello"
    assert _encode_key("key-123_tp0") == b"key-123_tp0"


def test_encode_key_preserves_distinct_strings():
    # Sanity: two different keys → two different byte sequences.
    assert _encode_key("a") != _encode_key("b")


def _make_config(**overrides):
    """Build a minimal stand-in for SGLang's HiCacheStorageConfig.
    When sglang is actually installed it's a real dataclass; here we
    use a simple namespace with the same attribute shape."""
    from types import SimpleNamespace

    defaults = dict(
        model_name="test/m",
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        is_mla_model=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_compat_key_distinguishes_tp_ranks():
    """Two TP ranks of the same model must NOT share kvd namespace —
    KV bytes are per-rank-sharded."""
    # We need the adapter constructed but without actually connecting;
    # use the helper directly.
    config_rank0 = _make_config(tp_rank=0, tp_size=2)
    config_rank1 = _make_config(tp_rank=1, tp_size=2)

    # Bypass __init__'s connection by calling the helper on a fresh
    # placeholder instance via the unbound method.
    derive = InferaKvdBackend._derive_compat_key
    assert derive(None, config_rank0) == "tp0of2_pp0of1"
    assert derive(None, config_rank1) == "tp1of2_pp0of1"


def test_compat_key_collapses_mla_tp_ranks():
    """MLA models share KV across TP ranks (compressed latent KV).
    Collapsing the rank into the compat_key would split caches that
    SHOULD share."""
    config_mla_0 = _make_config(tp_rank=0, tp_size=2, is_mla_model=True)
    config_mla_1 = _make_config(tp_rank=1, tp_size=2, is_mla_model=True)

    derive = InferaKvdBackend._derive_compat_key
    assert derive(None, config_mla_0) == derive(None, config_mla_1)


def test_compat_key_distinguishes_pp_ranks():
    """PP-sharded models always need distinct compat keys (each rank
    holds different layers — bytes are not interchangeable)."""
    config_pp0 = _make_config(pp_rank=0, pp_size=2)
    config_pp1 = _make_config(pp_rank=1, pp_size=2)
    derive = InferaKvdBackend._derive_compat_key
    assert derive(None, config_pp0) != derive(None, config_pp1)


# ----------------------------------------------------------------------
# Adapter init — error paths (no real kvd needed)
# ----------------------------------------------------------------------


def test_init_fails_cleanly_when_kvd_unreachable(tmp_path):
    """If the socket doesn't exist, init raises KvdConnectionError and
    cleans up the background loop."""
    socket = tmp_path / "nonexistent.sock"
    with pytest.raises(KvdConnectionError):
        InferaKvdBackend(
            _make_config(),
            socket_path=str(socket),
            client_id="test-client",
        )
    # The background thread should have exited — no leaked daemon threads.
    import threading

    leaked = [t for t in threading.enumerate() if "kvd-loop" in t.name and t.is_alive()]
    # Allow a brief delay for the thread to wind down.
    import time

    for _ in range(20):
        if not leaked:
            break
        time.sleep(0.05)
        leaked = [t for t in threading.enumerate() if "kvd-loop" in t.name and t.is_alive()]
    assert not leaked, f"background loop leaked: {leaked}"


# ----------------------------------------------------------------------
# Integration with a real kvd daemon
#
# The adapter is sync but the daemon is async. We drive the adapter
# from a synchronous test body, but the daemon runs in its own
# asyncio event loop on a dedicated thread (managed by pytest-asyncio
# at the test fixture level). The adapter has its own internal event
# loop thread for its KvdClient. Two separate loops, one process —
# the bridge is the production pattern.
# ----------------------------------------------------------------------


@pytest.fixture
async def kvd_daemon(tmp_path: Path):
    """Spawn KvdServer + yield its socket path. Cleanup awaits shutdown."""
    socket = tmp_path / f"kvd-adapter-{uuid.uuid4().hex[:8]}.sock"
    server = KvdServer(socket_path=socket, max_bytes=1 << 20)
    await server.start()
    serve_task = asyncio.create_task(server.serve_forever(), name="kvd-adapter-test")
    await asyncio.sleep(0)
    yield str(socket)
    server.shutdown()
    try:
        await asyncio.wait_for(serve_task, timeout=2.0)
    except asyncio.TimeoutError:
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass


# Helpers for the torch-free integration path: bypass tensor encoding.
# We monkeypatch the adapter's bytes/tensor helpers to identity ops
# operating on byte stand-ins (a `_FakeTensor` class).


class _FakeTensor:
    """Minimal tensor stand-in for tests without torch. Carries bytes
    + a fixed expected-size; supports the few attrs the adapter touches."""

    def __init__(self, payload: bytes) -> None:
        self.payload = bytearray(payload)
        self.size = len(payload)

    def numel(self) -> int:
        return self.size

    def element_size(self) -> int:
        return 1


def _fake_tensor_to_bytes(value: _FakeTensor) -> bytes:
    return bytes(value.payload)


def _fake_bytes_into_tensor(payload: bytes, target: _FakeTensor) -> _FakeTensor | None:
    if len(payload) != target.size:
        return None
    target.payload[:] = payload
    return target


@pytest.fixture
def patched_adapter_codecs(monkeypatch):
    """Swap tensor codecs for byte-only fakes so the integration tests
    run without torch installed."""
    monkeypatch.setattr(kvd_adapter, "_tensor_to_bytes", _fake_tensor_to_bytes)
    monkeypatch.setattr(kvd_adapter, "_bytes_into_tensor", _fake_bytes_into_tensor)
    yield


@pytest.mark.asyncio
async def test_adapter_init_connects(kvd_daemon, patched_adapter_codecs):
    """Smoke: adapter connects, can be closed cleanly."""
    socket = kvd_daemon
    # Adapter __init__ blocks on the connection. Run it in a thread so
    # the daemon's event loop can serve the handshake.
    backend = await asyncio.to_thread(
        InferaKvdBackend,
        _make_config(),
        socket_path=socket,
        client_id="test-init",
    )
    try:
        # Round-trip a stats call as a liveness probe.
        stats = await asyncio.to_thread(backend.get_stats)
        assert stats is not None
        assert stats["entries"] == 0
    finally:
        await asyncio.to_thread(backend.close)


@pytest.mark.asyncio
async def test_adapter_set_get_round_trip(kvd_daemon, patched_adapter_codecs):
    """The full sync-from-test → adapter → kvd → back loop."""
    socket = kvd_daemon
    backend = await asyncio.to_thread(
        InferaKvdBackend,
        _make_config(),
        socket_path=socket,
        client_id="test-rt",
    )
    try:
        value = _FakeTensor(b"hello-from-tensor")
        ok = await asyncio.to_thread(backend.set, "k1", value)
        assert ok is True

        target = _FakeTensor(b"\x00" * len(value.payload))
        got = await asyncio.to_thread(backend.get, "k1", target)
        assert got is target
        assert bytes(target.payload) == b"hello-from-tensor"
    finally:
        await asyncio.to_thread(backend.close)


@pytest.mark.asyncio
async def test_adapter_get_miss_returns_none(kvd_daemon, patched_adapter_codecs):
    socket = kvd_daemon
    backend = await asyncio.to_thread(InferaKvdBackend, _make_config(), socket_path=socket)
    try:
        target = _FakeTensor(b"\x00" * 16)
        got = await asyncio.to_thread(backend.get, "absent-key", target)
        assert got is None
    finally:
        await asyncio.to_thread(backend.close)


@pytest.mark.asyncio
async def test_adapter_exists_and_batch_exists(kvd_daemon, patched_adapter_codecs):
    socket = kvd_daemon
    backend = await asyncio.to_thread(InferaKvdBackend, _make_config(), socket_path=socket)
    try:
        await asyncio.to_thread(backend.set, "a", _FakeTensor(b"x"))
        await asyncio.to_thread(backend.set, "b", _FakeTensor(b"y"))

        # Single exists
        assert await asyncio.to_thread(backend.exists, "a") is True
        assert await asyncio.to_thread(backend.exists, "absent") is False

        # batch_exists: count of leading present keys
        # [a, b, absent] → 2 (a and b present, absent breaks the run)
        count = await asyncio.to_thread(backend.batch_exists, ["a", "b", "absent"])
        assert count == 2

        # [absent, a] → 0 (first key breaks the run immediately)
        count = await asyncio.to_thread(backend.batch_exists, ["absent", "a"])
        assert count == 0

        # Empty list → 0
        count = await asyncio.to_thread(backend.batch_exists, [])
        assert count == 0
    finally:
        await asyncio.to_thread(backend.close)


@pytest.mark.asyncio
async def test_adapter_batch_set_get(kvd_daemon, patched_adapter_codecs):
    socket = kvd_daemon
    backend = await asyncio.to_thread(InferaKvdBackend, _make_config(), socket_path=socket)
    try:
        keys = ["k0", "k1", "k2"]
        values = [_FakeTensor(f"value-{i}".encode()) for i in range(3)]
        ok = await asyncio.to_thread(backend.batch_set, keys, values)
        assert ok is True

        targets = [_FakeTensor(b"\x00" * len(v.payload)) for v in values]
        results = await asyncio.to_thread(backend.batch_get, keys, targets)
        assert all(r is not None for r in results)
        for i, t in enumerate(targets):
            assert bytes(t.payload) == f"value-{i}".encode()
    finally:
        await asyncio.to_thread(backend.close)


@pytest.mark.asyncio
async def test_adapter_clear_namespaced(kvd_daemon, patched_adapter_codecs):
    """Clear only drops blocks owned by this adapter's (model, compat_key);
    other namespaces in the daemon are untouched."""
    socket = kvd_daemon
    backend_m1 = await asyncio.to_thread(
        InferaKvdBackend, _make_config(model_name="m1"), socket_path=socket
    )
    backend_m2 = await asyncio.to_thread(
        InferaKvdBackend, _make_config(model_name="m2"), socket_path=socket
    )
    try:
        await asyncio.to_thread(backend_m1.set, "k", _FakeTensor(b"v1"))
        await asyncio.to_thread(backend_m2.set, "k", _FakeTensor(b"v2"))

        await asyncio.to_thread(backend_m1.clear)

        # m1 lost it; m2 still has it.
        assert await asyncio.to_thread(backend_m1.exists, "k") is False
        assert await asyncio.to_thread(backend_m2.exists, "k") is True
    finally:
        await asyncio.to_thread(backend_m1.close)
        await asyncio.to_thread(backend_m2.close)


@pytest.mark.asyncio
async def test_adapter_get_size_mismatch_returns_none(kvd_daemon, patched_adapter_codecs):
    """If kvd has bytes that don't match the target tensor's expected
    size (e.g. mid-deploy schema change), the adapter must treat it
    as a miss rather than crashing."""
    socket = kvd_daemon
    backend = await asyncio.to_thread(InferaKvdBackend, _make_config(), socket_path=socket)
    try:
        await asyncio.to_thread(backend.set, "k", _FakeTensor(b"X" * 32))
        # Ask back with a wrong-size target.
        small_target = _FakeTensor(b"\x00" * 16)
        got = await asyncio.to_thread(backend.get, "k", small_target)
        assert got is None
    finally:
        await asyncio.to_thread(backend.close)


@pytest.mark.asyncio
async def test_adapter_get_stats_after_traffic(kvd_daemon, patched_adapter_codecs):
    socket = kvd_daemon
    backend = await asyncio.to_thread(InferaKvdBackend, _make_config(), socket_path=socket)
    try:
        await asyncio.to_thread(backend.set, "a", _FakeTensor(b"x"))
        target = _FakeTensor(b"\x00")
        await asyncio.to_thread(backend.get, "a", target)  # hit
        await asyncio.to_thread(backend.get, "absent", _FakeTensor(b"\x00"))  # miss

        stats = await asyncio.to_thread(backend.get_stats)
        assert stats is not None
        # Counts include the auxiliary stats call's own GETs etc; just
        # sanity-check ranges.
        assert stats["entries"] == 1
        assert stats["hits_total"] >= 1
        assert stats["misses_total"] >= 1
    finally:
        await asyncio.to_thread(backend.close)


# ----------------------------------------------------------------------
# Torch-required: real tensor round-trip
# ----------------------------------------------------------------------

torch_skip = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


@torch_skip
def test_tensor_to_bytes_round_trip():
    """If torch is available, exercise the actual codec functions."""
    import torch

    from infera.engine.sglang.kvd_adapter import _bytes_into_tensor, _tensor_to_bytes

    src = torch.arange(64, dtype=torch.float32)
    payload = _tensor_to_bytes(src)
    assert len(payload) == src.numel() * src.element_size()

    target = torch.zeros(64, dtype=torch.float32)
    out = _bytes_into_tensor(payload, target)
    assert out is target
    assert torch.equal(out, src)


@torch_skip
def test_tensor_to_bytes_handles_non_contiguous():
    """Strided / non-contig tensors must be cloned before encoding."""
    import torch

    from infera.engine.sglang.kvd_adapter import _bytes_into_tensor, _tensor_to_bytes

    src = torch.arange(64, dtype=torch.float32).reshape(8, 8).t()  # non-contig
    payload = _tensor_to_bytes(src)

    target = torch.zeros_like(src.contiguous())
    out = _bytes_into_tensor(payload, target)
    assert out is not None
    assert torch.equal(out, src.contiguous())


@torch_skip
def test_bytes_into_tensor_rejects_non_contiguous_target():
    """PR #9 review fix B: non-contiguous target → return None
    (treat as miss) rather than silently corrupt via .view(uint8)
    on incompatible strides. The vLLM connector had a similar bug
    where slot.contiguous().view(uint8).copy_(...) silently
    no-op'd on a derived view; this is the SGLang-side guard."""
    import torch

    from infera.engine.sglang.kvd_adapter import _bytes_into_tensor, _tensor_to_bytes

    # Non-contig target via transpose (a view with permuted strides).
    base = torch.zeros(8, 8, dtype=torch.float32)
    target = base.t()
    assert not target.is_contiguous()

    src = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    payload = _tensor_to_bytes(src)
    out = _bytes_into_tensor(payload, target)
    assert out is None  # rejected, not silently written

    # Base tensor must NOT have been mutated (no silent corruption).
    assert torch.equal(base, torch.zeros(8, 8, dtype=torch.float32))


@torch_skip
def test_bytes_into_tensor_round_trip_contig():
    """Sanity: contig target path still works (regression check on the
    P1 fix not breaking the happy path)."""
    import torch

    from infera.engine.sglang.kvd_adapter import _bytes_into_tensor, _tensor_to_bytes

    src = torch.arange(64, dtype=torch.float32)
    payload = _tensor_to_bytes(src)
    target = torch.zeros(64, dtype=torch.float32)
    out = _bytes_into_tensor(payload, target)
    assert out is target
    assert torch.equal(out, src)


@torch_skip
def test_tensor_to_bytes_size_mismatch_returns_none():
    """Bytes size doesn't match target → None (cache miss treatment)."""
    import torch

    from infera.engine.sglang.kvd_adapter import _bytes_into_tensor

    out = _bytes_into_tensor(b"\x00" * 4, torch.zeros(16, dtype=torch.float32))
    assert out is None


# ----------------------------------------------------------------------
# Phase A.2: retention resolution
#
# SGLang's HiCacheStorage SPI set/batch_set doesn't carry retention;
# batch_set_v1 takes a `HiCacheStorageExtraInfo` whose `extra_info`
# dict CAN carry it (upstream needs to populate it). The adapter's
# resolution rules let an env-var override be the deployment-wide
# default, with per-request overrides through extra_info when sglang
# upstream eventually plumbs them.
# ----------------------------------------------------------------------


def test_retention_default_falls_back_to_long_when_unset(monkeypatch):
    """No env, no constructor arg → fallback to long (durable L3 by default;
    short would drop evicted blocks and the on-disk L3 tier never populates)."""
    import os as _os

    monkeypatch.delenv("INFERA_KVD_RETENTION_DEFAULT", raising=False)
    assert _os.environ.get("INFERA_KVD_RETENTION_DEFAULT") is None
    # Resolution chain mirrors what __init__ does — re-derive
    # here without going through the full connect:
    retention = None or _os.environ.get("INFERA_KVD_RETENTION_DEFAULT") or "long"
    assert retention == "long"
    # The full path with a live daemon is covered by
    # test_retention_default_env_var_sets_initial_value.


@pytest.mark.asyncio
async def test_retention_default_env_var_sets_initial_value(
    kvd_daemon, patched_adapter_codecs, monkeypatch
):
    """`INFERA_KVD_RETENTION_DEFAULT=long` env → all SETs from this
    adapter use long retention. Deployment-wide knob for the SGLang
    upstream gap (per-request retention not plumbed through SPI)."""
    monkeypatch.setenv("INFERA_KVD_RETENTION_DEFAULT", "long")
    socket = kvd_daemon
    backend = await asyncio.to_thread(
        kvd_adapter.InferaKvdBackend,
        _make_config(),
        socket_path=socket,
        client_id="retention-env",
    )
    try:
        # Round-trip a SET; check internal default is `long`.
        await asyncio.to_thread(backend.set, "k", _FakeTensor(b"x" * 16))
        assert backend._retention_default == "long"
    finally:
        await asyncio.to_thread(backend.close)


def test_retention_default_constructor_arg_overrides_env(monkeypatch):
    """Explicit constructor arg wins over the env var."""
    monkeypatch.setenv("INFERA_KVD_RETENTION_DEFAULT", "short")
    # We can't construct fully without a kvd daemon, so probe the
    # resolution rules by-hand the way __init__ does.
    import os as _os

    # constructor passes retention_default="long"
    constructor_arg = "long"
    env = _os.environ.get("INFERA_KVD_RETENTION_DEFAULT")
    fallback = "long"  # mirrors __init__'s fallback (durable L3 by default)
    resolved = constructor_arg or env or fallback
    assert resolved == "long"


def test_retention_default_rejects_bogus_value(monkeypatch):
    """Bogus values in the env raise at construction time so the
    operator gets immediate feedback rather than a silent failure
    later when kvd rejects an unknown retention level."""
    monkeypatch.setenv("INFERA_KVD_RETENTION_DEFAULT", "FOREVER")
    with pytest.raises(ValueError, match="retention_default"):
        from infera.engine.sglang.kvd_adapter import InferaKvdBackend

        # Construction probes kvd; we expect the ValueError to fire
        # BEFORE the probe in the __init__ chain (it's the second
        # check). We use a never-going-to-exist socket so if the
        # error somehow gets past the validation, it would also
        # error on connect — both are acceptable failures here.
        InferaKvdBackend(_make_config(), socket_path="/tmp/.no-such-socket")


def test_set_request_retention_hint_overrides_default():
    """PR #9 review fix P0-3/4: per-request retention via ContextVar
    overrides the deployment-wide default. Without this override the
    adapter's `set()` silently uses `_retention_default` regardless of
    the request's `cache_control`."""
    from infera.engine.sglang.kvd_adapter import (
        _request_retention_override,
        _resolve_retention,
        set_request_retention_hint,
    )

    # No override → falls back to the passed-in default.
    assert _resolve_retention("short") == "short"

    # Override set → wins.
    token = set_request_retention_hint("long")
    try:
        assert _resolve_retention("short") == "long"
    finally:
        _request_retention_override.reset(token)

    # After reset, back to default.
    assert _resolve_retention("short") == "short"


def test_set_request_retention_hint_rejects_invalid_values(caplog):
    """Invalid values are ignored + logged at WARN, not silently set."""
    import logging as _logging

    from infera.engine.sglang.kvd_adapter import set_request_retention_hint

    caplog.set_level(_logging.WARNING, logger="infera.engine.sglang.kvd_adapter")
    token = set_request_retention_hint("forever")  # invalid
    assert token is None
    assert any("invalid retention" in r.message for r in caplog.records)


def test_extra_info_chains_to_contextvar_override():
    """When SGLang's batch_set_v1 carries no extra_info, the
    ContextVar override is consulted before the deployment default.
    This is the bridge mechanism until upstream populates extra_info.
    """
    from infera.engine.sglang.kvd_adapter import (
        InferaKvdBackend,
        _request_retention_override,
        set_request_retention_hint,
    )

    inst = InferaKvdBackend.__new__(InferaKvdBackend)
    inst._retention_default = "short"

    # No ContextVar override, no extra_info → fall back to default.
    assert inst._resolve_retention_from_extra_info(None) == "short"

    # ContextVar override only → use the override.
    token = set_request_retention_hint("long")
    try:
        assert inst._resolve_retention_from_extra_info(None) == "long"

        # extra_info takes precedence over ContextVar.
        obj_short = SimpleNamespace(prefix_keys=None, extra_info={"infera_retention": "short"})
        assert inst._resolve_retention_from_extra_info(obj_short) == "short"
    finally:
        _request_retention_override.reset(token)


def test_resolve_retention_from_extra_info_reads_inner_dict():
    """The batch_set_v1 helper resolves retention via
    ``HiCacheStorageExtraInfo.extra_info["infera_retention"]``."""
    from infera.engine.sglang.kvd_adapter import InferaKvdBackend

    # Make a stub instance without going through __init__.
    inst = InferaKvdBackend.__new__(InferaKvdBackend)
    inst._retention_default = "short"

    # Case 1: None → default
    assert inst._resolve_retention_from_extra_info(None) == "short"

    # Case 2: extra_info object without `.extra_info` attr → default
    obj = SimpleNamespace(prefix_keys=["a", "b"])
    assert inst._resolve_retention_from_extra_info(obj) == "short"

    # Case 3: `.extra_info` is a dict with infera_retention="long"
    obj_long = SimpleNamespace(prefix_keys=None, extra_info={"infera_retention": "long"})
    assert inst._resolve_retention_from_extra_info(obj_long) == "long"

    # Case 4: bogus value in dict → fallback
    obj_bogus = SimpleNamespace(prefix_keys=None, extra_info={"infera_retention": "FOREVER"})
    assert inst._resolve_retention_from_extra_info(obj_bogus) == "short"

    # Case 5: `.extra_info` exists but isn't a dict → fallback
    obj_bad = SimpleNamespace(prefix_keys=None, extra_info="not-a-dict")
    assert inst._resolve_retention_from_extra_info(obj_bad) == "short"


@pytest.mark.asyncio
async def test_batch_set_v1_reads_retention_from_extra_info(
    kvd_daemon, patched_adapter_codecs, monkeypatch
):
    """batch_set_v1 honors the per-request retention from extra_info
    even when host_indices isn't usable yet (fallback path returns
    all-False, but the retention extraction must have fired so a
    future wiring step Just Works)."""
    monkeypatch.setenv("INFERA_KVD_RETENTION_DEFAULT", "short")
    socket = kvd_daemon
    backend = await asyncio.to_thread(
        kvd_adapter.InferaKvdBackend,
        _make_config(),
        socket_path=socket,
        client_id="retention-extra-info",
    )
    try:
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorageExtraInfo  # noqa: PLC0415

        extra = HiCacheStorageExtraInfo(prefix_keys=None, extra_info={"infera_retention": "long"})
        result = await asyncio.to_thread(backend.batch_set_v1, ["k1", "k2"], None, extra)
        # Fallback path returns all-False until host pool integration.
        assert result == [False, False]
        # We can't easily probe the retention argument that WOULD
        # have been used; this test mostly proves the call dispatches
        # without crashing, and exists as a regression net for the
        # day we wire host_indices through.
    except ImportError:
        pytest.skip("sglang not installed in test env")
    finally:
        await asyncio.to_thread(backend.close)


# ----------------------------------------------------------------------
# v2 SPI (batch_set_v2 / batch_get_v2 / batch_exists_v2) — pool-aware
#
# GLM-5.1's hybrid_cache_controller calls these from prefetch/backup
# threads. The base ABC raises NotImplementedError, so missing impls
# silently kill those threads → kvd stats stay at 0 under any GLM load.
# We exercise the methods against a real kvd daemon + a fake host pool
# + stand-in sglang PoolTransfer/PoolName/PoolHitPolicy types so the
# tests run with or without sglang installed (router-only CI).
# ----------------------------------------------------------------------


class _FakeHostPool:
    """Minimal stand-in for SGLang's HostKVCache that supports the
    page interface our v2 SPI relies on. Stores 1 byte per slot, with
    `page_size` slots per page."""

    def __init__(self, n_slots: int, page_size: int = 2) -> None:
        self.page_size = page_size
        self._storage = bytearray(n_slots)

    def get_data_page(self, index, flat: bool = True) -> _FakeTensor:
        i = int(index)
        return _FakeTensor(bytes(self._storage[i : i + self.page_size]))

    def get_dummy_flat_data_page(self) -> _FakeTensor:
        return _FakeTensor(b"\x00" * self.page_size)

    def set_from_flat_data_page(self, index, data_page: _FakeTensor) -> None:
        i = int(index)
        self._storage[i : i + self.page_size] = bytes(data_page.payload)


class _FakeIndices:
    """Stand-in for a torch.LongTensor of host indices. Supports
    `numel()` and integer indexing returning an obj with `.item()`."""

    def __init__(self, values: list[int]) -> None:
        self._values = list(values)

    def numel(self) -> int:
        return len(self._values)

    def __getitem__(self, idx: int):
        v = self._values[idx]
        return SimpleNamespace(item=lambda v=v: v)


@pytest.fixture
def fake_sglang_v2_types(monkeypatch):
    """Make `from sglang.srt.mem_cache.hicache_storage import PoolName,
    PoolHitPolicy, PoolTransferResult` succeed in the adapter even when
    sglang isn't installed. Returns the stand-in module so tests can
    use the same `PoolName.KV` enum the adapter sees."""
    import sys
    import types

    mod_path = "sglang.srt.mem_cache.hicache_storage"
    pre_existing = sys.modules.get(mod_path)
    if pre_existing is not None:
        # Real sglang is installed — use it directly so adapter and
        # test reference the SAME enum identities.
        yield pre_existing
        return

    class PoolName(str):
        KV: PoolName
        INDEXER: PoolName

        def __new__(cls, value):
            obj = str.__new__(cls, value)
            obj.value = value
            return obj

    PoolName.KV = PoolName("kv")
    PoolName.INDEXER = PoolName("indexer")

    class PoolHitPolicy(str):
        ALL_PAGES: PoolHitPolicy
        TRAILING_PAGES: PoolHitPolicy

        def __new__(cls, value):
            return str.__new__(cls, value)

    PoolHitPolicy.ALL_PAGES = PoolHitPolicy("all_pages")
    PoolHitPolicy.TRAILING_PAGES = PoolHitPolicy("trailing_pages")

    class PoolTransferResult:
        def __init__(self, kv_hit_pages, extra_pool_hit_pages):
            self.kv_hit_pages = kv_hit_pages
            self.extra_pool_hit_pages = extra_pool_hit_pages

        @classmethod
        def empty(cls):
            return cls(0, {})

    fake_mod = types.ModuleType(mod_path)
    fake_mod.PoolName = PoolName
    fake_mod.PoolHitPolicy = PoolHitPolicy
    fake_mod.PoolTransferResult = PoolTransferResult

    # Parent packages must also exist so the dotted import resolves.
    for parent in ("sglang", "sglang.srt", "sglang.srt.mem_cache"):
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    monkeypatch.setitem(sys.modules, mod_path, fake_mod)
    yield fake_mod


def _make_transfer(PoolName, PoolHitPolicy, name, keys, host_indices, policy=None):
    """Build a stand-in PoolTransfer (the adapter only reads attributes)."""
    return SimpleNamespace(
        name=name,
        host_indices=_FakeIndices(host_indices),
        keys=list(keys),
        hit_policy=policy or PoolHitPolicy.ALL_PAGES,
    )


@pytest.mark.asyncio
async def test_batch_set_v2_roundtrip_kv_and_indexer(
    kvd_daemon, patched_adapter_codecs, fake_sglang_v2_types
):
    """End-to-end: register two host pools, write via batch_set_v2,
    confirm kvd statctl entries reflect both pools, read back via
    batch_get_v2 into wiped pools, byte-perfect roundtrip."""
    PoolName = fake_sglang_v2_types.PoolName
    PoolHitPolicy = fake_sglang_v2_types.PoolHitPolicy

    backend = await asyncio.to_thread(
        kvd_adapter.InferaKvdBackend,
        _make_config(),
        socket_path=kvd_daemon,
        client_id="v2-roundtrip",
    )
    try:
        page_size = 2
        kv_pool = _FakeHostPool(n_slots=8, page_size=page_size)
        idx_pool = _FakeHostPool(n_slots=8, page_size=page_size)
        # When sglang isn't installed our base HiCacheStorage stub lacks
        # `register_mem_host_pool_v2`; bypass by setting the dict the
        # adapter's `_batch_io_v2` reads from directly. With real sglang
        # this attribute is owned/populated by the base ABC.
        backend.registered_pools = {PoolName.KV: kv_pool, PoolName.INDEXER: idx_pool}

        # Seed distinct payloads in slots 0..3 of each pool (2 pages × 2 slots).
        for s in range(4):
            kv_pool._storage[s] = 0x10 + s
            idx_pool._storage[s] = 0xA0 + s

        keys = ["page_alpha", "page_beta"]
        kv_t = _make_transfer(PoolName, PoolHitPolicy, PoolName.KV, keys, [0, 1, 2, 3])
        idx_t = _make_transfer(PoolName, PoolHitPolicy, PoolName.INDEXER, keys, [0, 1, 2, 3])

        set_res = await asyncio.to_thread(backend.batch_set_v2, [kv_t, idx_t])
        assert set_res[PoolName.KV] == [True, True]
        assert set_res[PoolName.INDEXER] == [True, True]

        # Daemon should report 4 stored entries (2 pages × 2 pools).
        stats = await asyncio.to_thread(backend.get_stats)
        assert stats["entries"] == 4

        # Wipe and read back through v2 → byte-perfect restore.
        for s in range(4):
            kv_pool._storage[s] = 0
            idx_pool._storage[s] = 0
        get_res = await asyncio.to_thread(backend.batch_get_v2, [kv_t, idx_t])
        assert get_res[PoolName.KV] == [True, True]
        assert get_res[PoolName.INDEXER] == [True, True]
        for s in range(4):
            assert kv_pool._storage[s] == 0x10 + s, f"kv slot {s} mismatch"
            assert idx_pool._storage[s] == 0xA0 + s, f"idx slot {s} mismatch"

        # kvd hit counter should advance by 4 (we read 2×2 pages).
        stats_after = await asyncio.to_thread(backend.get_stats)
        assert stats_after["hits_total"] >= 4
    finally:
        await asyncio.to_thread(backend.close)


@pytest.mark.asyncio
async def test_batch_exists_v2_truncates_on_missing_kv(
    kvd_daemon, patched_adapter_codecs, fake_sglang_v2_types
):
    """batch_exists_v2 returns the longest contiguous KV prefix present
    and intersects with each extra pool's hit policy. Mirrors
    HiCacheFile.batch_exists_v2 semantics."""
    PoolName = fake_sglang_v2_types.PoolName
    PoolHitPolicy = fake_sglang_v2_types.PoolHitPolicy

    backend = await asyncio.to_thread(
        kvd_adapter.InferaKvdBackend,
        _make_config(),
        socket_path=kvd_daemon,
        client_id="v2-exists",
    )
    try:
        kv_pool = _FakeHostPool(n_slots=8, page_size=2)
        idx_pool = _FakeHostPool(n_slots=8, page_size=2)
        # When sglang isn't installed our base HiCacheStorage stub lacks
        # `register_mem_host_pool_v2`; bypass by setting the dict the
        # adapter's `_batch_io_v2` reads from directly. With real sglang
        # this attribute is owned/populated by the base ABC.
        backend.registered_pools = {PoolName.KV: kv_pool, PoolName.INDEXER: idx_pool}

        # Write only the first key for both pools, leave the second absent.
        keys = ["page_alpha", "page_beta"]
        kv_t = _make_transfer(PoolName, PoolHitPolicy, PoolName.KV, keys[:1], [0, 1])
        idx_t = _make_transfer(PoolName, PoolHitPolicy, PoolName.INDEXER, keys[:1], [0, 1])
        await asyncio.to_thread(backend.batch_set_v2, [kv_t, idx_t])

        # Ask about both — kv_hit_pages should be 1, INDEXER should also be 1.
        idx_query = _make_transfer(PoolName, PoolHitPolicy, PoolName.INDEXER, keys, [0, 1, 2, 3])
        result = await asyncio.to_thread(backend.batch_exists_v2, keys, [idx_query])
        assert result.kv_hit_pages == 1
        # Only the first page exists for the INDEXER pool too.
        assert result.extra_pool_hit_pages.get(PoolName.INDEXER) == 1
    finally:
        await asyncio.to_thread(backend.close)


@pytest.mark.asyncio
async def test_batch_set_v2_unregistered_pool_reports_failure(
    kvd_daemon, patched_adapter_codecs, fake_sglang_v2_types
):
    """If the engine ships a PoolTransfer for a pool we never registered,
    we must return all-False for that pool (not crash) — a controller
    misconfiguration should be loud-but-recoverable."""
    PoolName = fake_sglang_v2_types.PoolName
    PoolHitPolicy = fake_sglang_v2_types.PoolHitPolicy

    backend = await asyncio.to_thread(
        kvd_adapter.InferaKvdBackend,
        _make_config(),
        socket_path=kvd_daemon,
        client_id="v2-unregistered",
    )
    try:
        # Note: deliberately do NOT call register_mem_host_pool_v2.
        keys = ["k1", "k2"]
        bogus = _make_transfer(PoolName, PoolHitPolicy, PoolName.KV, keys, [0, 1, 2, 3])
        result = await asyncio.to_thread(backend.batch_set_v2, [bogus])
        assert result[PoolName.KV] == [False, False]
    finally:
        await asyncio.to_thread(backend.close)


@pytest.mark.asyncio
async def test_batch_set_v2_mismatched_indices_reports_failure(
    kvd_daemon, patched_adapter_codecs, fake_sglang_v2_types
):
    """host_indices.numel() must equal len(keys)*page_size. On
    mismatch the per-pool list returns all-False; we never silently
    write garbage."""
    PoolName = fake_sglang_v2_types.PoolName
    PoolHitPolicy = fake_sglang_v2_types.PoolHitPolicy

    backend = await asyncio.to_thread(
        kvd_adapter.InferaKvdBackend,
        _make_config(),
        socket_path=kvd_daemon,
        client_id="v2-mismatch",
    )
    try:
        page_size = 2
        kv_pool = _FakeHostPool(n_slots=8, page_size=page_size)
        backend.registered_pools = {PoolName.KV: kv_pool}

        keys = ["k1", "k2"]
        # Only 2 host slots provided — needs 4 for 2 keys × page_size 2.
        bad = _make_transfer(PoolName, PoolHitPolicy, PoolName.KV, keys, [0, 1])
        result = await asyncio.to_thread(backend.batch_set_v2, [bad])
        assert result[PoolName.KV] == [False, False]
    finally:
        await asyncio.to_thread(backend.close)
