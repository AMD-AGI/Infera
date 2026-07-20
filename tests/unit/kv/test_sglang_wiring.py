###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for the SGLang worker-side KV wiring helpers.

These tests don't require SGLang or HuggingFace to be installed — we
stub the tokenizer-loading path. The wiring helpers are kept small
enough that the real value of the test is checking the orchestration:
publisher binds, snapshot server starts on the configured port, probe
attaches to the cache, stop() tears everything down cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from infera.engine.sglang import kv_wiring
from infera.engine.sglang.kv_wiring import (
    _find_radix_cache,
    build_and_start,
    compute_kv_metadata,
    resolve_advertise_endpoint,
)

# Each test gets its own inproc:// endpoint so ZMQ contexts don't bleed.
_endpoint_counter = 0


def _next_endpoint() -> str:
    global _endpoint_counter
    _endpoint_counter += 1
    return f"inproc://infera-wiring-test-{_endpoint_counter}"


# ----------------------------------------------------------------------
# Stub tokenizer + RadixCache
# ----------------------------------------------------------------------


class _StubTokenizer:
    """Quacks like a HF tokenizer for tokenize_canary's `encode` call."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) % 50_000 for c in text[:32]]


@dataclass
class _MockNode:
    key: list[int]
    parent: Any = None
    children: dict[int, Any] = field(default_factory=dict)


@dataclass
class _MockCache:
    root: _MockNode = field(default_factory=lambda: _MockNode(key=[]))
    inserts: list[list[int]] = field(default_factory=list)

    def insert(self, token_ids):
        self.inserts.append(list(token_ids))
        node = _MockNode(key=list(token_ids), parent=self.root)
        return node

    def evict(self, n: int = 1):
        return []

    def reset(self):
        self.root = _MockNode(key=[])


# ----------------------------------------------------------------------
# resolve_advertise_endpoint
# ----------------------------------------------------------------------


def test_resolve_advertise_replaces_0_0_0_0() -> None:
    assert resolve_advertise_endpoint("tcp://0.0.0.0:5557", "10.0.0.5") == "tcp://10.0.0.5:5557"


def test_resolve_advertise_replaces_wildcard() -> None:
    assert resolve_advertise_endpoint("tcp://*:5557", "host.example") == "tcp://host.example:5557"


def test_resolve_advertise_preserves_explicit_host() -> None:
    assert resolve_advertise_endpoint("tcp://10.0.0.5:5557", "10.99.99.99") == "tcp://10.0.0.5:5557"


def test_resolve_advertise_handles_missing_scheme() -> None:
    assert resolve_advertise_endpoint("0.0.0.0:5557", "host") == "tcp://host:5557"


# ----------------------------------------------------------------------
# _find_radix_cache
# ----------------------------------------------------------------------


def test_find_radix_cache_prefers_tree_cache() -> None:
    sentinel = object()

    class Eng:
        tree_cache = sentinel
        radix_cache = object()

    assert _find_radix_cache(Eng()) is sentinel


def test_find_radix_cache_falls_back_to_other_names() -> None:
    sentinel = object()

    class Eng:
        _radix_cache = sentinel

    assert _find_radix_cache(Eng()) is sentinel


def test_find_radix_cache_returns_none_when_missing() -> None:
    class Eng:
        pass

    assert _find_radix_cache(Eng()) is None


# ----------------------------------------------------------------------
# compute_kv_metadata
# ----------------------------------------------------------------------


def test_compute_kv_metadata_populates_endpoints(monkeypatch, tmp_path) -> None:
    # tokenizer.json so _resolve_tokenizer_path returns a file path.
    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text('{"version":"1.0"}')

    monkeypatch.setattr(kv_wiring, "_resolve_tokenizer_path", lambda _p, **_kw: str(tok_file))

    # Stub AutoTokenizer.from_pretrained.
    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_id, *args, **kwargs):
            return _StubTokenizer()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAutoTokenizer)

    md = compute_kv_metadata(
        model_id="test/Qwen3-0.6B",
        engine_block_size=1,
        index_block_size=64,
        events_endpoint="tcp://1.2.3.4:5557",
        snapshot_endpoint="http://1.2.3.4:8801",
    )
    assert md.engine_block_size == 1
    assert md.index_block_size == 64
    assert md.events_endpoint == "tcp://1.2.3.4:5557"
    assert md.snapshot_endpoint == "http://1.2.3.4:8801"
    assert md.tiers == ["device"]
    assert md.supports_events is True
    # Digest is sha256[:16] of the tiny stub tokenizer.json.
    assert len(md.tokenizer_digest) == 16
    # Canary was tokenized by our stub.
    assert md.tokenizer_canary
    assert all(isinstance(t, int) for t in md.tokenizer_canary)


def test_compute_kv_metadata_digest_uses_tokenizer_path_not_alias(monkeypatch) -> None:
    # Bug 1 regression: digest/canary resolve from tokenizer_path, not the
    # served-model-name alias in model_id (a non-local id would hit HF -> 401).
    seen: dict[str, Any] = {}

    def _fake_resolve(arg: str, *, trust_remote_code: bool = False) -> str:
        seen["resolve"] = arg
        seen["resolve_trc"] = trust_remote_code
        return arg

    monkeypatch.setattr(kv_wiring, "_resolve_tokenizer_path", _fake_resolve)
    monkeypatch.setattr(kv_wiring, "compute_tokenizer_digest", lambda _p: "deadbeefdeadbeef")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_id, *args, **kwargs):
            seen["from_pretrained"] = _id
            seen["from_pretrained_trc"] = kwargs.get("trust_remote_code")
            return _StubTokenizer()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAutoTokenizer)

    compute_kv_metadata(
        model_id="Kimi-K2.6",  # served-model-name alias
        tokenizer_path="/models/Kimi-K2.6-MXFP4",  # real local path
        engine_block_size=1,
        index_block_size=64,
        events_endpoint="tcp://1.2.3.4:5557",
        snapshot_endpoint="http://1.2.3.4:8801",
        trust_remote_code=True,
    )
    assert seen["resolve"] == "/models/Kimi-K2.6-MXFP4"
    assert seen["from_pretrained"] == "/models/Kimi-K2.6-MXFP4"
    # trust_remote_code must be threaded to both tokenizer loads (custom
    # tokenizers like Kimi's tiktoken won't load without it).
    assert seen["resolve_trc"] is True
    assert seen["from_pretrained_trc"] is True


# ----------------------------------------------------------------------
# build_and_start
# ----------------------------------------------------------------------


def _free_port() -> int:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.asyncio
async def test_build_and_start_attaches_probe_and_serves_snapshot(monkeypatch, tmp_path) -> None:
    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text('{"v":1}')
    monkeypatch.setattr(kv_wiring, "_resolve_tokenizer_path", lambda _p, **_kw: str(tok_file))

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_id, *args, **kwargs):
            return _StubTokenizer()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAutoTokenizer)

    cache = _MockCache()
    port = _free_port()

    wiring = await build_and_start(
        model_id="test/m",
        engine_block_size=1,
        index_block_size=4,
        publisher_id="w1",
        events_bind=_next_endpoint(),
        events_advertise="tcp://1.1.1.1:5557",
        snapshot_host="127.0.0.1",
        snapshot_port=port,
        snapshot_advertise=f"http://127.0.0.1:{port}",
        radix_cache=cache,
    )
    try:
        # Probe attached: inserting tokens emits stored events on the probe.
        cache.insert(list(range(8)))
        assert wiring.probe.stored_emitted == 2

        # The snapshot HTTP server is up (give uvicorn a tick to bind).
        import asyncio

        for _ in range(50):
            await asyncio.sleep(0.05)
            try:
                async with httpx.AsyncClient(timeout=2.0) as cli:
                    r = await cli.get(
                        f"http://127.0.0.1:{port}/v1/kv-snapshot",
                        params={
                            "publisher_id": "w1",
                            "model": "test/m",
                            "compat_key": wiring.metadata.tokenizer_digest,
                        },
                    )
                    break
            except (httpx.ConnectError, httpx.ReadError):
                continue
        else:
            pytest.fail("snapshot HTTP server never became reachable")
        assert r.status_code == 200
        body = r.json()
        # The snapshot reflects what the probe mirrored.
        assert body["publisher_id"] == "w1"
        assert body["model_name"] == "test/m"
        assert body["compat_key"] == wiring.metadata.tokenizer_digest
    finally:
        await wiring.stop()


@pytest.mark.asyncio
async def test_build_and_start_without_radix_cache(monkeypatch, tmp_path) -> None:
    """No cache passed → wiring still comes up; probe just never fires."""
    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text('{"v":1}')
    monkeypatch.setattr(kv_wiring, "_resolve_tokenizer_path", lambda _p, **_kw: str(tok_file))

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_id, *args, **kwargs):
            return _StubTokenizer()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAutoTokenizer)

    port = _free_port()
    wiring = await build_and_start(
        model_id="test/m",
        engine_block_size=1,
        index_block_size=4,
        publisher_id="w1",
        events_bind=_next_endpoint(),
        events_advertise="tcp://1.1.1.1:5557",
        snapshot_host="127.0.0.1",
        snapshot_port=port,
        snapshot_advertise=f"http://127.0.0.1:{port}",
        radix_cache=None,
    )
    try:
        assert wiring.snapshot_originals == {}
        assert wiring.radix_cache is None
        assert wiring.metadata.events_endpoint == "tcp://1.1.1.1:5557"
        assert wiring.metadata.snapshot_endpoint == f"http://127.0.0.1:{port}"
    finally:
        await wiring.stop()


@pytest.mark.asyncio
async def test_stop_detaches_probe_so_inserts_no_longer_fire(monkeypatch, tmp_path) -> None:
    tok_file = tmp_path / "tokenizer.json"
    tok_file.write_text('{"v":1}')
    monkeypatch.setattr(kv_wiring, "_resolve_tokenizer_path", lambda _p, **_kw: str(tok_file))

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_id, *args, **kwargs):
            return _StubTokenizer()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAutoTokenizer)

    cache = _MockCache()
    port = _free_port()
    wiring = await build_and_start(
        model_id="test/m",
        engine_block_size=1,
        index_block_size=4,
        publisher_id="w1",
        events_bind=_next_endpoint(),
        events_advertise="tcp://1.1.1.1:5557",
        snapshot_host="127.0.0.1",
        snapshot_port=port,
        snapshot_advertise=f"http://127.0.0.1:{port}",
        radix_cache=cache,
    )

    cache.insert(list(range(4)))
    assert wiring.probe.stored_emitted == 1

    await wiring.stop()

    # After detach, the cache.insert wrapper is restored — probe no
    # longer fires on subsequent inserts. (probe.stored_emitted is the
    # probe-side counter; if detach left a wrapper, it'd tick.)
    cache.insert(list(range(4, 8)))
    assert wiring.probe.stored_emitted == 1
