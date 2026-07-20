###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for LMCacheRemoteLongRegion using a mocked binding.

The binding shim (``_LMCacheBinding``) wraps the real LMCache objects
(config / metadata / LocalCPUBackend / connector + async bridge). It was
built + validated against a live Redis (see module docstring). These unit
tests mock the shim so CI runs without lmcache/redis; a real round-trip
test lives in tests/integration/test_l4_lmcache.py.
"""

from __future__ import annotations

import pytest

from infera.kvd.lmcache_long_region import (
    LMCacheRemoteConfig,
    LMCacheRemoteLongRegion,
    _chunk_hash_int,
    _LMCacheBinding,
    _parse_compat_key,
)
from infera.kvd.long_region_proto import (
    REASON_EMPTY_VALUE,
    REASON_NOT_STARTED,
    REASON_OVERSIZE,
    REASON_RPC_FAILED,
    REASON_WRONG_RETENTION,
)

# ----------------------------------------------------------------------
# Mock binding — in-memory dict keyed by the synthetic CacheEngineKey.
# Encodes bytes as themselves (no tensor machinery in the mock).
# ----------------------------------------------------------------------


class _MockBinding(_LMCacheBinding):
    def __init__(self) -> None:
        super().__init__()
        self.kv: dict = {}
        self.raise_on_put = False
        self.raise_on_get = False

    def connect(self, config):
        self._connected = True

    def make_key(self, model_name, world_size, worker_id, chunk_hash):
        return (model_name, world_size, worker_id, chunk_hash)

    def encode(self, payload: bytes):
        return bytes(payload)  # mock MemoryObj == raw bytes

    @staticmethod
    def decode(mo):
        return None if mo is None else bytes(mo)

    def put(self, key, mo):
        if self.raise_on_put:
            raise RuntimeError("simulated put failure")
        self.kv[key] = mo

    def get(self, key):
        if self.raise_on_get:
            raise RuntimeError("simulated get failure")
        return self.kv.get(key)

    def contains(self, key) -> bool:
        return key in self.kv

    def close(self):
        self.kv.clear()


@pytest.fixture
def config():
    return LMCacheRemoteConfig(
        remote_url="redis://127.0.0.1:6379",
        prefix="test-prefix",
        max_value_bytes=8 * 1024 * 1024,
    )


@pytest.fixture
def region(config):
    r = LMCacheRemoteLongRegion(config=config, binding=_MockBinding())
    r.start()
    yield r
    r.shutdown()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def test_parse_compat_key_tp_pp():
    assert _parse_compat_key("tp4-pp2-wabc") == (4, 2)


def test_parse_compat_key_fallback():
    assert _parse_compat_key("opaque") == (1, 0)
    assert _parse_compat_key("") == (1, 0)


def test_chunk_hash_int_stable_and_positive():
    a = _chunk_hash_int(b"\x01\x02\x03")
    b = _chunk_hash_int(b"\x01\x02\x03")
    assert a == b and a >= 0
    assert _chunk_hash_int(b"\x01") != _chunk_hash_int(b"\x02")


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_start_idempotent(region):
    region.start()
    assert region._started


def test_shutdown_idempotent(region):
    region.shutdown()
    region.shutdown()


# ----------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------


def test_put_round_trip(region):
    ok, reason = region.put(
        b"k1",
        b"\xab" * 1024,
        retention="long",
        model="m",
        compat_key="tp1-pp1",
        metadata={},
    )
    assert ok, reason
    assert region.get_bytes(b"k1", model="m", compat_key="tp1-pp1") == b"\xab" * 1024


def test_put_rejects_short_retention(region):
    ok, reason = region.put(b"k", b"x", retention="short", model="m", compat_key="c", metadata={})
    assert not ok and reason == REASON_WRONG_RETENTION


def test_put_rejects_empty(region):
    ok, reason = region.put(b"k", b"", retention="long", model="m", compat_key="c", metadata={})
    assert not ok and reason == REASON_EMPTY_VALUE


def test_put_rejects_oversize(region):
    ok, reason = region.put(
        b"k",
        b"x" * (8 * 1024 * 1024 + 1),
        retention="long",
        model="m",
        compat_key="c",
        metadata={},
    )
    assert not ok and reason == REASON_OVERSIZE


def test_put_before_start(config):
    r = LMCacheRemoteLongRegion(config=config, binding=_MockBinding())
    ok, reason = r.put(b"k", b"x", retention="long", model="m", compat_key="c", metadata={})
    assert not ok and reason == REASON_NOT_STARTED


def test_put_rpc_failure(region):
    region._binding.raise_on_put = True
    ok, reason = region.put(b"k", b"x", retention="long", model="m", compat_key="c", metadata={})
    assert not ok and reason == REASON_RPC_FAILED
    assert region.stats()["put_failures_total"] == 1


# ----------------------------------------------------------------------
# Read
# ----------------------------------------------------------------------


def test_get_miss(region):
    assert region.get_bytes(b"nope", model="m", compat_key="c") is None


def test_get_bytes_batch_order_preserved(region):
    region.put(b"k0", b"v0", retention="long", model="m", compat_key="c", metadata={})
    region.put(b"k2", b"v2", retention="long", model="m", compat_key="c", metadata={})
    out = region.get_bytes_batch([b"k0", b"k1", b"k2"], model="m", compat_key="c")
    assert out == [b"v0", None, b"v2"]


def test_exists_vectorized(region):
    region.put(b"k0", b"v", retention="long", model="m", compat_key="c", metadata={})
    assert region.exists([b"k0", b"k1"], model="m", compat_key="c") == [True, False]


def test_get_entry_synthesizes_shim(region):
    region.put(b"k1", b"abc", retention="long", model="m1", compat_key="tp2-pp1", metadata={})
    entry = region.get_entry(b"k1", model="m1", compat_key="tp2-pp1")
    assert entry is not None and entry.size_bytes == 3


def test_distinct_compat_keys_no_collision(region):
    region.put(b"k", b"A", retention="long", model="m", compat_key="fpA", metadata={})
    region.put(b"k", b"B", retention="long", model="m", compat_key="fpB", metadata={})
    assert region.get_bytes(b"k", model="m", compat_key="fpA") == b"A"
    assert region.get_bytes(b"k", model="m", compat_key="fpB") == b"B"


def test_get_batch_rpc_error_returns_none(region):
    region._binding.raise_on_get = True
    assert region.get_bytes_batch([b"k0", b"k1"], model="m", compat_key="c") == [None, None]


# ----------------------------------------------------------------------
# Maintenance
# ----------------------------------------------------------------------


def test_stats_required_keys(region):
    s = region.stats()
    for k in ("used_bytes", "max_bytes", "entries_count", "backend_name"):
        assert k in s
    assert s["backend_name"].startswith("lmcache_remote:")


def test_clear_is_noop(region):
    assert region.clear() == 0


def test_region_satisfies_protocol(region):
    from infera.kvd.long_region_proto import LongRegionLike

    assert isinstance(region, LongRegionLike)
