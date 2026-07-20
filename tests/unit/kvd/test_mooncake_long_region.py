###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for MooncakeStoreLongRegion using a mocked binding.

The binding shim mirrors the REAL MooncakeDistributedStore API (read from
the installed mooncake package source: setup() + put(str,value)/get(str)/
is_exist/remove/close). These unit tests mock the shim with an
in-memory dict; live-cluster tests are in tests/integration/test_l4_mooncake.py.
"""

from __future__ import annotations

import pytest

from infera.kvd.long_region_proto import (
    REASON_BACKEND_UNAVAILABLE,
    REASON_EMPTY_VALUE,
    REASON_NOT_STARTED,
    REASON_OVERSIZE,
    REASON_RPC_FAILED,
    REASON_WRONG_RETENTION,
)
from infera.kvd.mooncake_long_region import (
    MooncakeStoreConfig,
    MooncakeStoreLongRegion,
    _MooncakeBinding,
    _object_key,
    _verify_rdma_or_raise,
)


class _MockBinding(_MooncakeBinding):
    """In-memory dict matching the real put(str,bytes)->int / get(str)->bytes
    surface. put returns 0 on success (Mooncake convention)."""

    def __init__(self) -> None:
        super().__init__()
        self.kv: dict[str, bytes] = {}
        self.raise_on_put = False
        self.raise_on_get = False
        self.put_ret = 0

    def connect(self, config):
        self._connected = True

    def put(self, key: str, value: bytes) -> int:
        if self.raise_on_put:
            raise RuntimeError("simulated put failure")
        if self.put_ret == 0:
            self.kv[key] = bytes(value)
        return self.put_ret

    def get(self, key: str):
        if self.raise_on_get:
            raise RuntimeError("simulated get failure")
        return self.kv.get(key)

    def is_exist(self, key: str) -> bool:
        return key in self.kv

    def get_batch(self, keys):
        return [self.kv.get(k) for k in keys]

    def is_exist_batch(self, keys):
        return [k in self.kv for k in keys]

    def remove_batch(self, keys):
        return [0 if self.kv.pop(k, None) is not None else 1 for k in keys]

    def remove(self, key: str) -> int:
        return 0 if self.kv.pop(key, None) is not None else 1

    def close(self):
        self.kv.clear()


@pytest.fixture
def config():
    return MooncakeStoreConfig(
        master_address="127.0.0.1:50051",
        cluster_id="test-cluster",
        max_value_bytes=8 * 1024 * 1024,
    )


@pytest.fixture
def region(config):
    r = MooncakeStoreLongRegion(config=config, binding=_MockBinding())
    r.start()
    yield r
    r.shutdown()


# --- object key ---


def test_object_key_stable():
    a = _object_key("c", "m", "ck", b"\x01\x02")
    b = _object_key("c", "m", "ck", b"\x01\x02")
    assert a == b and isinstance(a, str)


def test_object_key_namespaced_by_cluster():
    assert _object_key("c1", "m", "ck", b"\x01") != _object_key("c2", "m", "ck", b"\x01")


# --- lifecycle ---


def test_start_idempotent(region):
    region.start()
    assert region._started


def test_shutdown_idempotent(region):
    region.shutdown()
    region.shutdown()


# --- write ---


def test_put_round_trip(region):
    ok, reason = region.put(
        b"k1", b"\xab" * 1024, retention="long", model="m", compat_key="c", metadata={}
    )
    assert ok, reason
    assert region.get_bytes(b"k1", model="m", compat_key="c") == b"\xab" * 1024


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
    r = MooncakeStoreLongRegion(config=config, binding=_MockBinding())
    ok, reason = r.put(b"k", b"x", retention="long", model="m", compat_key="c", metadata={})
    assert not ok and reason == REASON_NOT_STARTED


def test_put_nonzero_ret_is_backend_unavailable(region):
    region._binding.put_ret = 7  # non-zero = failure in Mooncake convention
    ok, reason = region.put(b"k", b"x", retention="long", model="m", compat_key="c", metadata={})
    assert not ok and reason == REASON_BACKEND_UNAVAILABLE


def test_put_rpc_failure(region):
    region._binding.raise_on_put = True
    ok, reason = region.put(b"k", b"x", retention="long", model="m", compat_key="c", metadata={})
    assert not ok and reason == REASON_RPC_FAILED
    assert region.stats()["put_failures_total"] == 1


# --- read ---


def test_get_miss(region):
    assert region.get_bytes(b"nope", model="m", compat_key="c") is None


def test_get_bytes_batch_order(region):
    region.put(b"k0", b"v0", retention="long", model="m", compat_key="c", metadata={})
    region.put(b"k2", b"v2", retention="long", model="m", compat_key="c", metadata={})
    out = region.get_bytes_batch([b"k0", b"k1", b"k2"], model="m", compat_key="c")
    assert out == [b"v0", None, b"v2"]


def test_exists(region):
    region.put(b"k0", b"v", retention="long", model="m", compat_key="c", metadata={})
    assert region.exists([b"k0", b"k1"], model="m", compat_key="c") == [True, False]


def test_get_entry_shim(region):
    region.put(b"k1", b"abc", retention="long", model="m1", compat_key="ck1", metadata={})
    e = region.get_entry(b"k1", model="m1", compat_key="ck1")
    assert e is not None and e.size_bytes == 3 and e.model == "m1"


# --- maintenance ---


def test_stats_required_keys(region):
    s = region.stats()
    for k in ("used_bytes", "max_bytes", "entries_count", "backend_name"):
        assert k in s
    assert s["backend_name"] == "mooncake_store"


def test_clear_removes_only_tracked_keys(region):
    region.put(b"k0", b"v", retention="long", model="m", compat_key="c", metadata={})
    region.put(b"k1", b"v", retention="long", model="m", compat_key="c", metadata={})
    # A key written by "another tenant" directly in the mock store.
    region._binding.kv["other-tenant:x"] = b"z"
    removed = region.clear()
    assert removed == 2
    assert "other-tenant:x" in region._binding.kv  # untouched


def test_region_satisfies_protocol(region):
    from infera.kvd.long_region_proto import LongRegionLike

    assert isinstance(region, LongRegionLike)


# --- RDMA pre-flight guard (refuse silent TCP fallback) ---


def test_rdma_guard_noop_for_tcp():
    # protocol=tcp must never raise, regardless of device_name.
    _verify_rdma_or_raise(MooncakeStoreConfig(master_address="m", cluster_id="c", protocol="tcp"))


def test_rdma_guard_requires_device_name():
    with pytest.raises(RuntimeError, match="requires device_name"):
        _verify_rdma_or_raise(
            MooncakeStoreConfig(master_address="m", cluster_id="c", protocol="rdma", device_name="")
        )


def test_rdma_guard_rejects_absent_device(tmp_path, monkeypatch):
    # A device that isn't present in /sys/class/infiniband must hard-fail.
    monkeypatch.delenv("INFERA_L4_ALLOW_TCP_FALLBACK", raising=False)
    with pytest.raises(RuntimeError, match="not found"):
        _verify_rdma_or_raise(
            MooncakeStoreConfig(
                master_address="m",
                cluster_id="c",
                protocol="rdma",
                device_name="definitely_not_a_real_nic_0",
            )
        )


def test_rdma_guard_hidden_override(monkeypatch):
    monkeypatch.setenv("INFERA_L4_ALLOW_TCP_FALLBACK", "1")
    # With the override, even a bogus device must not raise.
    _verify_rdma_or_raise(
        MooncakeStoreConfig(
            master_address="m",
            cluster_id="c",
            protocol="rdma",
            device_name="definitely_not_a_real_nic_0",
        )
    )
