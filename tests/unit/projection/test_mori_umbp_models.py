###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The MoRI KV-transfer and UMBP KV-offload models.

MoRI. The handoff between a split's pools was charged as the whole request's
KV over a link preset that had no measurement behind it. Two things were
wrong with that on MI355X: the link is one Pollara 400 NIC per GPU, which
MoRI-IO's own RDMA benchmark drives at 47.9 GB/s with a ~120 us floor per
batched transfer, and MoRI streams each layer's KV while the next layer
computes, so all but the last layer have landed by the time prefill retires.

UMBP. A per-node host-DRAM store that SGLang's radix tree offloads to
directly. Its capacity is the node's, and how many distinct tokens that holds
depends on how the worker keys it: pure TP keeps one copy per rank, attention
DP shares one keyspace across the node.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infera.projection.core.projection import training_config as tc
from infera.projection.core.projection.inference_projection import kv_cache
from infera.projection.core.projection.inference_projection import performance as perf_mod
from infera.projection.core.projection.inference_projection.collectives import (
    InferenceCollectiveModel,
)
from infera.projection.core.projection.inference_projection.kv_tiers import (
    TIER_DEFAULTS,
    KvTier,
    describe,
    tiers_for_pool,
)
from infera.projection.core.projection.training_config import DisaggregationConfig

_GiB = 1024.0**3
_LAYERS = 61
_REQ_BYTES = 8.0 * _GiB  # one request's per-rank KV


# -- MoRI ---------------------------------------------------------------------


def test_mori_presets_are_the_measured_rdma_link():
    assert tc._TRANSFER_BACKEND_PRESETS["mori"] == (47.9, 120.0)
    assert tc._TRANSFER_BACKEND_PRESETS["moriio"] == tc._TRANSFER_BACKEND_PRESETS["mori"]


def test_layerwise_is_a_property_of_the_engine():
    for backend, layerwise in (("mori", True), ("moriio", True), ("nixl", False),
                               ("mooncake", False), (None, False)):
        d = DisaggregationConfig(enabled=True, transfer_backend=backend)
        assert d.resolved_kv_transfer_layerwise() is layerwise


def test_an_explicit_link_overrides_the_bandwidth_but_not_the_streaming():
    d = DisaggregationConfig(enabled=True, transfer_backend="moriio", kv_transfer_bw_gbps=100.0)
    assert d.resolved_kv_transfer_bw_gbps() == 100.0
    assert d.resolved_kv_transfer_latency_us() == 120.0
    assert d.resolved_kv_transfer_layerwise()


class _Comm:
    """The real bytes/bw + floor arithmetic, without a collective model to build."""

    def kv_transfer_ms(self, kv_bytes, *, bw_gbps=None, latency_us=0.0):
        stub = SimpleNamespace(_args=SimpleNamespace(pod_bw=50.0))
        return InferenceCollectiveModel.kv_transfer_ms(
            stub, kv_bytes, bw_gbps=bw_gbps, latency_us=latency_us
        )


@pytest.fixture
def handoff(monkeypatch):
    """``_kv_transfer_ms`` for one request of ``_REQ_BYTES`` over ``_LAYERS``."""
    monkeypatch.setattr(perf_mod, "_layers_on_rank", lambda cfg: _LAYERS)
    monkeypatch.setattr(
        kv_cache,
        "estimate_kv_cache",
        lambda cfg, layers, concurrency, context_len: SimpleNamespace(bytes_total=_REQ_BYTES),
    )

    def run(**disagg):
        proj = SimpleNamespace(
            cfg=SimpleNamespace(disaggregation_config=DisaggregationConfig(enabled=True, **disagg)),
            _comm=_Comm(),
        )
        decode = SimpleNamespace(cfg=SimpleNamespace(), _comm=None)
        return perf_mod.InferencePerformanceProjector._kv_transfer_ms(proj, decode, 1, 100_000)

    return run


def test_a_layerwise_handoff_is_the_last_layer_plus_the_floor(handoff):
    want = (_REQ_BYTES / _LAYERS) / _GiB / 47.9 * 1e3 + 120.0 / 1e3
    assert handoff(transfer_backend="mori") == pytest.approx(want)
    assert handoff(transfer_backend="moriio") == pytest.approx(want)


def test_layerwise_beats_a_whole_request_engine_on_the_same_link(handoff):
    link = dict(kv_transfer_bw_gbps=47.9, kv_transfer_latency_us=120.0)
    whole = handoff(transfer_backend="nixl", **link)
    assert whole == pytest.approx(_REQ_BYTES / _GiB / 47.9 * 1e3 + 0.12)
    assert handoff(transfer_backend="mori", **link) < whole / 10


def test_an_explicit_bandwidth_still_prices_the_layerwise_handoff(handoff):
    got = handoff(transfer_backend="mori", kv_transfer_bw_gbps=100.0)
    assert got == pytest.approx((_REQ_BYTES / _LAYERS) / _GiB / 100.0 * 1e3 + 0.12)


# -- UMBP ---------------------------------------------------------------------


def test_umbp_parses_as_a_node_total_on_the_host_link():
    t = KvTier.parse("umbp:1500")
    assert t.per_node and t.capacity_gb == 1500.0
    assert (t.bw_gbps, t.latency_us) == TIER_DEFAULTS["dram"]
    t = KvTier.parse("umbp:1200:50:3")
    assert (t.capacity_gb, t.bw_gbps, t.latency_us) == (1200.0, 50.0, 3.0)
    assert not KvTier.parse("dram:150").per_node


def _pool(tp: int, dp: int = 1, pp: int = 1):
    return SimpleNamespace(
        model_parallel_config=SimpleNamespace(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            attention_data_parallel_size=dp,
        )
    )


def test_pure_tp_splits_the_node_over_its_ranks():
    """Each TP rank keeps its own copy, so the node holds node / (TP * bytes) tokens."""
    (t,) = tiers_for_pool([KvTier.parse("umbp:1200")], _pool(tp=4))
    assert t.capacity_gb == 300.0 and not t.per_node and t.node_gb == 1200.0
    bpt, bs = 35136.0, 256
    assert t.capacity_blocks(bpt, bs) == int(1200 * _GiB / 4 // (bpt * bs))


def test_attention_dp_shares_one_keyspace_across_the_node():
    """Every token is stored once: node / bytes distinct tokens."""
    (t,) = tiers_for_pool([KvTier.parse("umbp:1200")], _pool(tp=8, dp=8))
    assert t.capacity_gb == 1200.0
    bpt, bs = 35136.0, 256
    assert t.capacity_blocks(bpt, bs) == int(1200 * _GiB // (bpt * bs))
    (half,) = tiers_for_pool([KvTier.parse("umbp:1200")], _pool(tp=8, dp=4))
    assert half.capacity_gb == 600.0


def test_a_per_gpu_tier_is_left_alone():
    t = KvTier.parse("dram:150")
    assert tiers_for_pool([t], _pool(tp=8, dp=8)) == [t]


def test_describe_says_the_capacity_is_a_node_total():
    (t,) = tiers_for_pool([KvTier.parse("umbp:1200")], _pool(tp=4))
    (line,) = describe([t], 35136.0, 256)
    assert "1200 GB node total" in line and "300 GB/GPU-equivalent" in line
    (raw,) = describe([KvTier.parse("umbp:1200")], 35136.0, 256)
    assert "GB/node" in raw
