###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""KV storage tiers below HBM: placement, reuse, and what a load costs.

A tier only earns its place if three things hold: with no tier the block cache
is exactly the radix cache it was, a tier keeps reuse that HBM alone would have
thrown away, and the reuse it keeps is not free -- a hit held below HBM waits
for its blocks to cross that tier's link before it can prefill.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from infera.projection.core.projection.inference_projection import des as des_mod
from infera.projection.core.projection.inference_projection.kv_tiers import (
    KvTier,
    resolve_tiers,
)


def _blocks(prefix: int, n_shared: int, uniq: int, n_uniq: int) -> list[int]:
    return [prefix * 1000 + i for i in range(n_shared)] + [
        10**6 + uniq * 1000 + i for i in range(n_uniq)
    ]


class _ReferenceRadix:
    """The single-tier leaf-first cache as it was before tiers, for parity."""

    def __init__(self, cap: int) -> None:
        import heapq

        self._h = heapq
        self.cap = cap
        self.parent: dict[int, int | None] = {}
        self.kids: dict[int, int] = {}
        self.acc: dict[int, int] = {}
        self.leaves: list[tuple[int, int]] = []
        self.tick = 0
        self.evictions = 0

    def match(self, blocks):
        m = 0
        for b in blocks:
            if b not in self.kids:
                break
            m += 1
        return m

    def insert(self, blocks):
        prev = None
        for b in blocks:
            self.tick += 1
            if b in self.kids:
                self.acc[b] = self.tick
                if self.kids[b] == 0:
                    self._h.heappush(self.leaves, (self.tick, b))
            else:
                self.parent[b] = prev
                self.kids[b] = 0
                self.acc[b] = self.tick
                if prev is not None and prev in self.kids:
                    self.kids[prev] += 1
                self._h.heappush(self.leaves, (self.tick, b))
            prev = b
        while self.cap > 0 and len(self.kids) > self.cap:
            v = None
            while self.leaves:
                t, b = self._h.heappop(self.leaves)
                if b in self.kids and self.acc[b] == t and not self.kids[b]:
                    v = b
                    break
            if v is None:
                return
            p = self.parent.pop(v)
            del self.kids[v], self.acc[v]
            self.evictions += 1
            if p is not None and p in self.kids:
                self.kids[p] -= 1
                if self.kids[p] == 0:
                    self._h.heappush(self.leaves, (self.acc[p], p))


@pytest.mark.parametrize("cls", [des_mod._BlockCache, des_mod._TieredBlockCache])
def test_without_tiers_the_cache_is_the_radix_cache_it_was(cls):
    rng = random.Random(7)
    ref = _ReferenceRadix(40)
    new = cls(40)
    for i in range(600):
        blk = _blocks(rng.randrange(6), rng.randrange(1, 12), i, rng.randrange(0, 8))
        assert new.prefix_match(blk) == ref.match(blk)
        assert new.prefix_match_tiers(blk) == [ref.match(blk)]
        new.insert(blk)
        ref.insert(blk)
    assert new.evictions == ref.evictions
    assert len(new) == len(ref.kids)


def test_hbm_victims_move_down_a_tier_tail_first():
    c = des_mod._TieredBlockCache(2, lower_tiers=[2])
    c.insert([1, 2])
    c.insert([3, 4])
    # The older conversation's tail goes first, then its head.
    assert c._tier == {1: 1, 2: 1, 3: 0, 4: 0}
    assert c.prefix_match_tiers([1, 2, 5]) == [0, 2]
    c.insert([1, 2, 5])
    # A hit is loaded back into HBM along with everything before it, and the
    # last tier drops its least-recently-used leaf when it overflows.
    assert c._tier == {1: 0, 2: 0, 3: 1, 5: 1}
    assert c.evictions == 1
    assert c.promotions[1] == 2


def test_a_parent_is_never_slower_than_its_child():
    rng = random.Random(3)
    c = des_mod._TieredBlockCache(30, lower_tiers=[50, 80])
    for i in range(800):
        c.insert(_blocks(rng.randrange(10), rng.randrange(1, 20), i, rng.randrange(0, 10)))
        for b, t in c._tier.items():
            p = c._parent.get(b)
            if p is not None and p in c._tier:
                assert c._tier[p] <= t
    assert [c._count[i] <= cap for i, cap in enumerate(c.capacities)] == [True] * 3


def _route(lower=(), hbm=24, n=400, seed=0):
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        r = des_mod._Req(idx=i, arrival_ms=float(i), prompt_len=16 * 512, output_len=64)
        r.blocks = _blocks(rng.randrange(12), 12, i, 4)
        reqs.append(r)
    _, summary = des_mod._route_and_warm(
        reqs,
        policy="round_robin",
        num_instances=1,
        block_size=512,
        cache_blocks=hbm,
        rng=random.Random(seed),
        lower_tiers=list(lower),
        tier_names=["dram", "ssd"][: len(lower)],
    )
    return reqs, summary


def test_a_tier_keeps_reuse_hbm_alone_drops():
    _, hbm_only = _route()
    reqs, tiered = _route(lower=[200])
    assert tiered["block_hit_rate"] > hbm_only["block_hit_rate"] + 0.2
    t = tiered["tiers"]
    assert set(t) == {"hbm", "dram"}
    assert t["dram"]["hit_blocks_frac"] > 0
    assert any(r.load_blocks and r.load_blocks[0] > 0 for r in reqs)


@dataclass
class _ReqCfg:
    input_seq_len: int = 8192
    output_seq_len: int = 32
    max_concurrency: int = 4
    max_num_batched_tokens: int = 8192
    chunked_prefill_size: int = 0
    speculative_num_tokens: int = 0
    speculative_acceptance_rate: float = 0.0
    tokenize_overhead_us: float = 0.0
    detokenize_overhead_us: float = 0.0

    def resolved_max_concurrency(self) -> int:
        return self.max_concurrency

    def resolved_max_context_len(self) -> int:
        return self.input_seq_len + self.output_seq_len

    def resolved_prefix_cache_hit_rate(self) -> float:
        return 0.0


@dataclass
class _Cfg:
    request_config: _ReqCfg


class _Kernel:
    def decode_step_latency_ms(self, batch, ctx, q_len):
        return 2.0

    def mixed_step_latency_ms(self, num_decode, prefill_tokens, ctx, prefill_kv, q_len, **kw):
        return 2.0 + 0.001 * prefill_tokens


def _sim(block_ms: float, load_blocks=(8,), clients=4, n=16):
    reqs = []
    for i in range(n):
        r = des_mod._Req(idx=i, arrival_ms=0.0, prompt_len=8192, output_len=32)
        r.cached_prefix = r.num_computed = 8 * 512
        r.load_blocks = tuple(load_blocks)
        reqs.append(r)
    return des_mod.simulate_once(
        _Cfg(_ReqCfg(max_concurrency=clients)),
        _Kernel(),
        rate_per_s=0.0,
        arrival_model="closed",
        prebuilt=reqs,
        closed_loop_clients=clients,
        warmup_frac=0.0,
        tier_block_ms=[block_ms] if block_ms else [],
    )


def test_a_hit_below_hbm_waits_for_its_blocks():
    free = _sim(0.0)
    slow = _sim(5.0)
    # Eight 5 ms blocks is 40 ms per request; four clients queue on one link.
    assert slow.ttft_arrival["mean"] > free.ttft_arrival["mean"] + 40.0
    assert slow.packing["tier_load_ms_mean"] >= 40.0
    # The GPU did not wait on the link, so decode is no slower. (It can be
    # faster: staggered admissions put fewer prefills into decode steps.)
    assert slow.tpot["mean"] <= free.tpot["mean"] * 1.01


def test_one_link_serialises_concurrent_loads():
    one = _sim(5.0, clients=1, n=4)
    four = _sim(5.0, clients=4, n=4)
    assert four.packing["tier_load_ms_mean"] > one.packing["tier_load_ms_mean"] * 2


def test_a_prompt_past_the_context_limit_is_reported_not_hidden():
    reqs = [
        des_mod._Req(idx=i, arrival_ms=0.0, prompt_len=20000 if i == 2 else 8192, output_len=32)
        for i in range(8)
    ]
    res = des_mod.simulate_once(
        _Cfg(_ReqCfg(max_concurrency=1)),
        _Kernel(),
        rate_per_s=0.0,
        arrival_model="closed",
        prebuilt=reqs,
        closed_loop_clients=1,
        warmup_frac=0.0,
    )
    # The third request can never finish its prefill, so the run stops there.
    assert res.num_requests == 2
    assert res.packing["stalled_reqs"] == 1
    assert "stalled_reqs" not in _sim(0.0).packing


def test_tier_spec_parsing_and_defaults():
    t = KvTier.parse("ssd:4000")
    assert (t.name, t.capacity_gb, t.bw_gbps, t.latency_us) == ("ssd", 4000.0, 12.0, 80.0)
    t = KvTier.parse("dram:512:100:1")
    assert (t.bw_gbps, t.latency_us) == (100.0, 1.0)
    with pytest.raises(ValueError):
        KvTier.parse("weird:10")
    assert resolve_tiers(None, offload_gb=256, offload_bw=64)[0].name == "dram"
    assert resolve_tiers(["dram:0"]) == []
    # 1 MiB/token, 512-token blocks: 1 GiB holds 2 blocks; a block moves in ~8.4 ms at 64 GB/s.
    t = KvTier("dram", 1.0, 64.0, 0.0)
    assert t.capacity_blocks(1024**2, 512) == 2
    assert t.block_load_ms(1024**2, 512) == pytest.approx(512 * 1024**2 / 64e9 * 1e3)
