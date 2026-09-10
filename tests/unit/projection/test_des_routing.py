###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The fleet model is only useful if it routes the way the deployed router does.

Cache hits in the simulator are *derived* from the routing decision, so the
routing rule sets the prefix-hit rate, the load split, and therefore every TTFT
number the fleet model reports. The serving router scores a continuous trade
between cache overlap and load; scoring overlap first and using load only to
break exact ties is a different policy, and it cannot answer the question the
overlap weight exists to ask -- what does dialling locality against balance cost.
"""

from __future__ import annotations

import random

from infera.projection.core.projection.inference_projection.des import (
    _Req,
    _route_and_warm,
)

BLOCK = 16


def _route(block_lists, weight, num_instances=2):
    """Route a fixed block-sequence stream and report the split + hit rate."""
    reqs = [
        _Req(
            idx=i,
            arrival_ms=float(i),
            prompt_len=len(blocks) * BLOCK,
            output_len=8,
            blocks=list(blocks),
        )
        for i, blocks in enumerate(block_lists)
    ]
    per_inst, summary = _route_and_warm(
        reqs,
        policy="kv",
        num_instances=num_instances,
        block_size=BLOCK,
        cache_blocks=0,
        rng=random.Random(0),
        overlap_weight=weight,
    )
    return [len(x) for x in per_inst], summary["block_hit_rate"]


# A hot prefix every request shares, plus a tail unique to each: both instances
# can hold the prefix, so the router has a real choice between reusing it and
# spreading the work.
SHARED = [[1, 2, 3, 4] + [1000 + i] for i in range(40)]


def test_zero_overlap_weight_routes_purely_by_load():
    """At weight 0 the cache term drops out of the cost entirely.

    This is the end of the dial that says "ignore locality"; if a cache edge can
    still sway the pick here, the weight is not actually scaling the overlap
    term and no value of it means what it claims.
    """
    split, _ = _route(SHARED, weight=0.0)
    assert split == [20, 20]


def test_raising_the_overlap_weight_buys_hits_with_balance():
    """The trade the weight exists to express has to show up in both directions.

    A weight that raised the hit rate without concentrating traffic, or skewed
    it without buying hits, would be measuring something other than locality.
    """
    even_split, cold_hits = _route(SHARED, weight=0.0)
    skewed_split, warm_hits = _route(SHARED, weight=20.0)
    assert warm_hits > cold_hits
    assert max(skewed_split) > max(even_split)


def test_an_instance_keeps_the_prefix_it_has_fully_cached():
    """Load is charged as blocks *missed*, not blocks requested.

    An instance answering a fully-cached prompt does no prefill work, so it must
    accrue no load and keep winning that prompt. Charging it the whole request
    instead would grow its load until the prompt bounced to a cold instance,
    which then has to recompute it -- affinity that decays under its own success
    reports cache hits the real router would not lose.
    """
    repeated = [[7, 8, 9]] * 20
    split, hit_rate = _route(repeated, weight=1.0)
    assert sorted(split) == [0, 20]
    # 19 of 20 requests hit all three blocks; only the first is cold.
    assert hit_rate == 57 / 60
