#!/usr/bin/env python3
"""CPU check of the CPU-tier overlay (scripts/make_kvoffload_overlay.py). Run in the vLLM image with the two overlay
files mounted over the image's own:
  test_kvoffload.py VENDOR_DIR
1. #69540's own tests (overlay-kvoffload/test_lru-*.py), including the differential test that replays random
   store / access / pin / unpin / evict / finish sequences against a copy of the replaced heap policy;
2. #68845: get_external_cache_hit_sources of the patched scheduler against v0.34.0's (copied below) on random
   adjacent / overlapping / gapped load ranges from mixed tiers.
"""
import glob
import os
import random
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
from vllm.v1.metrics.cache_hit_source import CacheHitSource


def reference_sources(self, request, num_external_tokens):
    """v0.34.0's OffloadingConnectorScheduler.get_external_cache_hit_sources."""
    sources = {}
    if num_external_tokens == 0:
        return sources
    req_status = self._req_status[request.request_id]
    start = req_status.num_locally_computed_tokens
    end = start + num_external_tokens
    events = [(end, 0, CacheHitSource.EXTERNAL_UNSPECIFIED)]
    for lo, hi, key in req_status.load_key_ranges:
        lo, hi = max(start, lo), min(end, hi)
        if lo >= hi:
            continue
        source = CacheHitSource(self.manager.get_load_source(key, req_status.req_context))
        events.extend(((lo, 1, source), (hi, -1, source)))
    active = Counter()
    position = start
    for boundary, delta, source in sorted(events):
        if boundary > position:
            tiers = [tier for tier, count in active.items() if count > 0]
            tier = CacheHitSource.outermost(tiers) if tiers else CacheHitSource.EXTERNAL_UNSPECIFIED
            sources[tier] = sources.get(tier, 0) + boundary - position
        active[source] += delta
        position = boundary
    return sources


def coalesce_differential(trials=3000):
    rng = random.Random(0)
    tiers = ["host", "disk", "p2p", "external_unspecified"]
    for _ in range(trials):
        ranges, src, pos = [], {}, rng.randrange(0, 256, 64)
        for i in range(rng.randrange(0, 40)):
            if rng.random() < 0.1:
                pos += rng.randrange(1, 200)  # gap
            if rng.random() < 0.1:
                pos = max(0, pos - rng.randrange(1, 200))  # overlap
            size = rng.choice([64, 64, 64, 128, rng.randrange(1, 300)])
            key = f"k{i}"
            src[key] = rng.choice(tiers[:2]) if rng.random() < 0.9 else rng.choice(tiers)
            ranges.append((pos, pos + size, key))
            pos += size
        start = rng.randrange(0, 300)
        status = SimpleNamespace(num_locally_computed_tokens=start, load_key_ranges=ranges, req_context=None)
        stub = SimpleNamespace(_req_status={"r": status},
                               manager=SimpleNamespace(get_load_source=lambda key, ctx: src[key]))
        request = SimpleNamespace(request_id="r")
        n = rng.randrange(0, pos + 100)
        want = reference_sources(stub, request, n)
        got = OffloadingConnectorScheduler.get_external_cache_hit_sources(stub, request, n)
        assert got == want, (start, n, ranges, src, got, want)
    print(f"#68845 coalescing: {trials} random cases equal to v0.34.0")


def main():
    vendor = sys.argv[1]
    tests = glob.glob(os.path.join(vendor, "test_lru-*.py"))
    assert len(tests) == 1, tests
    rc = pytest.main(["-q", "-p", "no:cacheprovider", tests[0]])
    assert rc == 0, f"#69540 tests failed: {rc}"
    coalesce_differential()
    print("kvoffload overlay test passed")


if __name__ == "__main__":
    main()
