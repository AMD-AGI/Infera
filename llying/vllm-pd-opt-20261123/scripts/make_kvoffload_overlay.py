#!/usr/bin/env python3
"""Build the CPU-tier scheduler-thread overlay (prefill only) from the image's own files: two open vLLM PRs that only cut
CPU time on the prefill's scheduler thread, eviction / hit accounting unchanged.
  #69540 (head f1793481): LRUCachePolicy without the O(#evictable) heap rebuilds (_maybe_compact_heap, bulk heapify),
         which block the scheduler thread for 0.5-2 s with a ~1.7M-chunk tier (ours is 1.9 TB). The PR's lru.py is a
         drop-in for v0.34.0 (lru.py is unchanged from v0.34.0 to the PR's base; policies/base.py differs only in
         docstrings), vendored in overlay-kvoffload/; the image's lru.py is checked to be v0.34.0's.
  #68845: get_external_cache_hit_sources merges adjacent same-tier load ranges before the endpoint sweep.
(#65519, bounding CPU->GPU load descriptors per step, is not included: its 8192 default would spread one of our
~48k-descriptor, 71 ms loads over about six engine steps.)
  make_kvoffload_overlay.py SITE_PACKAGES VENDOR_DIR OUT_DIR   -> OUT_DIR/{lru.py,offloading_scheduler.py}
"""
import hashlib
import os
import shutil
import sys

LRU_REL = "vllm/v1/kv_offload/cpu/policies/lru.py"
LRU_V0340_SHA256 = "e9da7f44785969de105dcb0a2a88d382db572e7b53b612d58e8c64b05ceb8d9e"
LRU_VENDORED = "lru-pr69540-f1793481.py"
SCHED_REL = "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"

COALESCE = (
    "        for lo, hi, key in req_status.load_key_ranges:\n"
    "            lo, hi = max(start, lo), min(end, hi)\n"
    "            if lo >= hi:\n"
    "                continue\n"
    "            source = CacheHitSource(\n"
    "                self.manager.get_load_source(key, req_status.req_context)\n"
    "            )\n"
    "            events.extend(((lo, 1, source), (hi, -1, source)))\n",
    "        # Adjacent chunks from the same tier need only one pair of endpoints.\n"
    "        # Overlapping ranges still go through the sweep below.\n"
    "        run_start = run_end = start\n"
    "        run_source = CacheHitSource.EXTERNAL_UNSPECIFIED\n"
    "        get_load_source = self.manager.get_load_source\n"
    "        for lo, hi, key in req_status.load_key_ranges:\n"
    "            if lo < start:\n"
    "                lo = start\n"
    "            if hi > end:\n"
    "                hi = end\n"
    "            if lo >= hi:\n"
    "                continue\n"
    "            source = get_load_source(key, req_status.req_context)\n"
    "            if lo == run_end and source == run_source:\n"
    "                run_end = hi\n"
    "                continue\n"
    "            if run_start < run_end:\n"
    "                events.extend(((run_start, 1, run_source), (run_end, -1, run_source)))\n"
    "            run_start, run_end, run_source = lo, hi, CacheHitSource(source)\n"
    "        if run_start < run_end:\n"
    "            events.extend(((run_start, 1, run_source), (run_end, -1, run_source)))\n",
)


def main():
    sp, vendor, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    lru = os.path.join(sp, LRU_REL)
    digest = hashlib.sha256(open(lru, "rb").read()).hexdigest()
    if digest != LRU_V0340_SHA256:
        sys.exit(f"{lru} is not v0.34.0's (sha256 {digest}); the vendored #69540 lru.py may not fit")
    shutil.copyfile(os.path.join(vendor, LRU_VENDORED), os.path.join(out, "lru.py"))
    print(f"wrote {out}/lru.py (#69540, vendored {LRU_VENDORED})")
    src = os.path.join(sp, SCHED_REL)
    s = open(src).read()
    old, new = COALESCE
    n = s.count(old)
    if n != 1:
        sys.exit(f"expected exactly one load_key_ranges loop in {src}, found {n}")
    open(os.path.join(out, "offloading_scheduler.py"), "w").write(s.replace(old, new))
    print(f"wrote {out}/offloading_scheduler.py (#68845, from {SCHED_REL})")


if __name__ == "__main__":
    main()
