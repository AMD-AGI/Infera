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
Also writes lru_instr.py: v0.34.0's lru.py with timers around its two O(n) rebuilds (a log line per rebuild >= 50 ms,
cumulative count / seconds / max every 60 s), to measure on a baseline side what #69540 removes.
  make_kvoffload_overlay.py SITE_PACKAGES VENDOR_DIR OUT_DIR   -> OUT_DIR/{lru.py,offloading_scheduler.py,lru_instr.py}
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


INSTR = [
    ("from typing_extensions import override\n",
     "from typing_extensions import override\n"
     "\n"
     "import time as _vpd_time\n"
     "\n"
     "from vllm.logger import init_logger as _vpd_init_logger\n"
     "\n"
     "_vpd_logger = _vpd_init_logger(__name__)\n"
     "_vpd_stats = [0, 0.0, 0.0, 0.0]  # rebuilds, total s, max s, last summary time\n"
     "\n"
     "\n"
     "def _vpd_record(kind: str, n: int, dt: float) -> None:\n"
     "    st = _vpd_stats\n"
     "    st[0] += 1\n"
     "    st[1] += dt\n"
     "    st[2] = max(st[2], dt)\n"
     "    if dt >= 0.05:\n"
     "        _vpd_logger.info(\"[vpd-lru] %s rebuild of %d entries took %.3f s\", kind, n, dt)\n"
     "    now = _vpd_time.monotonic()\n"
     "    if now - st[3] >= 60:\n"
     "        _vpd_logger.info(\n"
     "            \"[vpd-lru] rebuilds %d, total %.2f s, max %.3f s\", st[0], st[1], st[2]\n"
     "        )\n"
     "        st[3] = now\n"),
    ("            return\n"
     "        self._heap = [(self._ranks[key], key) for key in self._evictable]\n"
     "        heapq.heapify(self._heap)\n",
     "            return\n"
     "        t0 = _vpd_time.perf_counter()\n"
     "        self._heap = [(self._ranks[key], key) for key in self._evictable]\n"
     "        heapq.heapify(self._heap)\n"
     "        _vpd_record(\"compact\", len(self._heap), _vpd_time.perf_counter() - t0)\n"),
    ("        if len(updated_evictable) >= max(64, len(self._evictable) // 4):\n"
     "            self._heap = [(self._ranks[key], key) for key in self._evictable]\n"
     "            heapq.heapify(self._heap)\n",
     "        if len(updated_evictable) >= max(64, len(self._evictable) // 4):\n"
     "            t0 = _vpd_time.perf_counter()\n"
     "            self._heap = [(self._ranks[key], key) for key in self._evictable]\n"
     "            heapq.heapify(self._heap)\n"
     "            _vpd_record(\"bulk\", len(self._heap), _vpd_time.perf_counter() - t0)\n"),
]


def main():
    sp, vendor, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    lru = os.path.join(sp, LRU_REL)
    digest = hashlib.sha256(open(lru, "rb").read()).hexdigest()
    if digest != LRU_V0340_SHA256:
        sys.exit(f"{lru} is not v0.34.0's (sha256 {digest}); the vendored #69540 lru.py may not fit")
    shutil.copyfile(os.path.join(vendor, LRU_VENDORED), os.path.join(out, "lru.py"))
    print(f"wrote {out}/lru.py (#69540, vendored {LRU_VENDORED})")
    s = open(lru).read()
    for old, new in INSTR:
        if s.count(old) != 1:
            sys.exit(f"expected exactly one {old.strip()[:60]!r} in {lru}")
        s = s.replace(old, new)
    open(os.path.join(out, "lru_instr.py"), "w").write(s)
    print(f"wrote {out}/lru_instr.py (v0.34.0 lru.py + rebuild timers)")
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
