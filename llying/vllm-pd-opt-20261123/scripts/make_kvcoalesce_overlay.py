#!/usr/bin/env python3
"""Build the CPU-tier copy-coalescing overlay (prefill) from the image's own gpu_worker.py (checked to be v0.34.0's).

Why: in a C152 prefill trace (a54, 6.4 s per PP stage) the main thread sat in `_C_cache_ops::swap_blocks_batch`
(hipMemcpyBatchAsync, host cost about linear in the number of copy descriptors) for 0.29-0.40 s per stage with the GPU
idle for 4.1-5.7 % of the window; a few loads / stores of long prefixes dominate (one call took 214 ms). The prefill is
the throughput ceiling at C152, so this idle is lost throughput.

What: SingleDirectionOffloadingHandler.transfer_async builds one descriptor per (block, layer ref). Where consecutive
descriptors are adjacent on both sides (src[i] + size[i] == src[i+1] and dst[i] + size[i] == dst[i+1]) they are merged
into one copy of the summed size before submission. The bytes copied, the stream, the ordering and the events are
unchanged; only the number of descriptors handed to the driver drops.

Switches (env, read once at import):
  VPD_KVOFF_COALESCE=1  merge adjacent descriptors (default 0 = v0.34.0 behaviour)
  VPD_KVOFF_STATS=1     every 60 s log per direction: transfers, descriptors before / after merging (counted even when
                        merging is off), host seconds in the swap call, the largest single call
  make_kvcoalesce_overlay.py SITE_PACKAGES OUT_DIR   -> OUT_DIR/gpu_worker.py
"""
import hashlib
import os
import sys

REL = "vllm/v1/kv_offload/cpu/gpu_worker.py"
V0340_SHA256 = "4791ba53822fa9640a1ff3e2c7abf4e840b6a77c2461b6c76141596c368cad94"

HELPERS = (
    "logger = init_logger(__name__)\n",
    "logger = init_logger(__name__)\n"
    "\n"
    "# vpd overlay (make_kvcoalesce_overlay.py): merge adjacent copy descriptors, and count them\n"
    "import os as _vpd_os\n"
    "\n"
    "_VPD_COALESCE = _vpd_os.environ.get(\"VPD_KVOFF_COALESCE\", \"0\") == \"1\"\n"
    "_VPD_STATS = _vpd_os.environ.get(\"VPD_KVOFF_STATS\", \"0\") == \"1\"\n"
    "_vpd_stats = {}\n"
    "_vpd_last_log = [time.monotonic()]\n"
    "\n"
    "\n"
    "def vpd_coalesce(all_src: np.ndarray, all_dst: np.ndarray, all_sizes: np.ndarray, n: int) -> int:\n"
    "    \"\"\"Merge runs of descriptors that are adjacent on both sides, in place in the first n entries.\n"
    "\n"
    "    Returns the new count. Entry order is kept; a merged entry starts at the run's first source / destination\n"
    "    and its size is the run's total, so the same bytes are copied to the same places.\"\"\"\n"
    "    if n < 2:\n"
    "        return n\n"
    "    s, d, z = all_src[:n], all_dst[:n], all_sizes[:n]\n"
    "    cont = (s[1:] == s[:-1] + z[:-1]) & (d[1:] == d[:-1] + z[:-1])\n"
    "    if not cont.any():\n"
    "        return n\n"
    "    starts = np.flatnonzero(np.concatenate(([True], ~cont)))\n"
    "    m = len(starts)\n"
    "    new_z = np.add.reduceat(z, starts)\n"
    "    new_s = s[starts]\n"
    "    new_d = d[starts]\n"
    "    s[:m] = new_s\n"
    "    d[:m] = new_d\n"
    "    z[:m] = new_z\n"
    "    return m\n"
    "\n"
    "\n"
    "def vpd_count_runs(all_src: np.ndarray, all_dst: np.ndarray, all_sizes: np.ndarray, n: int) -> int:\n"
    "    if n < 2:\n"
    "        return n\n"
    "    s, d, z = all_src[:n], all_dst[:n], all_sizes[:n]\n"
    "    cont = (s[1:] == s[:-1] + z[:-1]) & (d[1:] == d[:-1] + z[:-1])\n"
    "    return n - int(np.count_nonzero(cont))\n"
    "\n"
    "\n"
    "def _vpd_record(direction: str, n_before: int, n_after: int, host_s: float) -> None:\n"
    "    st = _vpd_stats.setdefault(direction, [0, 0, 0, 0.0, 0.0])\n"
    "    st[0] += 1\n"
    "    st[1] += n_before\n"
    "    st[2] += n_after\n"
    "    st[3] += host_s\n"
    "    st[4] = max(st[4], host_s)\n"
    "    now = time.monotonic()\n"
    "    if now - _vpd_last_log[0] >= 60.0:\n"
    "        _vpd_last_log[0] = now\n"
    "        for k, v in _vpd_stats.items():\n"
    "            logger.info(\n"
    "                \"vpd_kvoff %s: transfers %d, descriptors %d -> %d (coalesce %s), \"\n"
    "                \"host in swap %.2f s, max call %.1f ms\",\n"
    "                k, v[0], v[1], v[2], \"on\" if _VPD_COALESCE else \"off\", v[3], v[4] * 1e3)\n"
    "            v[:] = [0, 0, 0, 0.0, 0.0]\n",
)

MERGE = (
    "        assert op_idx <= num_copy_ops\n"
    "        src = src[:op_idx]\n",
    "        assert op_idx <= num_copy_ops\n"
    "        _vpd_n_before = op_idx\n"
    "        if _VPD_COALESCE:\n"
    "            op_idx = vpd_coalesce(all_src, all_dst, all_sizes, op_idx)\n"
    "        _vpd_n_after = (\n"
    "            op_idx\n"
    "            if _VPD_COALESCE or not _VPD_STATS\n"
    "            else vpd_count_runs(all_src, all_dst, all_sizes, op_idx)\n"
    "        )\n"
    "        src = src[:op_idx]\n",
)

TIMED = (
    "                if op_idx > 0:\n"
    "                    self._swap_blocks_batch(\n"
    "                        src,\n"
    "                        dst,\n"
    "                        sizes,\n"
    "                        is_src_access_order_any=is_src_access_order_any,\n"
    "                    )\n",
    "                if op_idx > 0:\n"
    "                    _vpd_t0 = time.perf_counter()\n"
    "                    self._swap_blocks_batch(\n"
    "                        src,\n"
    "                        dst,\n"
    "                        sizes,\n"
    "                        is_src_access_order_any=is_src_access_order_any,\n"
    "                    )\n"
    "                    if _VPD_STATS:\n"
    "                        _vpd_record(\n"
    "                            \"store\" if self.gpu_to_cpu else \"load\",\n"
    "                            _vpd_n_before,\n"
    "                            _vpd_n_after,\n"
    "                            time.perf_counter() - _vpd_t0,\n"
    "                        )\n",
)


def main():
    sp, out = sys.argv[1], sys.argv[2]
    path = os.path.join(sp, REL)
    text = open(path).read()
    got = hashlib.sha256(text.encode()).hexdigest()
    assert got == V0340_SHA256, f"{path} is not v0.34.0's gpu_worker.py ({got})"
    for old, new in (HELPERS, MERGE, TIMED):
        assert text.count(old) == 1, f"anchor not found exactly once: {old[:60]!r}"
        text = text.replace(old, new)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "gpu_worker.py"), "w") as f:
        f.write(text)
    print("wrote", os.path.join(out, "gpu_worker.py"))


if __name__ == "__main__":
    main()
