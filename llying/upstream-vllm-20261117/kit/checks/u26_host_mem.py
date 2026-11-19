#!/usr/bin/env python3
"""U26 validation, run inside the vLLM image with the PR's files mounted: host memory taken by CPU-offload-sized
pinned tensors through torch's caching host allocator (what v0.34.0 does on ROCm) vs the PR's pinned_host_tensor,
plus is_pinned(), a GPU round trip, and the release of the exact-size memory when the last view goes away.
  u26_host_mem.py [GB_PER_TENSOR=20] [N_TENSORS=2]
Prints one JSON line per phase (GiB from /proc/meminfo) and a final "verdict" line.
"""

import gc
import json
import sys
import time

import torch

from vllm.v1.kv_offload.cpu.host_register import pinned_host_tensor

gb = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
row = 1 << 20
rows = int(gb * 1e9) // row


def meminfo() -> dict[str, float]:
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        m[k] = int(v.split()[0]) / 1048576
    return {k: round(m[k], 2) for k in ("MemAvailable", "Shmem", "AnonPages", "Mlocked")}


def report(phase: str, **kw) -> dict[str, float]:
    mi = meminfo()
    print(json.dumps({"phase": phase, **kw, **mi}), flush=True)
    return mi


torch.cuda.init()
requested_gib = n * rows * row / 2**30
base = report("start", tensors=n, gb_per_tensor=rows * row / 1e9, requested_gib=round(requested_gib, 2))

t0 = time.monotonic()
pinned = [torch.zeros((rows, row), dtype=torch.int8, pin_memory=True) for _ in range(n)]
m = report("torch_pinned", seconds=round(time.monotonic() - t0, 1))
torch_delta = m["Shmem"] - base["Shmem"]
del pinned
gc.collect()
try:
    torch._C._host_emptyCache()
except AttributeError:
    pass
base2 = report("torch_freed")

t0 = time.monotonic()
exact = [pinned_host_tensor(rows, row) for _ in range(n)]
assert all(t is not None for t in exact), "pinned_host_tensor returned None"
m = report("exact_pinned", seconds=round(time.monotonic() - t0, 1), is_pinned=[t.is_pinned() for t in exact])
exact_delta = m["Shmem"] - base2["Shmem"]

g = torch.randint(-128, 127, (row,), dtype=torch.int8, device="cuda")
exact[-1][rows - 1].copy_(g, non_blocking=True)
torch.cuda.synchronize()
back = exact[-1][rows - 1].to("cuda", non_blocking=True)
torch.cuda.synchronize()
round_trip = bool(torch.equal(back, g))
del exact, back
gc.collect()
time.sleep(2)
m = report("exact_freed")
released = base2["Shmem"] - m["Shmem"] > -1.0

verdict = {
    "phase": "verdict",
    "requested_gib": round(requested_gib, 2),
    "torch_shmem_delta_gib": round(torch_delta, 2),
    "exact_shmem_delta_gib": round(exact_delta, 2),
    "round_trip": round_trip,
    "exact_released": released,
    "pass": round_trip and released and abs(exact_delta - requested_gib) < 0.05 * requested_gib + 1,
}
print(json.dumps(verdict), flush=True)
