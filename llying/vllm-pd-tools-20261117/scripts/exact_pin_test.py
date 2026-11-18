#!/usr/bin/env python3
"""Allocate CPU-offload-sized host tensors two ways and compare host memory: torch pinned allocator (what vLLM 0.34
uses on ROCm: torch.zeros(pin_memory=True), rounded up to a power of two) vs an exact anonymous mmap registered with
vLLM's host_register (hipHostRegister). Also checks is_pinned() and a GPU round trip.
  exact_pin_test.py [GB_PER_TENSOR=20] [N_TENSORS=2]
"""
import mmap
import sys
import time

import torch

from vllm.v1.kv_offload.cpu.host_register import host_register, host_unregister

gb = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
size = int(gb * 1e9) // 4096 * 4096


def meminfo():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        m[k] = int(v.split()[0]) // 1024 // 1024
    return {k: m[k] for k in ("MemAvailable", "Shmem", "AnonPages", "Mlocked")}


torch.cuda.init()
print("start", meminfo(), flush=True)
t0 = time.time()
pinned = [torch.zeros((size,), dtype=torch.int8, pin_memory=True) for _ in range(n)]
print(f"torch pinned {n} x {size / 1e9:.1f} GB in {time.time() - t0:.1f} s:", meminfo(), flush=True)
del pinned
torch.cuda.empty_cache()
try:
    torch._C._host_emptyCache()
except Exception:
    pass
print("after free", meminfo(), flush=True)

t0 = time.time()
maps, tensors, ptrs = [], [], []
chunk = 64 * 1024**3
for _ in range(n):
    m = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    t = torch.frombuffer(m, dtype=torch.int8)
    base = t.data_ptr()
    for off in range(0, size, chunk):
        ok = host_register(base + off, min(chunk, size - off))
        assert ok, "host_register failed"
        ptrs.append(base + off)
    maps.append(m)
    tensors.append(t)
print(f"mmap + host_register {n} x {size / 1e9:.1f} GB in {time.time() - t0:.1f} s:", meminfo(), flush=True)
print("is_pinned:", [t.is_pinned() for t in tensors])
g = torch.arange(1 << 20, dtype=torch.int32, device="cuda").view(torch.int8)
tensors[0][: g.numel()].copy_(g, non_blocking=True)
torch.cuda.synchronize()
back = tensors[0][: g.numel()].to("cuda", non_blocking=True)
torch.cuda.synchronize()
print("round trip ok:", bool(torch.equal(back, g)))
for p in ptrs:
    host_unregister(p)
print("done", meminfo())
