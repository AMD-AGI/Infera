#!/usr/bin/env python3
"""Cost of the offload copy paths for KV-like descriptors: N blocks of SIZE bytes scattered over a GPU buffer, copied
to / from pinned host memory on a stream of the buffer's device.
  swap_perf_test.py GPU BUF_DEV METHOD DIRECTION [N] [SIZE] [CONTIG]
METHOD: batch | percopy. CONTIG=1 places consecutive descriptors back to back on both sides (coalescable).
Prints submit (CPU) time and completion time per round, best of 5.
"""
import ctypes
import os
import random
import sys
import time

import torch
from vllm import _custom_ops as ops

gpu, buf_dev, method, direction = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 5120
size = int(sys.argv[6]) if len(sys.argv) > 6 else 42 * 1024
contig = len(sys.argv) > 7 and sys.argv[7] == "1"
torch.cuda.set_device(buf_dev)
dev = torch.device("cuda", buf_dev)
nblk = 4 * n
if gpu == "gem":
    alloc = torch.cuda.memory.CUDAPluggableAllocator(os.environ["INFERA_AMD_GEM_ALLOCATOR"], "hip_gem_alloc", "hip_gem_free")
    pool = torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        g = torch.empty(nblk * size, dtype=torch.uint8, device=dev)
else:
    g = torch.empty(nblk * size, dtype=torch.uint8, device=dev)
g.fill_(7)
h = torch.zeros(n * size, dtype=torch.uint8, pin_memory=True)
blocks = list(range(n)) if contig else random.Random(0).sample(range(nblk), n)
gp = [g.data_ptr() + b * size for b in blocks]
hp = [h.data_ptr() + i * size for i in range(n)]
src, dst = (gp, hp) if direction == "d2h" else (hp, gp)
sp = torch.tensor(src, dtype=torch.int64)
dp = torch.tensor(dst, dtype=torch.int64)
sz = torch.full((n,), size, dtype=torch.int64)
hip = ctypes.CDLL("libamdhip64.so")
hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
s = torch.cuda.Stream()
best = None
for _ in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(s):
        if method == "batch":
            ops.swap_blocks_batch(sp, dp, sz, False)
        else:
            for a, b in zip(src, dst):
                hip.hipMemcpyAsync(b, a, size, 4, s.cuda_stream)
    t1 = time.perf_counter()
    s.synchronize()
    t2 = time.perf_counter()
    r = (t1 - t0, t2 - t0)
    best = r if best is None or r[1] < best[1] else best
gb = n * size / 1e9
print(f"{gpu} cuda:{buf_dev} {method} {direction} n={n} size={size} contig={int(contig)}: submit {best[0] * 1e3:.2f} ms, "
      f"done {best[1] * 1e3:.2f} ms, {gb / best[1]:.1f} GB/s", flush=True)
