#!/usr/bin/env python3
"""Which (GPU memory, host memory) pair makes vLLM's offload copy (ops.swap_blocks_batch -> hipMemcpyBatchAsync on ROCm)
fault. One pair per process, since a GPU memory fault kills it:
  swap_fault_test.py GPU HOST DIRECTION [N] [SIZE]
GPU: normal | gem (torch MemPool over $INFERA_AMD_GEM_ALLOCATOR, needs it in LD_PRELOAD); HOST: pinned (torch
pin_memory) | hostreg (anonymous mmap + vllm host_register, as the CPU offload tier); DIRECTION: d2h | h2d.
"""
import ctypes
import mmap
import os
import sys

import torch
from vllm import _custom_ops as ops
from vllm.v1.kv_offload.cpu.host_register import host_register

gpu, host, direction = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 64
size = int(sys.argv[5]) if len(sys.argv) > 5 else 64 * 1024
# SPAN_GB: GPU buffer size; descriptors are spread evenly over it (large offsets into one BO, as a KV pool).
span = int(float(os.environ.get("SPAN_GB", "0")) * (1 << 30)) or n * size
total = n * size
dev = torch.device("cuda", 0)
if gpu == "gem":
    alloc = torch.cuda.memory.CUDAPluggableAllocator(os.environ["INFERA_AMD_GEM_ALLOCATOR"], "hip_gem_alloc", "hip_gem_free")
    pool = torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        g = torch.empty(span, dtype=torch.uint8, device=dev)
else:
    g = torch.empty(span, dtype=torch.uint8, device=dev)
g.fill_(7)
stride = (span - size) // max(n - 1, 1) // 256 * 256 if span > total else size
if host == "pinned":
    h = torch.zeros(total, dtype=torch.uint8, pin_memory=True)
    haddr, keep = h.data_ptr(), h
else:
    if host == "shm":  # a /dev/shm file mapping, like the offload tier's shared region
        fd = os.open("/dev/shm/swap_fault_test", os.O_CREAT | os.O_RDWR)
        os.ftruncate(fd, total)
        mm = mmap.mmap(fd, total, mmap.MAP_SHARED)
    else:
        mm = mmap.mmap(-1, total)
    buf = (ctypes.c_char * total).from_buffer(mm)
    haddr, keep = ctypes.addressof(buf), (mm, buf)
    assert host_register(haddr, total), "host_register failed"
gaddr = g.data_ptr()
gofs = [gaddr + i * stride for i in range(n)]
hofs = [haddr + i * size for i in range(n)]
sp = torch.tensor(gofs if direction == "d2h" else hofs, dtype=torch.int64)
dp = torch.tensor(hofs if direction == "d2h" else gofs, dtype=torch.int64)
sz = torch.full((n,), size, dtype=torch.int64)
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    ops.swap_blocks_batch(sp, dp, sz, False)
s.synchronize()
if direction == "d2h":
    first = ctypes.string_at(haddr, 16)
    print(f"{gpu} {host} {direction} n={n} size={size}: ok, host bytes {first[:4]!r}", flush=True)
else:
    print(f"{gpu} {host} {direction} n={n} size={size}: ok", flush=True)
