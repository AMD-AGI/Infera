#!/usr/bin/env python3
"""vLLM's offload copy (ops.swap_blocks_batch -> hipMemcpyBatchAsync) with the GPU buffer and the copy stream on chosen
devices, one case per process (a GPU memory fault kills it):
  swap_dev_test.py GPU BUF_DEV STREAM_DEV [THREAD] [METHOD] [DIRECTION]
GPU: normal | gem; BUF_DEV: device the KV-like buffer lives on; STREAM_DEV: device of the copy stream (-1: the current
device of the copying thread). THREAD=thread: create the stream and copy on a fresh Python thread that never calls
set_device (its current device is the default, 0), as a connector thread would; main = the main thread.
METHOD: batch (ops.swap_blocks_batch) | percopy (one hipMemcpyAsync per descriptor) | torch (Tensor.copy_ of one
contiguous range). DIRECTION: d2h | h2d.
"""
import ctypes
import os
import sys
import threading

import torch
from vllm import _custom_ops as ops

gpu, buf_dev, stream_dev = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "main"
method = sys.argv[5] if len(sys.argv) > 5 else "batch"
direction = sys.argv[6] if len(sys.argv) > 6 else "d2h"
n, size = 256, 64 * 1024
torch.cuda.set_device(buf_dev)
dev = torch.device("cuda", buf_dev)
if gpu == "gem":
    alloc = torch.cuda.memory.CUDAPluggableAllocator(os.environ["INFERA_AMD_GEM_ALLOCATOR"], "hip_gem_alloc", "hip_gem_free")
    pool = torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        g = torch.empty(n * size * 4, dtype=torch.uint8, device=dev)
else:
    g = torch.empty(n * size * 4, dtype=torch.uint8, device=dev)
g.fill_(7)
torch.cuda.synchronize(dev)
h = torch.full((n * size,), 9, dtype=torch.uint8, pin_memory=True)
gp = [g.data_ptr() + i * size * 4 for i in range(n)]
hp = [h.data_ptr() + i * size for i in range(n)]
src, dst = (gp, hp) if direction == "d2h" else (hp, gp)
sp = torch.tensor(src, dtype=torch.int64)
dp = torch.tensor(dst, dtype=torch.int64)
sz = torch.full((n,), size, dtype=torch.int64)
hip = ctypes.CDLL("libamdhip64.so")
hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]


def copy(tag):
    tag += f" (current device {torch.cuda.current_device()})"
    s = torch.cuda.Stream(device=stream_dev) if stream_dev >= 0 else torch.cuda.Stream()
    with torch.cuda.stream(s):
        if method == "batch":
            ops.swap_blocks_batch(sp, dp, sz, False)
        elif method == "percopy":
            for a, b in zip(src, dst):
                rc = hip.hipMemcpyAsync(b, a, size, 4, s.cuda_stream)
                assert rc == 0, rc
        elif direction == "d2h":
            h[: n * size].copy_(g[: n * size], non_blocking=True)
        else:
            g[: n * size].copy_(h[: n * size], non_blocking=True)
    s.synchronize()
    check = ctypes.string_at(h.data_ptr(), 4) if direction == "d2h" else bytes(g[:4].cpu().tolist())
    print(f"{gpu} buf=cuda:{buf_dev} stream={s.device} {method} {direction} {tag}: ok, bytes {check!r}", flush=True)


if mode == "thread":
    print(f"main current device {torch.cuda.current_device()}", flush=True)
    t = threading.Thread(target=copy, args=("thread",))
    t.start()
    t.join()
else:
    copy("main thread")
