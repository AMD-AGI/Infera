#!/usr/bin/env python3
"""Driver-allocated pinned host memory at an exact size: hipHostAlloc (or hipHostMalloc) called directly, wrapped
as a torch tensor, instead of torch's pinned caching allocator (which rounds up to a power of two). Checks the host
memory added, is_pinned(), and a GPU round trip.
  exact_pin_test2.py [GB_PER_TENSOR=20] [N_TENSORS=2]
"""
import ctypes
import sys
import time

import torch

from vllm.utils.system_utils import find_loaded_library

gb = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
size = int(gb * 1e9) // 4096 * 4096


def meminfo():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        m[k] = int(v.split()[0]) // 1024 // 1024
    return {k: m[k] for k in ("MemAvailable", "Shmem", "AnonPages")}


torch.cuda.init()
lib = ctypes.CDLL(find_loaded_library("libamdhip64"))
alloc = None
for name in ("hipHostAlloc", "hipHostMalloc"):
    if hasattr(lib, name):
        alloc = getattr(lib, name)
        print("using", name)
        break
alloc.restype = ctypes.c_int
alloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
free = lib.hipHostFree
free.restype = ctypes.c_int
free.argtypes = [ctypes.c_void_p]
print("start", meminfo(), flush=True)
t0 = time.time()
ptrs, tensors = [], []
for _ in range(n):
    p = ctypes.c_void_p()
    rc = alloc(ctypes.byref(p), size, 0)
    assert rc == 0 and p.value, f"alloc rc={rc}"
    buf = (ctypes.c_byte * size).from_address(p.value)
    tensors.append(torch.frombuffer(buf, dtype=torch.int8))
    ptrs.append(p.value)
print(f"{n} x {size / 1e9:.1f} GB in {time.time() - t0:.1f} s:", meminfo(), flush=True)
print("is_pinned:", [t.is_pinned() for t in tensors])
g = torch.arange(1 << 20, dtype=torch.int32, device="cuda").view(torch.int8)
tensors[0][: g.numel()].copy_(g, non_blocking=True)
torch.cuda.synchronize()
back = tensors[-1][: g.numel()].copy_(tensors[0][: g.numel()]).to("cuda", non_blocking=True)
torch.cuda.synchronize()
print("round trip ok:", bool(torch.equal(back, g)))
del tensors, back
for p in ptrs:
    assert free(p) == 0
print("freed", meminfo())
