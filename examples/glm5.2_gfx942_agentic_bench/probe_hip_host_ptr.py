#!/usr/bin/env python3
"""Does a host VA registered with hipHostRegister equal its device pointer?

SGLang's hicache host pools put raw host data_ptr()s into a device-side pointer
table that a GPU kernel dereferences, so "no" means the kernel faults. Run on
the target arch; prints host VA vs hipHostGetDevicePointer for each allocation
strategy sglang can take.
"""

import ctypes

import torch

hip = ctypes.CDLL("libamdhip64.so")
hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
hip.hipHostGetDevicePointer.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
    ctypes.c_uint,
]

def dev_ptr(host_ptr: int) -> int:
    out = ctypes.c_void_p()
    rc = hip.hipHostGetDevicePointer(ctypes.byref(out), ctypes.c_void_p(host_ptr), 0)
    return out.value or 0 if rc == 0 else -rc


def report(label: str, t: torch.Tensor) -> bool:
    h = t.data_ptr()
    d = dev_ptr(h)
    print(f"  {label:<36} host=0x{h:x}  devPtr=0x{d:x}  same={h == d}")
    return h == d


torch.zeros(1, device="cuda")  # force a HIP context
print(f"arch={torch.cuda.get_device_properties(0).gcnArchName}  hip={torch.version.hip}")

# What sglang does today on ROCm: anonymous mmap + hipHostRegister.
from sglang.srt.mem_cache.mmap_allocator import alloc_mmap  # noqa: E402

keep, bad = [], 0
for n in (8 << 20, 512 << 20, 4 << 30):
    print(f"size={n >> 20} MiB")
    for flags, name in ((0, "mmap + hipHostRegister"), (0x8, "  + hipHostRegisterMapped")):
        buf = alloc_mmap((n,), torch.uint8)
        assert hip.hipHostRegister(ctypes.c_void_p(buf.data_ptr()), n, flags) == 0
        keep.append(buf)
        bad += not report(name, buf)
    # What the patch switches to, and what "npu"/"musa" already use.
    bad += not report("torch pin_memory (hipHostMalloc)", torch.empty(n, dtype=torch.uint8, pin_memory=True))

# The failing shape upstream: one registration per layer, ~7 GB in total, which
# is what a GLM-5.2 DSA indexer host pool looks like.
print("78 x 96 MiB registrations (indexer-pool shape)")
n = 96 << 20
for i in range(78):
    buf = alloc_mmap((n,), torch.uint8)
    assert hip.hipHostRegister(ctypes.c_void_p(buf.data_ptr()), n, 0) == 0
    keep.append(buf)
    h = buf.data_ptr()
    if h != dev_ptr(h):
        bad += 1
        print(f"  layer {i}: host=0x{h:x} devPtr=0x{dev_ptr(h):x} DIFFERS")
print(f"  mismatches: {bad}")
print("VERDICT:", "host VA == device VA everywhere" if not bad else f"{bad} MISMATCHES")
