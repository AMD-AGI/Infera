"""Do the real host pools allocate buffers whose host VA == device pointer?

Rather than test allocators in isolation, resolve the allocator the way each
pool does (ALLOC_MEMORY_FUNCS[torch.device(...)]), allocate through it, and
measure the pointer identity that the device-side pointer tables depend on.

All four pools (dsa, mha, mla, mamba) build torch.uint64 tables of host
data_ptr()s on the device, so all four depend on this property.
"""
import ctypes, torch
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS, HostTensorAllocator

lib = ctypes.CDLL("libamdhip64.so")
lib.hipHostGetDevicePointer.restype = ctypes.c_int

def devptr(p):
    out = ctypes.c_void_p()
    rc = lib.hipHostGetDevicePointer(ctypes.byref(out), ctypes.c_void_p(p), ctypes.c_uint(0))
    return None if rc != 0 else (out.value or 0)

key = torch.device("cuda:0")          # what the pools actually key with
f = ALLOC_MEMORY_FUNCS[key]
print(f"  dispatch[{key}] -> {f.__name__}")

buf = f((16, 4096), dtype=torch.uint8, device="cpu", pin_memory=True,
        allocator=HostTensorAllocator())
h = buf.data_ptr(); d = devptr(h)
print(f"  host=0x{h:x}  devPtr=0x{d:x}  same={h == d}  pinned={buf.is_pinned()}")
print("  RESULT:", "SAFE for the pointer tables" if h == d else "UNSAFE -- kernel will fault")
