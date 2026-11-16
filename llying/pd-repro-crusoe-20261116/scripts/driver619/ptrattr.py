#!/usr/bin/env python3
"""What hipPointerGetAttributes reports for a GEM-imported KV buffer vs plain device memory (Mooncake's getMemoryLocation
uses it: type == device -> location "hip:<dev>", else NUMA lookup / wildcard, which disables MC_ENABLE_DEST_LOCAL_RAIL)."""
import ctypes as C
import os
import torch

hip = C.CDLL(None) if os.environ.get("LD_PRELOAD") else C.CDLL("libamdhip64.so")  # global scope: preload first, as Mooncake links it


class Attr(C.Structure):  # hipPointerAttribute_t (ROCm 7): type, device, devicePointer, hostPointer, isManaged, allocationFlags
    _fields_ = [("type", C.c_int), ("device", C.c_int), ("devicePointer", C.c_void_p), ("hostPointer", C.c_void_p),
                ("isManaged", C.c_int), ("allocationFlags", C.c_uint)]


def show(tag, ptr):
    a = Attr()
    rc = hip.hipPointerGetAttributes(C.byref(a), C.c_void_p(ptr))
    print(f"{tag}: rc={rc} type={a.type} device={a.device} isManaged={a.isManaged}", flush=True)


torch.cuda.set_device(int(os.environ.get("DEV", "3")))
k = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
show("kfd (plain torch)", k.data_ptr())
so = os.environ.get("SGLANG_AMD_GEM_ALLOCATOR")
if so:
    alloc = torch.cuda.memory.CUDAPluggableAllocator(so, "hip_gem_alloc", "hip_gem_free")
    pool = torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        g = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    show("gem (MemPool)", g.data_ptr())
print("hipMemoryType: 0=unregistered/host? 1=host 2=device 3=managed (ROCm 7 enum: hipMemoryTypeHost=1, Device=2)")
