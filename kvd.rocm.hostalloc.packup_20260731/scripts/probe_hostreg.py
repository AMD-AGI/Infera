#!/usr/bin/env python3
"""Why does mmap+cudaHostRegister memory fault when a GPU kernel dereferences it,
while torch pin_memory does not?

Three candidate mechanisms, all checked here in one process:

  M1  HIP maps registered host memory at a DIFFERENT device address, so the raw
      host VA in the pointer table is not valid on device.
      -> compare hipHostGetDevicePointer(host_ptr) with host_ptr.
  M2  XNACK / page-migration is off, so the GPU cannot fault-in host pages that
      were not allocated through the HIP allocator.
      -> report HSA_XNACK and the agent's xnack mode.
  M3  The registration silently didn't cover the range (rc==0 but no mapping).
      -> query hipPointerGetAttributes on the host pointer.

Nothing here launches the transfer kernel, so it cannot abort the process; every
mechanism is reported for every allocation mode.
"""
import ctypes
import math
import mmap
import os

import torch

HOST_REGISTER_DEFAULT = 0x0
HOST_REGISTER_PORTABLE = 0x1
HOST_REGISTER_MAPPED = 0x2

_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ctypes.c_long]


def find_hip():
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


hip = find_hip()


def alloc_mmap_host(n_bytes, shared=True, populate=True):
    alloc_bytes = math.ceil(n_bytes / mmap.PAGESIZE) * mmap.PAGESIZE
    flags = mmap.MAP_ANONYMOUS | (mmap.MAP_SHARED if shared else mmap.MAP_PRIVATE)
    if populate:
        flags |= getattr(mmap, "MAP_POPULATE", 0x08000)
    ptr = _libc.mmap(None, alloc_bytes, mmap.PROT_READ | mmap.PROT_WRITE,
                     flags, -1, 0)
    if ptr is None or ptr == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), "mmap failed")
    arr = (ctypes.c_byte * alloc_bytes).from_address(ptr)
    return torch.frombuffer(arr, dtype=torch.uint8, count=n_bytes), ptr


def dev_ptr_of(host_ptr):
    """M1: hipHostGetDevicePointer."""
    if hip is None:
        return None, "no libamdhip64"
    out = ctypes.c_void_p()
    rc = hip.hipHostGetDevicePointer(ctypes.byref(out),
                                     ctypes.c_void_p(host_ptr),
                                     ctypes.c_uint(0))
    if rc != 0:
        return None, f"rc={rc}"
    return out.value, "ok"


class HipPointerAttr(ctypes.Structure):
    # hipPointerAttribute_t layout varies across ROCm; we only read the first
    # fields defensively and report raw bytes if the shape looks wrong.
    _fields_ = [("type", ctypes.c_int),
                ("device", ctypes.c_int),
                ("devicePointer", ctypes.c_void_p),
                ("hostPointer", ctypes.c_void_p),
                ("isManaged", ctypes.c_int),
                ("allocationFlags", ctypes.c_uint)]


def ptr_attrs(p):
    """M3: hipPointerGetAttributes."""
    if hip is None:
        return "no libamdhip64"
    a = HipPointerAttr()
    rc = hip.hipPointerGetAttributes(ctypes.byref(a), ctypes.c_void_p(p))
    if rc != 0:
        return f"rc={rc}"
    return (f"type={a.type} device={a.device} "
            f"devPtr={(a.devicePointer or 0):#x} hostPtr={(a.hostPointer or 0):#x} "
            f"managed={a.isManaged} flags={a.allocationFlags:#x}")


def register(host_ptr, n, flags):
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostRegister(host_ptr, n, flags)
    return int(rc)


def main():
    torch.cuda.init()
    print("=== M2: XNACK / environment ===")
    for k in ("HSA_XNACK", "HSA_NO_SCRATCH_RECLAIM", "HIP_VISIBLE_DEVICES",
              "SGLANG_HUGEPAGE_SIZE"):
        print(f"  {k}={os.environ.get(k)!r}")
    try:
        props = torch.cuda.get_device_properties(0)
        print(f"  gcnArch={getattr(props,'gcnArchName',None)} "
              f"name={props.name}")
    except Exception as e:
        print("  props failed:", e)
    try:
        with open("/sys/module/amdgpu/parameters/noretry") as fh:
            print(f"  amdgpu.noretry={fh.read().strip()}")
    except OSError as e:
        print(f"  amdgpu.noretry=<unreadable: {e}>")

    N = 8 << 20  # 8 MiB is plenty to characterise the mapping

    print("\n=== per-allocation-mode mapping check ===")

    # 1) torch pin_memory (the mode that WORKED in the kernel probe)
    t = torch.zeros(N, dtype=torch.uint8, device="cpu", pin_memory=True)
    p = t.data_ptr()
    dp, msg = dev_ptr_of(p)
    print(f"\n[pin_memory]        host={p:#x}")
    print(f"  hipHostGetDevicePointer -> {dp if dp is None else hex(dp)} ({msg}) "
          f"same={dp == p}")
    print(f"  attrs: {ptr_attrs(p)}")

    # 2..n) mmap variants + cudaHostRegister
    for label, shared, populate, flags in (
        ("mmap_default", True, True, HOST_REGISTER_DEFAULT),
        ("mmap_mapped", True, True, HOST_REGISTER_MAPPED),
        ("mmap_portable_mapped", True, True,
         HOST_REGISTER_PORTABLE | HOST_REGISTER_MAPPED),
        ("mmap_private", False, True, HOST_REGISTER_DEFAULT),
    ):
        buf, raw = alloc_mmap_host(N, shared=shared, populate=populate)
        p = buf.data_ptr()
        rc = register(p, N, flags)
        dp, msg = dev_ptr_of(p)
        print(f"\n[{label}]  host={p:#x} register_rc={rc}")
        print(f"  hipHostGetDevicePointer -> {dp if dp is None else hex(dp)} ({msg}) "
              f"same={dp == p}")
        print(f"  attrs: {ptr_attrs(p)}")


if __name__ == "__main__":
    main()
