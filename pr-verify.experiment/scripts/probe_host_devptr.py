#!/usr/bin/env python3
"""Measure whether a host buffer's DEVICE pointer equals its host VA, per
allocation strategy, on this GPU.

This is the measurement the whole hicache-allocator fix rests on. sglang's
host pools hand `data_ptr()` of host buffers to GPU kernels through a
device-side pointer table, which is only correct if the two addresses agree.

Run inside a ROCm (or CUDA) container that has torch. Prints one line per
strategy; `same=False` on any line means that strategy is unsafe for the
pointer-table design on this GPU.
"""

import ctypes
import sys

import torch

N_BYTES = 8 << 20  # 8 MiB, same size the original gfx950 measurement used


def _hip():
    for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def dev_ptr(lib, host_ptr):
    """hipHostGetDevicePointer / cudaHostGetDevicePointer -> int, or None."""
    out = ctypes.c_void_p()
    fn = getattr(lib, "hipHostGetDevicePointer", None) or getattr(
        lib, "cudaHostGetDevicePointer", None
    )
    rc = fn(ctypes.byref(out), ctypes.c_void_p(host_ptr), ctypes.c_uint(0))
    return None if int(rc) != 0 else (out.value or 0)


def report(label, host_ptr, lib):
    d = dev_ptr(lib, host_ptr)
    if d is None:
        print(f"  {label:<34} host={host_ptr:#x}  devPtr=<query failed>")
        return None
    print(f"  {label:<34} host={host_ptr:#x}  devPtr={d:#x}  same={d == host_ptr}")
    return d == host_ptr


def main():
    lib = _hip()
    if lib is None:
        sys.exit("cannot load libamdhip64")
    lib.hipHostGetDevicePointer.restype = ctypes.c_int
    lib.hipHostRegister.restype = ctypes.c_int

    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} gcn={getattr(props, 'gcnArchName', '?')}")
    print(f"torch:  {torch.__version__} hip={torch.version.hip}")
    print()

    results = {}

    # Strategy 1 -- what alloc_with_pin_memory does (torch pin_memory=True,
    # hipHostMalloc underneath).
    t = torch.empty(N_BYTES, dtype=torch.uint8, pin_memory=True)
    results["pin_memory"] = report("[pin_memory]", t.data_ptr(), lib)

    # Strategy 2 -- what alloc_with_host_register does (anonymous mmap, then
    # hipHostRegister). Flag values: 0, hipHostRegisterMapped=0x2,
    # hipHostRegisterPortable|Mapped=0x3.
    import mmap as _mmap

    for label, flags in (
        ("[mmap + hipHostRegister]", 0),
        ("[  + hipHostRegisterMapped]", 0x2),
        ("[  + Portable|Mapped]", 0x3),
    ):
        buf = _mmap.mmap(-1, N_BYTES)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        rc = lib.hipHostRegister(ctypes.c_void_p(addr), ctypes.c_size_t(N_BYTES), flags)
        if int(rc) != 0:
            print(f"  {label:<34} hipHostRegister failed rc={int(rc)}")
            continue
        results[label] = report(label, addr, lib)

    print()
    mismatched = [k for k, v in results.items() if v is False]
    if mismatched:
        print("VERDICT: host VA != device pointer for:", ", ".join(mismatched))
        print("         The pointer-table design is UNSAFE with those strategies here.")
    else:
        print("VERDICT: host VA == device pointer for every strategy on this GPU.")
        print("         This GPU does not exhibit the fault; it is a negative control.")


if __name__ == "__main__":
    main()
