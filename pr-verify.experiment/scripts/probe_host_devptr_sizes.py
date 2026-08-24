#!/usr/bin/env python3
"""Sweep buffer SIZE against the host-VA == device-pointer question.

The original gfx950 fault report measured a 7.33 GB host indexer buffer and saw
host != devPtr. `probe_host_devptr.py` uses 8 MiB and, on this box, measures them
EQUAL on the same arch and ROCm version. Size is the most obvious uncontrolled
variable between the two, so sweep it rather than guess.

Prints one line per (strategy, size). `same=False` anywhere means the
pointer-table design is unsafe for that combination on this GPU.
"""

import ctypes
import mmap as _mmap
import sys

import torch

SIZES = [
    ("8 MiB", 8 << 20),
    ("256 MiB", 256 << 20),
    ("1 GiB", 1 << 30),
    ("4 GiB", 4 << 30),
    ("7.33 GB", 7_330_000_000),
]


def _hip():
    for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def dev_ptr(lib, host_ptr):
    out = ctypes.c_void_p()
    fn = getattr(lib, "hipHostGetDevicePointer", None) or getattr(
        lib, "cudaHostGetDevicePointer", None
    )
    rc = fn(ctypes.byref(out), ctypes.c_void_p(host_ptr), ctypes.c_uint(0))
    return None if int(rc) != 0 else (out.value or 0)


def report(label, host_ptr, lib):
    d = dev_ptr(lib, host_ptr)
    if d is None:
        print(f"  {label:<46} host={host_ptr:#x}  devPtr=<query failed>", flush=True)
        return None
    same = d == host_ptr
    delta = "" if same else f"  delta={d - host_ptr:+#x}"
    print(
        f"  {label:<46} host={host_ptr:#x}  devPtr={d:#x}  same={same}{delta}",
        flush=True,
    )
    return same


def main():
    lib = _hip()
    if lib is None:
        sys.exit("cannot load libamdhip64")
    lib.hipHostGetDevicePointer.restype = ctypes.c_int
    lib.hipHostRegister.restype = ctypes.c_int
    lib.hipHostUnregister.restype = ctypes.c_int

    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} gcn={getattr(props, 'gcnArchName', '?')}")
    print(f"torch:  {torch.__version__} hip={torch.version.hip}")
    print()

    mismatched = []

    for size_label, n in SIZES:
        print(f"--- size {size_label} ({n} bytes)")

        # Strategy 1: torch pin_memory=True (hipHostMalloc underneath).
        try:
            t = torch.empty(n, dtype=torch.uint8, pin_memory=True)
            if report(f"[pin_memory] {size_label}", t.data_ptr(), lib) is False:
                mismatched.append(f"pin_memory@{size_label}")
            del t
        except Exception as e:  # noqa: BLE001 - report and keep sweeping
            print(f"  [pin_memory] {size_label:<34} FAILED: {type(e).__name__}: {e}")

        # Strategy 2: anonymous mmap + hipHostRegister, all three flag variants.
        for flag_label, flags in (
            ("mmap+Register", 0),
            ("mmap+Mapped", 0x2),
            ("mmap+Portable|Mapped", 0x3),
        ):
            label = f"[{flag_label}] {size_label}"
            try:
                buf = _mmap.mmap(-1, n)
            except Exception as e:  # noqa: BLE001
                print(f"  {label:<46} mmap FAILED: {type(e).__name__}: {e}")
                continue
            addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
            rc = lib.hipHostRegister(ctypes.c_void_p(addr), ctypes.c_size_t(n), flags)
            if int(rc) != 0:
                print(f"  {label:<46} hipHostRegister failed rc={int(rc)}")
                buf.close()
                continue
            if report(label, addr, lib) is False:
                mismatched.append(f"{flag_label}@{size_label}")
            lib.hipHostUnregister(ctypes.c_void_p(addr))
            del addr
            buf.close()
        print()

    if mismatched:
        print("VERDICT: host VA != device pointer for:", ", ".join(mismatched))
        print("         The pointer-table design is UNSAFE for those combinations.")
    else:
        print("VERDICT: host VA == device pointer at every size and strategy tested.")
        print("         This GPU does not exhibit the fault at any tested size.")


if __name__ == "__main__":
    main()
