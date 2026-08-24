#!/usr/bin/env python3
"""#33968 write-back repro: does the GPU kernel survive dereferencing host VAs?

The probe (`probe_host_devptr.py`) establishes that on this machine
hipHostRegister maps pages at a device address that differs from the host VA.
This script establishes the *consequence*, which is what the PR actually claims:
the HiCache write-back kernel takes a device-resident table of **host** pointers
(`DSAIndexerPoolHost.index_k_data_ptrs`) and dereferences them from the device.

It calls the same kernel the real path calls --
`sgl_kernel.kvcacheio.transfer_kv_all_layer_mla(src_layers=..., dst_layers=...)`
-- with `dst_layers` built exactly the way `pool_host/dsa.py:159` builds it:

    torch.tensor([x.data_ptr() for x in host_buffers], dtype=torch.uint64,
                 device=<gpu>)

One arm per allocation strategy:

  host_register  what stock sglang picks on ROCm (ALLOC_MEMORY_FUNCS default,
                 since torch.device('cuda:0') misses the "cuda" key)
  pin_memory     what PR #33968 routes HIP to (hipHostMalloc)

PASS/FAIL is not about speed: the question is whether the kernel faults, and if
it does not, whether the bytes actually arrive at the host buffer. A silent
wrong-address write is the worse outcome and the data check catches it.

Run each arm in its own process (`--arm`), because a GPU memory access fault
poisons the HIP context for everything after it.
"""

import argparse
import ctypes
import os
import sys

import torch

LAYERS = 4
PAGES = 64
ITEM = 512  # bytes per page-layer slot; multiple of 8 so the kernel path is used
DTYPE = torch.uint8


def _hip():
    for name in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def alloc_pin_memory(shape, dtype):
    """What #33968 routes HIP to: torch pin_memory -> hipHostMalloc."""
    return torch.empty(shape, dtype=dtype, pin_memory=True)


def alloc_host_register(shape, dtype):
    """What stock sglang does on ROCm: plain host memory + hipHostRegister.

    Mirrors alloc_with_host_register in pool_host/common.py closely enough for
    the pointer-identity question: the buffer is ordinary (mmap-backed) host
    memory registered with the runtime, NOT hipHostMalloc.
    """
    lib = _hip()
    if lib is None:
        sys.exit("cannot load libamdhip64")
    lib.hipHostRegister.restype = ctypes.c_int
    t = torch.empty(shape, dtype=dtype)
    nbytes = t.numel() * t.element_size()
    # hipHostRegisterMapped | hipHostRegisterPortable
    rc = lib.hipHostRegister(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(nbytes),
                             ctypes.c_uint(0x2 | 0x1))
    if rc != 0:
        sys.exit(f"hipHostRegister failed rc={rc}")
    return t


ARMS = {"pin_memory": alloc_pin_memory, "host_register": alloc_host_register}


def devptr_of(host_ptr):
    lib = _hip()
    out = ctypes.c_void_p()
    lib.hipHostGetDevicePointer.restype = ctypes.c_int
    rc = lib.hipHostGetDevicePointer(ctypes.byref(out), ctypes.c_void_p(host_ptr),
                                     ctypes.c_uint(0))
    return None if rc != 0 else (out.value or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    args = ap.parse_args()

    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"arm={args.arm}")
    print(f"device: {props.name} gcn={getattr(props, 'gcnArchName', '?')}")
    print(f"torch:  {torch.__version__} hip={torch.version.hip}")

    alloc = ARMS[args.arm]

    # Device-side source, one buffer per layer, filled with a known pattern.
    src_bufs = [
        torch.full((PAGES, ITEM), 0xAB, dtype=DTYPE, device=dev) for _ in range(LAYERS)
    ]
    # Host-side destination, allocated the way this arm allocates.
    dst_bufs = [alloc((PAGES, ITEM), DTYPE) for _ in range(LAYERS)]
    for b in dst_bufs:
        b.fill_(0)

    same = devptr_of(dst_bufs[0].data_ptr()) == dst_bufs[0].data_ptr()
    print(f"host VA == device pointer for this arm: {same}")

    # The pointer table, built exactly as pool_host/dsa.py:159 builds it:
    # host data_ptr()s, in a tensor that lives on the DEVICE.
    src_ptrs = torch.tensor([b.data_ptr() for b in src_bufs],
                            dtype=torch.uint64, device=dev)
    dst_ptrs = torch.tensor([b.data_ptr() for b in dst_bufs],
                            dtype=torch.uint64, device=dev)

    n = 8
    src_idx = torch.arange(n, dtype=torch.int64, device=dev)
    dst_idx = torch.arange(n, dtype=torch.int64, device=dev)

    print(f"calling transfer_kv_all_layer_mla: {LAYERS} layers, {n} pages, item={ITEM}B")
    sys.stdout.flush()

    transfer_kv_all_layer_mla(
        src_layers=src_ptrs,
        dst_layers=dst_ptrs,
        src_indices=src_idx,
        dst_indices=dst_idx,
        item_size=ITEM,
        num_layers=LAYERS,
    )
    torch.cuda.synchronize()
    print("kernel returned and synchronized without fault")

    # Surviving the launch is not enough -- a wrong-address write can be silent.
    ok = True
    for li, b in enumerate(dst_bufs):
        got = b[:n]
        if not torch.all(got == 0xAB):
            bad = int((got != 0xAB).sum())
            print(f"  layer {li}: MISMATCH, {bad}/{got.numel()} bytes not 0xAB")
            ok = False
    if ok:
        print("data check: all copied pages carry the expected pattern")
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL (kernel ran but data did not arrive)")
        sys.exit(1)


if __name__ == "__main__":
    main()
