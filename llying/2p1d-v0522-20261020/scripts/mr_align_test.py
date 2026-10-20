"""Register VRAM ranges with Mooncake's RDMA transport (dma-buf path) to see whether a range
starting at a 1 GiB-aligned address fails on ionic, as two decode KV buffers did.
Run inside the infera-sglang image with GPU and /dev/infiniband access:
  python3 mr_align_test.py <local_ip> <rdma_devices>"""

import os
import sys

import torch
from mooncake.engine import TransferEngine

GIB, MIB = 1 << 30, 1 << 20
ip, devices = sys.argv[1], sys.argv[2]
os.environ.setdefault("MC_TE_FILTERS", devices)
te = TransferEngine()
assert te.initialize(ip, "P2PHANDSHAKE", "rdma", devices) == 0

buf = torch.empty(5 * GIB, dtype=torch.uint8, device="cuda")
base = buf.data_ptr()
aligned = (base + GIB - 1) // GIB * GIB
print(f"buffer {base:#x}, first 1 GiB-aligned address {aligned:#x}")


def reg(addr: int, length: int, label: str) -> None:
    rc = te.register_memory(addr, length)
    print(f"{label:40s} addr={addr:#x} len={length / MIB:.0f} MiB -> rc={rc}", flush=True)
    if rc == 0:
        te.unregister_memory(addr)


reg(aligned, 2 * GIB, "aligned, 2 GiB")
reg(aligned + 2 * MIB, 2 * GIB, "aligned + 2 MiB, 2 GiB")
reg(aligned, GIB, "aligned, 1 GiB")
reg(aligned, 512 * MIB, "aligned, 512 MiB")
reg(aligned, 2 * MIB, "aligned, 2 MiB")
reg(aligned + GIB // 2, 2 * GIB, "aligned + 512 MiB, 2 GiB")
