"""On an allocation that starts 1 GiB-aligned (the failing case), try ways to register it with
Mooncake's RDMA transport on all given rails.
  python3 mr_align_test3.py <local_ip> <rdma_devices>"""

import sys

import torch
from mooncake.engine import TransferEngine

GIB, MIB = 1 << 30, 1 << 20
SIZE = 2320846848
ip, devices = sys.argv[1], sys.argv[2]
te = TransferEngine()
assert te.initialize(ip, "P2PHANDSHAKE", "rdma", devices) == 0

bufs = []
while torch.cuda.mem_get_info()[0] > SIZE + 4 * GIB:
    bufs.append(torch.empty(SIZE, dtype=torch.uint8, device="cuda"))
aligned = [b.data_ptr() for b in bufs if b.data_ptr() % GIB == 0]
print(f"{len(bufs)} buffers, aligned: {[hex(p) for p in aligned]}", flush=True)


def reg(addr: int, length: int, label: str, keep: bool = False) -> int:
    rc = te.register_memory(addr, length)
    print(f"{label:34s} addr={addr:#x} len={length / MIB:.0f} MiB -> rc={rc}", flush=True)
    if rc == 0 and not keep:
        te.unregister_memory(addr)
    return rc


for p in aligned:
    reg(p, SIZE, "whole buffer")
    reg(p, GIB, "first 1 GiB")
    reg(p, 512 * MIB, "first 512 MiB")
    reg(p, 2 * MIB, "first 2 MiB")
    reg(p + 2 * MIB, SIZE - 2 * MIB, "skip first 2 MiB")
    if reg(p, 2 * MIB, "split: first 2 MiB", keep=True) == 0:
        reg(p + 2 * MIB, SIZE - 2 * MIB, "split: rest")
        te.unregister_memory(p)
