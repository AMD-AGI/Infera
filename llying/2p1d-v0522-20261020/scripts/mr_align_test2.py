"""Allocate KV-buffer-sized tensors (2,320,846,848 bytes, the decode's per-layer buffer) until the
GPU is nearly full, then register whole buffers with Mooncake's RDMA transport on all given rails:
every buffer whose allocation starts 1 GiB-aligned, and a few that do not, as a control.
  python3 mr_align_test2.py <local_ip> <rdma_devices>"""

import sys

import torch
from mooncake.engine import TransferEngine

GIB = 1 << 30
SIZE = 2320846848
ip, devices = sys.argv[1], sys.argv[2]
te = TransferEngine()
assert te.initialize(ip, "P2PHANDSHAKE", "rdma", devices) == 0

bufs = []
while torch.cuda.mem_get_info()[0] > SIZE + 4 * GIB:
    bufs.append(torch.empty(SIZE, dtype=torch.uint8, device="cuda"))
bases = [b.data_ptr() for b in bufs]
aligned = [p for p in bases if p % GIB == 0]
control = [p for p in bases if p % GIB][:4]
print(f"{len(bases)} buffers, {len(aligned)} start 1 GiB-aligned", flush=True)
for p in aligned + control:
    rc = te.register_memory(p, SIZE)
    print(f"base={p:#x} aligned={p % GIB == 0} -> rc={rc}", flush=True)
