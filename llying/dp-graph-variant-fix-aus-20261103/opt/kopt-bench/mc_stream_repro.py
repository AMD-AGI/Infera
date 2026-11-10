#!/usr/bin/env python3
"""REPORT-20261110 section 6: minimal repro of the start-up segfault on amdgpu 6.19 nodes, following the order of an
sglang PD scheduler: a torch.cuda.Stream(priority=0) first, then a KV-pool-sized VRAM buffer registered with a Mooncake
TransferEngine (RDMA over ionic, 1 GiB chunks), then the stream's handle (scheduler.py run_event_loop reads
schedule_stream.cuda_stream there and crashes in c10::hip::HIPStream::stream()).
A non-PD server (no Mooncake) does not crash on the same node.

Usage (inside the infera image, one process per GPU): GPU=i GB=40 NIC=ionic_i mc_stream_repro.py
"""
import os
import socket
import sys
import time

import torch

gpu, gb, nic = int(os.environ.get("GPU", "0")), int(os.environ.get("GB", "40")), os.environ.get("NIC", "ionic_0")
chunk = int(os.environ.get("CHUNK_GB", "1")) << 30
torch.cuda.set_device(gpu)
s = torch.cuda.Stream(priority=0)
f = torch.cuda.Stream(priority=0)
print(f"[{gpu}] streams {hex(s.cuda_stream)} {hex(f.cuda_stream)}", flush=True)
kv = torch.empty(gb << 30, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
from mooncake.engine import TransferEngine  # noqa: E402

te = TransferEngine()
ip = os.environ.get("IP") or socket.gethostbyname(socket.gethostname())
rc = te.initialize(ip, "P2PHANDSHAKE", "rdma", nic)
print(f"[{gpu}] initialize rc={rc}", flush=True)
t0 = time.time()
if os.environ.get("BATCH", "0") == "1":
    ptrs = [kv.data_ptr() + o for o in range(0, kv.numel(), chunk)]
    lens = [min(chunk, kv.numel() - o) for o in range(0, kv.numel(), chunk)]
    rc = te.batch_register_memory(ptrs, lens)
else:
    rc = te.register_memory(kv.data_ptr(), kv.numel())
print(f"[{gpu}] register {gb} GiB rc={rc} in {time.time() - t0:.1f} s", flush=True)
for i in range(int(os.environ.get("POLL", "20"))):
    print(f"[{gpu}] after register {i}: {hex(s.cuda_stream)} {hex(f.cuda_stream)} eq={s.cuda_stream == f.cuda_stream}",
          flush=True)
    with torch.cuda.stream(s):
        kv[:1024].add_(1)
    torch.cuda.synchronize()
    time.sleep(float(os.environ.get("SLEEP", "1")))
print(f"[{gpu}] ok", flush=True)
sys.exit(0)
