#!/usr/bin/env python3
"""Cross-node Mooncake RDMA check with host memory, in the direction vLLM uses (the prefill side writes into the
decode side with batch_transfer_sync_write over a P2P handshake).
  mc_xnode.py target    DIR DEVICES [MB=256]   register a buffer, publish DIR/target.json, wait for DIR/done, verify
  mc_xnode.py initiator DIR DEVICES [MB=256]   wait for DIR/target.json, write a pattern into the target's buffer
DEVICES is the Mooncake device_name list (e.g. ionic_0 or ionic_0,...,ionic_7). Env (MC_*) is passed through.
"""
import json
import os
import socket
import sys
import time

import numpy as np
from mooncake.engine import TransferEngine

role, d, devices = sys.argv[1:4]
mb = int(sys.argv[4]) if len(sys.argv) > 4 else 256
size = mb << 20
ip = os.environ.get("HOST_IP") or socket.gethostbyname(socket.gethostname())
env = {k: v for k, v in os.environ.items() if k.startswith("MC_")}
print(f"{role} {ip} devices={devices} {mb} MB env={env}", flush=True)

te = TransferEngine()
assert te.initialize(ip, "P2PHANDSHAKE", "rdma", devices) == 0, "initialize failed"
session = f"{ip}:{te.get_rpc_port()}"
buf = np.zeros(size, dtype=np.uint8)
ptr = buf.ctypes.data
assert te.register_memory(ptr, size) == 0, "register_memory failed"

if role == "target":
    json.dump({"session": session, "ptr": ptr, "size": size}, open(f"{d}/target.json.tmp", "w"))
    os.rename(f"{d}/target.json.tmp", f"{d}/target.json")
    t_end = time.time() + 600
    while not os.path.exists(f"{d}/done") and time.time() < t_end:
        time.sleep(1)
    expect = (np.arange(size, dtype=np.uint64) % 251).astype(np.uint8)
    print("target: content", "OK" if np.array_equal(buf, expect) else f"MISMATCH ({int((buf != expect).sum())} bytes)")
else:
    while not os.path.exists(f"{d}/target.json"):
        time.sleep(1)
    t = json.load(open(f"{d}/target.json"))
    buf[:] = (np.arange(size, dtype=np.uint64) % 251).astype(np.uint8)
    for n_chunks in (1, 64, 1024):
        chunk = size // n_chunks
        src = [ptr + i * chunk for i in range(n_chunks)]
        dst = [t["ptr"] + i * chunk for i in range(n_chunks)]
        t0 = time.time()
        rc = te.batch_transfer_sync_write(t["session"], src, dst, [chunk] * n_chunks)
        dt = time.time() - t0
        print(f"initiator: {n_chunks} x {chunk >> 10} KB -> {t['session']}: rc={rc} {dt:.3f} s"
              f" {size / dt / 1e9 if rc == 0 else 0:.2f} GB/s", flush=True)
    open(f"{d}/done", "w").close()
