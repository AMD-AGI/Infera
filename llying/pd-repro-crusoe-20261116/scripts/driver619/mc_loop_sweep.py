#!/usr/bin/env python3
"""Same-node Mooncake RDMA writes as the one-node PD deployment does them (crusoe, 11-16): KV transfers time out there
(30 s "Sync batch data transfer timeout") for 16k prompts while short ones pass.

target:    allocates --gib of 6 GiB buffers (pool kfd = plain torch, gem = gemalloc MemPool), registers them with Mooncake
           (batch_register_memory, as SGLang does), publishes its address list and waits for every initiator.
initiator: a 1 GiB source buffer in the same pool type, registered too; then
  sweep  (--sweep): for every target buffer, one 1 GiB write in 16 x 64 MiB slices and one 128 MiB write in 2048 x 64 KiB
         slices (KV-page-like), each timed, rc recorded
  load   (--load SECS): 64 KiB-slice batches (2048 per batch) to regions spread over the whole target, for SECS seconds;
         several initiators run this at once against one target (4 PP prefill ranks into one decode rank)
One JSON line per step. Usage: --role target|initiator --pool kfd|gem --gib N --nic ionic_X --sig DIR [--id K]
"""
import argparse
import json
import os
import socket
import sys
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--role", choices=["target", "initiator"], required=True)
p.add_argument("--pool", choices=["kfd", "gem", "host"], default="kfd")  # host: pinned host memory (initiator only)
p.add_argument("--gib", type=int, default=120)
p.add_argument("--nic", default="ionic_4")
p.add_argument("--sig", required=True)
p.add_argument("--id", type=int, default=0)
p.add_argument("--initiators", type=int, default=1)
p.add_argument("--sweep", type=int, default=1)
p.add_argument("--load", type=float, default=0)
p.add_argument("--tag", default="")
a = p.parse_args()
GiB, MiB, KiB = 1 << 30, 1 << 20, 1 << 10
BUF = int(os.environ.get("MC_BUF_GIB", "6")) * GiB  # 6 GiB KFD buffers fail hipIpcGetMemHandle on crusoe v3
SO = os.environ.get("SGLANG_AMD_GEM_ALLOCATOR", "")
T0 = time.time()


def out(step, **kw):
    d = dict(tag=a.tag, role=a.role, id=a.id, pool=a.pool, nic=a.nic, step=step, t=round(time.time() - T0, 1))
    d.update(kw)
    print(json.dumps(d), flush=True)


def wait(path, secs=900):
    for _ in range(secs * 5):
        if os.path.exists(path):
            return True
        time.sleep(0.2)
    return False


torch.cuda.set_device(0)
n = max(1, a.gib * GiB // BUF) if a.role == "target" else 1
size = BUF if a.role == "target" else GiB
pool = None
if a.pool == "gem":
    alloc = torch.cuda.memory.CUDAPluggableAllocator(SO, "hip_gem_alloc", "hip_gem_free")
    pool = torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        bufs = [torch.zeros(size, dtype=torch.uint8, device="cuda") for _ in range(n)]
elif a.pool == "host":
    bufs = [torch.zeros(size, dtype=torch.uint8).pin_memory() for _ in range(n)]
else:
    bufs = [torch.zeros(size, dtype=torch.uint8, device="cuda") for _ in range(n)]
torch.cuda.synchronize()
from mooncake.engine import TransferEngine  # noqa: E402

ip = socket.gethostbyname(socket.gethostname())
te = TransferEngine()
rc = te.initialize(ip, "P2PHANDSHAKE", "rdma", a.nic)
t = time.time()
rc2 = te.batch_register_memory([b.data_ptr() for b in bufs], [b.numel() for b in bufs])
out("registered", buffers=n, gib=round(n * size / GiB, 1), init_rc=rc, reg_rc=rc2, secs=round(time.time() - t, 1))
if rc or rc2:
    sys.exit(1)

if a.role == "target":
    info = {"hostname": f"{ip}:{te.get_rpc_port()}", "addrs": [b.data_ptr() for b in bufs]}
    with open(os.path.join(a.sig, "target.json.tmp"), "w") as f:
        json.dump(info, f)
    os.replace(os.path.join(a.sig, "target.json.tmp"), os.path.join(a.sig, "target.json"))
    done = [wait(os.path.join(a.sig, f"done.{i}"), 1800) for i in range(a.initiators)]
    out("end", initiators_done=sum(done))
    sys.exit(0)

if not wait(os.path.join(a.sig, "target.json"), 600):
    out("error", what="no target")
    sys.exit(1)
tgt = json.load(open(os.path.join(a.sig, "target.json")))
host, addrs = tgt["hostname"], tgt["addrs"]
src0 = bufs[0].data_ptr()


def write(dst_base, slice_bytes, count):
    src = [src0 + (i * slice_bytes) % GiB for i in range(count)]
    dst = [dst_base + i * slice_bytes for i in range(count)]
    t = time.time()
    r = te.batch_transfer_sync_write(host, src, dst, [slice_bytes] * count)
    dt = time.time() - t
    return r, round(dt, 3), round(slice_bytes * count / dt / 1e9, 2) if r >= 0 and dt > 0 else 0


if a.sweep:
    step = max(1, len(addrs) // 20)
    for i, base in list(enumerate(addrs))[::step]:
        r1, s1, g1 = write(base, 64 * MiB, min(int(os.environ.get("BIG_N", "16")), BUF // (64 * MiB)))
        r2, s2, g2 = write(base + BUF // 2, 64 * KiB, 2048)
        out("sweep", buf=i, off_gib=round(i * BUF / GiB, 1), big_rc=r1, big_s=s1, big_gbs=g1, small_rc=r2, small_s=s2, small_gbs=g2)
if a.load:
    open(os.path.join(a.sig, f"ready.{a.id}"), "w").close()
    for _ in range(600):
        if all(os.path.exists(os.path.join(a.sig, f"ready.{i}")) for i in range(a.initiators)):
            break
        time.sleep(0.1)
    t, it, bad, worst, nb = time.time(), 0, 0, 0.0, 0
    while time.time() - t < a.load:
        base = addrs[(it * 7 + a.id) % len(addrs)] + (a.id % 4) * (BUF // 4)
        r, s, g = write(base, 64 * KiB, 2048)
        it += 1
        worst = max(worst, s)
        if r < 0:
            bad += 1
        else:
            nb += 64 * KiB * 2048
    el = time.time() - t
    out("load", batches=it, failed=bad, worst_batch_s=worst, gb_s=round(nb / el / 1e9, 2), secs=round(el, 1))
open(os.path.join(a.sig, f"done.{a.id}"), "w").close()
out("end")
