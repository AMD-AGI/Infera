#!/usr/bin/env python3
"""DRIVER619-20261113: the production start-up crash in miniature, and the GEM-backed fix, with torch + Mooncake.

The target allocates a KV-sized set of 6 GiB buffers (KFD: plain torch; GEM: a torch.cuda.MemPool on
gemalloc/libhip_gem_alloc.so), registers them with Mooncake (batch_register_memory, as SGLang does), then does what
scheduler.py run_event_loop does: reads .cuda_stream of a fresh torch.cuda.Stream(priority=0), which segfaults on
amdgpu 6.19 once the registration is charged twice. The initiator then RDMA-writes 1 GiB into the target across a 1 GiB
MR boundary (offset 4.5 GiB of the first buffer) and the target verifies it.

Usage (one process per GPU, ROCR_VISIBLE_DEVICES set): mc_gem_test.py --role target|initiator --pool gem|kfd --gib N
  --nic ionic_X --sig DIR
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
p.add_argument("--pool", choices=["gem", "kfd"], default="gem")
p.add_argument("--gib", type=int, default=150)
p.add_argument("--nic", default="ionic_4")
p.add_argument("--sig", required=True)
p.add_argument("--tag", default="")
a = p.parse_args()
GiB = 1 << 30
BUF = 6 * GiB
OFF = 4 * GiB + GiB // 2
SO = "/perf_apps/liyingli/bench_agentx/kfd619/gemalloc/build/libhip_gem_alloc.so"
T0 = time.time()


def out(step, **kw):
    free, total = torch.cuda.mem_get_info()
    d = dict(tag=a.tag, role=a.role, pool=a.pool, step=step, t=round(time.time() - T0, 1),
             rep_free_gib=round(free / GiB, 2), alloc_gib=round(torch.cuda.memory_allocated() / GiB, 2))
    d.update(kw)
    print(json.dumps(d), flush=True)


def wait(path, secs=900):
    for _ in range(secs * 2):
        if os.path.exists(path):
            return True
        time.sleep(0.5)
    return False


torch.cuda.set_device(0)
out("start", driver=open("/sys/module/amdgpu/version").read().strip(),
    env={k: os.environ.get(k) for k in ("MOONCAKE_DISABLE_HIP_DMABUF", "MC_HIP_KFD_PIN_FIX", "MC_DISABLE_HIP")})
n = max(1, a.gib // 6)
pool = None
if a.pool == "gem":
    alloc = torch.cuda.memory.CUDAPluggableAllocator(SO, "hip_gem_alloc", "hip_gem_free")
    pool = torch.cuda.MemPool(alloc.allocator())
    with torch.cuda.use_mem_pool(pool):
        bufs = [torch.zeros(BUF, dtype=torch.uint8, device="cuda") for _ in range(n)]
else:
    bufs = [torch.zeros(BUF, dtype=torch.uint8, device="cuda") for _ in range(n)]
torch.cuda.synchronize()
out("allocated", buffers=n, gib=n * 6)

from mooncake.engine import TransferEngine  # noqa: E402

ip = socket.gethostbyname(socket.gethostname())
te = TransferEngine()
rc = te.initialize(ip, "P2PHANDSHAKE", "rdma", a.nic)
t = time.time()
rc2 = te.batch_register_memory([b.data_ptr() for b in bufs], [b.numel() for b in bufs])
out("registered", init_rc=rc, reg_rc=rc2, secs=round(time.time() - t, 1))
if rc or rc2:
    sys.exit(1)

s = torch.cuda.Stream(priority=0)
out("new_stream", handle=hex(s.cuda_stream))
with torch.cuda.stream(s):
    bufs[-1][:GiB].fill_(1)
s.synchronize()
try:
    extra = torch.empty(8 * GiB, dtype=torch.uint8, device="cuda")
    out("extra_8gib", ok=True)
    del extra
except Exception as e:  # noqa: BLE001
    out("extra_8gib", ok=False, err=str(e)[:160])

pat = torch.arange(GiB // 4, dtype=torch.int32, device="cuda").view(torch.uint8)
if a.role == "target":
    bufs[0].zero_()
    torch.cuda.synchronize()
    with open(os.path.join(a.sig, "target.json.tmp"), "w") as f:
        json.dump({"hostname": f"{ip}:{te.get_rpc_port()}", "addr": bufs[0].data_ptr() + OFF}, f)
    os.replace(os.path.join(a.sig, "target.json.tmp"), os.path.join(a.sig, "target.json"))
    ok = wait(os.path.join(a.sig, "done"))
    torch.cuda.synchronize()
    good = ok and bool(torch.equal(bufs[0][OFF:OFF + GiB], pat)) and int(bufs[0][:OFF].count_nonzero()) == 0
    out("verified", done=ok, ok=good)
    if len(bufs) >= 3:
        def gbs(fn, nbytes, n=20):
            fn()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return round(nbytes * n / (time.time() - t) / 1e9, 1)

        g1, g2 = bufs[1][:4 * GiB], bufs[2][:4 * GiB]
        k1 = torch.empty(4 * GiB, dtype=torch.uint8, device="cuda")
        k2 = torch.empty(4 * GiB, dtype=torch.uint8, device="cuda")
        out("gpu_bw", copy_this_pool=gbs(lambda: g2.copy_(g1), 8 * GiB), copy_default=gbs(lambda: k2.copy_(k1), 8 * GiB),
            read_this_pool=gbs(lambda: g1.view(torch.int32).sum(), 4 * GiB),
            read_default=gbs(lambda: k1.view(torch.int32).sum(), 4 * GiB))
        del k1, k2
else:
    if not wait(os.path.join(a.sig, "target.json"), 420):
        out("error", what="no target")
        sys.exit(1)
    tgt = json.load(open(os.path.join(a.sig, "target.json")))
    bufs[0][:GiB].copy_(pat)
    torch.cuda.synchronize()
    chunk, k = 64 << 20, 16
    src = [bufs[0].data_ptr() + i * chunk for i in range(k)]
    dst = [tgt["addr"] + i * chunk for i in range(k)]
    rc = te.batch_transfer_sync_write(tgt["hostname"], src, dst, [chunk] * k)
    t, it = time.time(), 0
    while time.time() - t < 5 and rc >= 0:
        rc = te.batch_transfer_sync_write(tgt["hostname"], src, dst, [chunk] * k)
        it += 1
    gbs = it * GiB / (time.time() - t) / 1e9 if it else 0
    out("transferred", rc=rc, iters=it, gb_s=round(gbs, 2))
    open(os.path.join(a.sig, "done"), "w").close()
out("end")
