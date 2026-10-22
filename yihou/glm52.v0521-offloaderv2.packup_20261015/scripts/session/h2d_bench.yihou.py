#!/usr/bin/env python3
"""Task v0521cap2 OffloaderV2 gate: pinned host<->device copy bandwidth on MI355X, in the sglang image (torch ROCm).
Size = one offloaded MoE layer per GPU (1.21 GB). Cases: (1) each GPU alone H2D, (2) 4 GPUs concurrent H2D,
(3) 4 GPUs concurrent H2D + D2H, (4) case 2 with an extra 64 GiB pinned host allocation present (HiCache-like pinned pool, scaled down)."""
import os, sys, time, json
import torch, torch.multiprocessing as mp
SZ = int(1.21e9); REP = 8

def worker(gpu, mode, barrier, q):
    torch.cuda.set_device(gpu)
    h = torch.empty(SZ, dtype=torch.uint8, pin_memory=True); h.fill_(1)
    d = torch.empty(SZ, dtype=torch.uint8, device="cuda")
    h2 = torch.empty(SZ, dtype=torch.uint8, pin_memory=True) if mode == "both" else None
    d2 = torch.empty(SZ, dtype=torch.uint8, device="cuda") if mode == "both" else None
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    for _ in range(2): d.copy_(h, non_blocking=True)
    torch.cuda.synchronize()
    barrier.wait()
    t = time.perf_counter()
    for _ in range(REP):
        with torch.cuda.stream(s1): d.copy_(h, non_blocking=True)
        if d2 is not None:
            with torch.cuda.stream(s2): h2.copy_(d2, non_blocking=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    q.put((gpu, mode, round(SZ * REP / dt / 1e9, 2)))

def run(gpus, mode):
    ctx = mp.get_context("spawn"); b = ctx.Barrier(len(gpus)); q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(g, mode, b, q)) for g in gpus]
    [p.start() for p in ps]; [p.join() for p in ps]
    return sorted(q.get() for _ in gpus)

if __name__ == "__main__":
    n = torch.cuda.device_count(); print("visible GPUs", n, torch.version.hip, flush=True)
    res = {}
    for g in range(n): res[f"alone_gpu{g}_h2d"] = run([g], "h2d")
    res["concurrent4_h2d"] = run(list(range(n)), "h2d")
    res["concurrent4_h2d_plus_d2h"] = run(list(range(n)), "both")
    big = torch.empty(64 * 2**30, dtype=torch.uint8, pin_memory=True)  # extra pinned host pool present
    res["concurrent4_h2d_with_64GiB_extra_pinned"] = run(list(range(n)), "h2d")
    del big
    for k, v in res.items(): print(k, v, flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
