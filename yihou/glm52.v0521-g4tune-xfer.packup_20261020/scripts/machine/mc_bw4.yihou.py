# Task g4tune 2026-10-19: Mooncake microbenchmark #4 (load side via YX_LOADSIDE=send|recv; derived from #2) - production-like registration (many separate buffers) and
# optional concurrent GPU compute load. One pair GPU0 -> GPU4 (and 4 pairs), 360 KiB blocks, ~2 GiB per call.
import os, time, json, threading
import multiprocessing as mp
BLK = 368640; TOTAL = 2048 << 20; IP = os.environ["YX_IP"]

def mkbufs(torch, gpu, nbuf):
    per = (TOTAL // nbuf // BLK + 1) * BLK
    return [torch.empty(per, dtype=torch.uint8, device=f"cuda:{gpu}") for _ in range(nbuf)], per

def recv(gpu, nbuf, q, done):
    import torch
    from mooncake.engine import TransferEngine
    torch.cuda.set_device(gpu)
    bufs, per = mkbufs(torch, gpu, nbuf)
    te = TransferEngine(); assert te.initialize(IP, "P2PHANDSHAKE", "rdma", "") == 0
    for b in bufs: assert te.register_memory(b.data_ptr(), b.numel()) == 0
    q.put((gpu, f"{IP}:{te.get_rpc_port()}", [b.data_ptr() for b in bufs], per))
    if os.environ.get("YX_LOADSIDE") == "recv":
        a = torch.randn(8192, 8192, device=f"cuda:{gpu}", dtype=torch.bfloat16)
        while not done.is_set():
            for _ in range(20): a = (a @ a).clamp_(-1, 1)
            torch.cuda.synchronize(gpu)
    done.wait()

def send(gpu, sess, dptrs, per, nbuf, load, barrier, out):
    import torch
    from mooncake.engine import TransferEngine
    torch.cuda.set_device(gpu)
    bufs, _ = mkbufs(torch, gpu, nbuf)
    te = TransferEngine(); assert te.initialize(IP, "P2PHANDSHAKE", "rdma", "") == 0
    for b in bufs: assert te.register_memory(b.data_ptr(), b.numel()) == 0
    nper = per // BLK; n = TOTAL // BLK
    src = []; dst = []
    for i in range(n):  # round-robin over buffers, like per-layer KV pools
        k, j = i % nbuf, (i // nbuf) % nper
        src.append(bufs[k].data_ptr() + j * BLK); dst.append(dptrs[k] + j * BLK)
    lens = [BLK] * n
    stop = threading.Event()
    def burn():
        a = torch.randn(8192, 8192, device=f"cuda:{gpu}", dtype=torch.bfloat16); s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            while not stop.is_set():
                for _ in range(20): a = (a @ a).clamp_(-1, 1)
                s.synchronize()
    th = threading.Thread(target=burn) if (load and os.environ.get("YX_LOADSIDE", "send") == "send") else None
    barrier.wait()
    if th: th.start(); time.sleep(2)
    t_build = time.time(); _ = [x for x in src]; t_build = time.time() - t_build
    ts = []
    for r in range(5):
        t = time.time(); rc = te.batch_transfer_sync_write(sess, src, dst, lens); ts.append((rc, time.time() - t))
    if th: stop.set(); th.join()
    out.put({"gpu": gpu, "nbuf": nbuf, "load": load, "blocks": n, "rc_bad": sum(1 for rc, _ in ts if rc != 0),
             "call_s": [round(d, 3) for _, d in ts], "GBps_mean": round(n * BLK * len(ts) / sum(d for _, d in ts) / 1e9, 2)})

if __name__ == "__main__":
    mp.set_start_method("spawn")
    for pairs, nbuf, load in [(4, 61, 1), (2, 61, 1)]:
        q, out, done = mp.Queue(), mp.Queue(), mp.Event()
        rs = [mp.Process(target=recv, args=(g + 4, nbuf, q, done)) for g in range(pairs)]; [p.start() for p in rs]
        info = {}
        for _ in range(pairs):
            g, s, d, per = q.get(timeout=180); info[g - 4] = (s, d, per)
        bar = mp.Barrier(pairs)
        ss = [mp.Process(target=send, args=(g, info[g][0], info[g][1], info[g][2], nbuf, load, bar, out)) for g in range(pairs)]
        [p.start() for p in ss]
        res = [out.get(timeout=600) for _ in range(pairs)]
        [p.join() for p in ss]; done.set(); [p.join() for p in rs]
        for x in res:
            x["pairs"] = pairs; print(json.dumps(x), flush=True)
