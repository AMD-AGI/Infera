# Task g4tune 2026-10-19: Mooncake TransferEngine microbenchmark on ONE node, production env (HIP path, P2PHANDSHAKE).
# Pairs: sender GPU s -> receiver GPU s+4, one process per GPU. Each sender times batch_transfer_sync_write calls
# for several block shapes, alone (pairs=1) and with all 4 pairs at once (pairs=4), and with T threads per sender.
import os, sys, time, json, threading
import multiprocessing as mp
BLK = int(os.environ.get("YX_BLK", 368640))          # ~360 KiB, the mean block size seen in e2e/e2f logs
TOTAL = int(os.environ.get("YX_TOTAL_MB", 2048)) << 20  # bytes per call
IP = os.environ["YX_IP"]

def recv(gpu, q, done):
    import torch
    from mooncake.engine import TransferEngine
    torch.cuda.set_device(gpu)
    buf = torch.empty(TOTAL * 4, dtype=torch.uint8, device=f"cuda:{gpu}")
    te = TransferEngine(); assert te.initialize(IP, "P2PHANDSHAKE", "rdma", "") == 0
    assert te.register_memory(buf.data_ptr(), buf.numel()) == 0
    q.put((gpu, f"{IP}:{te.get_rpc_port()}", buf.data_ptr()))
    done.wait()

def send(gpu, sess, dst, cfgs, threads, barrier, out):
    import torch
    from mooncake.engine import TransferEngine
    torch.cuda.set_device(gpu)
    buf = torch.empty(TOTAL * 4, dtype=torch.uint8, device=f"cuda:{gpu}")
    te = TransferEngine(); assert te.initialize(IP, "P2PHANDSHAKE", "rdma", "") == 0
    assert te.register_memory(buf.data_ptr(), buf.numel()) == 0
    res = []
    for name, blk in cfgs:
        n = TOTAL // blk
        def work(k, times):
            base = k * TOTAL
            src = [buf.data_ptr() + base + i * blk for i in range(n)]
            dsts = [dst + base + i * blk for i in range(n)]
            lens = [blk] * n
            for r in range(3):
                t = time.time(); rc = te.batch_transfer_sync_write(sess, src, dsts, lens); dt = time.time() - t
                times.append((rc, dt))
        barrier.wait()
        tl = [[] for _ in range(threads)]
        th = [threading.Thread(target=work, args=(k, tl[k])) for k in range(threads)]
        t0 = time.time(); [x.start() for x in th]; [x.join() for x in th]; wall = time.time() - t0
        calls = [c for t in tl for c in t]
        res.append({"gpu": gpu, "cfg": name, "blk": blk, "blocks_per_call": n, "threads": threads,
                    "rc_bad": sum(1 for rc, _ in calls if rc != 0), "call_s_mean": round(sum(d for _, d in calls) / len(calls), 4),
                    "agg_GBps": round(threads * 3 * n * blk / wall / 1e9, 2)})
    out.put(res)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    cfgs = [("1x2GiB", TOTAL), ("4MiB", 4 << 20), ("360KiB", BLK), ("64KiB", 65536)]
    for pairs, threads in [(1, 1), (1, 4), (4, 1), (4, 4)]:
        q, out, done = mp.Queue(), mp.Queue(), mp.Event()
        rs = [mp.Process(target=recv, args=(g + 4, q, done)) for g in range(pairs)]; [p.start() for p in rs]
        info = {}
        for _ in range(pairs):
            g, s, d = q.get(timeout=120); info[g - 4] = (s, d)
        bar = mp.Barrier(pairs)
        ss = [mp.Process(target=send, args=(g, info[g][0], info[g][1], cfgs, threads, bar, out)) for g in range(pairs)]
        [p.start() for p in ss]
        allr = [out.get(timeout=600) for _ in range(pairs)]
        [p.join() for p in ss]; done.set(); [p.join() for p in rs]
        for r in allr:
            for x in r:
                x["pairs"] = pairs; print(json.dumps(x), flush=True)
