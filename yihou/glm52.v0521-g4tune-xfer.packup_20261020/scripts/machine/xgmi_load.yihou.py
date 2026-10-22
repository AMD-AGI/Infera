# Task g4tune 2026-10-20: is GPU->GPU XGMI copy itself slow while the SENDER GPU computes, or only Mooncake's path?
# Pairs GPU g -> GPU g+4 (g=0..3, prefill->decode as in production), all pairs at once, one process per pair.
# Per pair, with and without a bf16 matmul burn on the sender GPU (same burn as mc_bw2), time:
#   push_1   : one 2 GiB copy, stream on the sender device
#   push_blk : 2 GiB as 5825 x 360 KiB copies, stream on the sender device (production block size)
#   pull_blk : same blocks, stream on the receiver device
import os, time, json, threading
import multiprocessing as mp
BLK = 368640; N = 5825; TOTAL = BLK * N


def pair(g, load, bar, out):
    import torch
    s, d = g, g + 4
    a = torch.empty(TOTAL, dtype=torch.uint8, device=f"cuda:{s}")
    b = torch.empty(TOTAL, dtype=torch.uint8, device=f"cuda:{d}")

    def run(kind):
        dev = d if kind == "pull_blk" else s
        st = torch.cuda.Stream(device=dev)
        torch.cuda.synchronize(s); torch.cuda.synchronize(d)
        t = time.time()
        with torch.cuda.device(dev), torch.cuda.stream(st):
            if kind == "push_1":
                b.copy_(a, non_blocking=True)
            else:
                for i in range(N):
                    b[i * BLK:(i + 1) * BLK].copy_(a[i * BLK:(i + 1) * BLK], non_blocking=True)
        st.synchronize()
        return time.time() - t

    stop = threading.Event()

    def burn():
        x = torch.randn(8192, 8192, device=f"cuda:{s}", dtype=torch.bfloat16)
        bs = torch.cuda.Stream(device=s)
        with torch.cuda.device(s), torch.cuda.stream(bs):
            while not stop.is_set():
                for _ in range(20):
                    x = (x @ x).clamp_(-1, 1)
                bs.synchronize()

    for kind in ("push_1", "push_blk", "pull_blk"):
        run(kind)  # warm-up
    th = threading.Thread(target=burn) if load else None
    bar.wait()
    if th:
        th.start(); time.sleep(2)
    res = {}
    for kind in ("push_1", "push_blk", "pull_blk"):
        ts = [run(kind) for _ in range(3)]
        res[kind] = {"s": [round(x, 3) for x in ts], "GBps": round(TOTAL * 3 / sum(ts) / 1e9, 2)}
    if th:
        stop.set(); th.join()
    out.put({"pair": f"{s}->{d}", "load": load, **res})


if __name__ == "__main__":
    mp.set_start_method("spawn")
    for load in (0, 1):
        out, bar = mp.Queue(), mp.Barrier(4)
        ps = [mp.Process(target=pair, args=(g, load, bar, out)) for g in range(4)]
        [p.start() for p in ps]
        rs = sorted([out.get(timeout=900) for _ in range(4)], key=lambda r: r["pair"])
        [p.join() for p in ps]
        for r in rs:
            print(json.dumps(r), flush=True)
