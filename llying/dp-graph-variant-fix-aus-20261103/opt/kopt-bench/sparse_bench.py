#!/usr/bin/env python3
"""KF sparse-MLA decode kernel (yihou overlay kf_sparse_mla.triton_sparse_mla_fwd, what production verify runs) on
GLM-5.2 shapes: rows = verify tokens on one DP rank, 64 heads, q fp8 (512 + 64), kv fp8 [pool, 1, 576], top-k 2048.
Timed under CUDA-graph replay over NSETS index sets. Index modes: `uniform` (every row its own random slots) and
`request` (the 4 verify rows of a request share 90% of their slots, drawn from that request's pages).

Usage (production image): sparse_bench.py --rows 64,72,88,96 --pools 3093376,4094720
"""
import argparse
import importlib.util
import statistics as st

import torch

H, DV, DR, TOPK, PAGE = 64, 512, 64, 2048, 64
FP8 = torch.float8_e4m3fn


def load_kf():
    spec = importlib.util.spec_from_file_location("kf_sparse_mla", "/yihou_overlay/yihou_kf/kf_sparse_mla.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_indices(rows, pool, mode, g):
    if mode == "uniform":
        return torch.randint(0, pool, (rows, 1, TOPK), device="cuda", dtype=torch.int32, generator=g)
    reqs = (rows + 3) // 4
    out = []
    for _ in range(reqs):
        pages = torch.randint(0, pool // PAGE, (2 * TOPK // PAGE + 64,), device="cuda", generator=g)
        slots = (pages[:, None] * PAGE + torch.arange(PAGE, device="cuda")).flatten()
        base = slots[torch.randperm(slots.numel(), device="cuda", generator=g)[:TOPK]]
        for _ in range(4):
            r = base.clone()
            k = TOPK // 10
            r[torch.randperm(TOPK, device="cuda", generator=g)[:k]] = slots[
                torch.randint(0, slots.numel(), (k,), device="cuda", generator=g)]
            out.append(r)
    return torch.stack(out[:rows]).view(rows, 1, TOPK).to(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="64,72,80,84,88,96,104,112,128")
    ap.add_argument("--pools", default="3093376,4094720")
    ap.add_argument("--modes", default="request,uniform")
    ap.add_argument("--nsets", type=int, default=8)
    a = ap.parse_args()
    kf = load_kf()
    g = torch.Generator(device="cuda").manual_seed(0)
    for pool in [int(p) for p in a.pools.split(",")]:
        kv = (torch.randn(pool, 1, DV + DR, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5).to(FP8)
        for mode in a.modes.split(","):
            for rows in [int(r) for r in a.rows.split(",")]:
                qn = (torch.randn(rows, H, DV, device="cuda", generator=g) * 0.5).to(FP8)
                qr = (torch.randn(rows, H, DR, device="cuda", generator=g) * 0.5).to(FP8)
                idx = [make_indices(rows, pool, mode, g) for _ in range(a.nsets)]

                def run():
                    for i in idx:
                        kf.triton_sparse_mla_fwd(qn, qr, kv, i, 0.0625, DV)

                run()
                torch.cuda.synchronize()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    run()
                torch.cuda.current_stream().wait_stream(s)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    run()
                for _ in range(3):
                    gr.replay()
                torch.cuda.synchronize()
                t = []
                for _ in range(20):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    gr.replay()
                    e1.record()
                    e1.synchronize()
                    t.append(e0.elapsed_time(e1) * 1e3 / a.nsets)
                splits = kf._decode_splitk_splits(rows, TOPK, kf._sparse_mla_block_k(kv), kf._cu_count(qn.device))
                us = st.median(t)
                print(f"pool {pool:8d} {mode:8s} rows {rows:4d}: {us:6.1f} us/call  {us / rows:5.3f} us/row  "
                      f"splits {splits}", flush=True)
                del gr
        del kv


if __name__ == "__main__":
    main()
