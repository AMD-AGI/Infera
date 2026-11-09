#!/usr/bin/env python3
"""KF indexer logits (kf_indexer.deepgemm_fp8_paged_mqa_logits, what production verify runs) on K3 verify rows:
each request contributes 4 consecutive rows that share one page table (contexts L+1 .. L+4). Arms:
  prod     _LPT_GROUP_ROWS 6, _LPT_NG2 16 (fused2 only when rows % 6 == 0 and rows <= 96, else preshuffle)
  k3       _LPT_GROUP_ROWS 4, _LPT_NG2 32 (fused2 for every K3 row count up to 128)
  preshuf  _FUSE 0, _LPT 0 (the unfused kernel; correctness reference)
Timed under CUDA-graph replay over NSETS page-table sets.

Usage (production image): indexer_fused2_bench.py --rows 64,72,80,88,96,104,112,128
"""
import argparse
import importlib.util
import statistics as st

import torch

HEADS, DIM, PAGE = 32, 128, 64


def load():
    spec = importlib.util.spec_from_file_location("kf_indexer", "/yihou_overlay/yihou_kf/kf_indexer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_case(rows, n_pages, max_len, g):
    reqs = rows // 4
    ctx = torch.randint(30000, 150000, (reqs,), generator=g, device="cuda")
    seq = (ctx[:, None] + torch.arange(1, 5, device="cuda")[None, :]).flatten().to(torch.int32)
    max_blocks = (max_len + PAGE - 1) // PAGE
    bt = torch.zeros(rows, max_blocks, dtype=torch.int32, device="cuda")
    for r in range(reqs):
        nb = (int(ctx[r]) + 4 + PAGE - 1) // PAGE
        pages = torch.randperm(n_pages, generator=g, device="cuda")[:nb].to(torch.int32)
        bt[4 * r:4 * r + 4, :nb] = pages
    q = (torch.randn(rows, HEADS, DIM, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    w = torch.rand(rows, HEADS, generator=g, device="cuda")
    return q, w, seq, bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="64,72,80,88,96,104,112,128")
    ap.add_argument("--pages", type=int, default=40000)
    ap.add_argument("--nsets", type=int, default=4)
    a = ap.parse_args()
    ki = load()
    g = torch.Generator(device="cuda").manual_seed(0)
    max_len = 150008
    vals = (torch.randn(a.pages, PAGE, DIM, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    scales = torch.rand(a.pages, PAGE, generator=g, device="cuda") + 0.5
    kv = torch.empty(a.pages, PAGE * (DIM + 4), dtype=torch.uint8, device="cuda")
    kv[:, :PAGE * DIM] = vals.view(torch.uint8).view(a.pages, -1)
    kv[:, PAGE * DIM:] = scales.view(torch.uint8).view(a.pages, -1)
    kv = kv.view(a.pages, PAGE, 1, DIM + 4)
    arms = {"prod": (6, 16, 1, 1), "k3": (4, 32, 1, 1), "preshuf": (6, 16, 0, 0)}

    def set_arm(name):
        grp, ng2, fuse, lpt = arms[name]
        ki._LPT_GROUP_ROWS, ki._LPT_NG2, ki._FUSE, ki._LPT = grp, ng2, fuse, lpt

    for rows in [int(r) for r in a.rows.split(",")]:
        cases = [make_case(rows, a.pages, max_len, g) for _ in range(a.nsets)]
        outs = {}
        times = {}
        for name in arms:
            set_arm(name)

            def run():
                return [ki.aiter_paged_mqa_logits(q, kv, w, seq, bt, max_len, preshuffle=True, kv_block_size=PAGE)
                        for q, w, seq, bt in cases]

            outs[name] = run()[0].clone()
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                run()
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=s):
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
            times[name] = st.median(t)
            del gr
        seq = cases[0][2]
        valid = torch.arange(max_len, device="cuda")[None, :] < seq[:, None]
        ref = outs["preshuf"][valid]
        diffs = {n: ((outs[n][valid] - ref).abs().max() / ref.abs().max()).item() for n in ("prod", "k3")}
        print(f"rows {rows:4d}: prod {times['prod']:6.1f} us  k3 {times['k3']:6.1f} us  preshuf {times['preshuf']:6.1f} us  "
              f"| max rel diff vs preshuf: prod {diffs['prod']:.1e}, k3 {diffs['k3']:.1e}", flush=True)


if __name__ == "__main__":
    main()
