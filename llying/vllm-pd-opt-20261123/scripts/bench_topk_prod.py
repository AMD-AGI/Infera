#!/usr/bin/env python3
"""GLM-5.2 DSA indexer decode top-k (k = 2048) on production-like rows (1 GPU, in the vLLM image).

The a8 trace has aiter's one-block-per-row `radix_topk_one_block_decode` at 143.6 us per call, 22 calls per step
(~3.2 ms of a 48 ms step); the A-line bench (uniform 40K-200K rows) had it at ~52 us for 64 rows. In P4D4 AgentX the
longest request on a rank is often 500K-800K tokens (a8 steplog: about half the steps on rank 0), and a one-block
kernel takes as long as its longest row. Scenarios: requests with lognormal context (median 86K, p90 267K: the AgentX
ISL distribution), 4 rows each (MTP3: lengths L-3..L), plus fixed long requests.

Candidates: aiter (what v0.34 runs), vllm_c (torch.ops._C.top_k_per_row_decode), triton (sglang #47872 port from the
A-line: est / chunked collect over a persistent grid / select; exact). Exactness check as in the A-line bench.
  bench_topk_prod.py TRITON_DIR [VPD_DIR]   env DIST (randn|relu|zsparse|zmid|mixed), ITERS (30), CANDS
DIST zsparse: all 0 except 0.15% positive (a 700K row has ~1,050 > 0, so the k-th is a 0 tied ~700K times: the a11
failure); zmid: 1% positive; mixed: randn with half the entries set to 0. triton_v = the tie-guarded copy in
VPD_DIR (vpd_triton_topk_decode.py), triton = the A-line port.
"""
import math
import os
import sys

import torch

sys.path.insert(0, sys.argv[1])
if len(sys.argv) > 2:
    sys.path.insert(0, sys.argv[2])
TOPK, NEXT_N, W = 2048, 4, 1048576
ITERS = int(os.environ.get("ITERS", 30))
DIST = os.environ.get("DIST", "randn")
dev = torch.device("cuda")

SCEN = [  # name, number of lognormal requests, fixed long requests, seed
    ("16 reqs lognormal", 16, [], 1),
    ("16 reqs lognormal + one 700K", 15, [700_000], 2),
    ("20 reqs lognormal + 600K, 800K", 18, [600_000, 800_000], 3),
    ("12 reqs lognormal, none > 300K", 12, [], 4),
    ("24 reqs lognormal + one 750K", 23, [750_000], 5),
]


def ctx_list(n, fixed, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, generator=g)
    c = [int(min(max(86_000 * math.exp(0.884 * float(x)), 4_000), 900_000)) for x in z]
    if seed == 4:
        c = [min(x, 300_000) for x in c]
    return c + fixed


def make(ctx, seed):
    lens = torch.tensor([L - (NEXT_N - 1) + j for L in ctx for j in range(NEXT_N)], dtype=torch.int32, device=dev)
    rows = lens.numel()
    logits = torch.full((rows, W), float("-inf"), device=dev)
    gd = torch.Generator(device=dev).manual_seed(seed)
    for r in range(rows):
        n = int(lens[r])
        x = torch.randn(n, device=dev, generator=gd)
        if DIST == "relu":
            x = torch.relu(x) * (1 + torch.rand(n, device=dev, generator=gd)) + 0.01 * x
        elif DIST in ("zsparse", "zmid"):
            frac = 0.0015 if DIST == "zsparse" else 0.01
            x = torch.where(torch.rand(n, device=dev, generator=gd) < frac, x.abs() + 0.1, torch.zeros_like(x))
        elif DIST == "mixed":
            x = torch.where(torch.rand(n, device=dev, generator=gd) < 0.5, torch.zeros_like(x), x)
        logits[r, :n] = x
    return logits, lens


def check(logits, lens, out):
    bad = 0
    for r in range(logits.shape[0]):
        n = int(lens[r]); k = min(n, TOPK)
        sel, pad = out[r, :k], out[r, k:TOPK]
        if (pad != -1).any() or (sel < 0).any() or (sel >= n).any() or sel.unique().numel() != k:
            bad += 1; continue
        kth = torch.topk(logits[r, :n], k).values[-1]
        bad += int((logits[r, sel.long()] < kth).any())
    return bad


def main():
    want = os.environ.get("CANDS", "aiter vllm_c triton triton_v").split()
    cands = {}
    if "aiter" in want:
        from aiter.ops.topk import top_k_per_row_decode as at
        cands["aiter"] = lambda lg, ln, o: at(lg, 1, ln, o, lg.shape[0], lg.stride(0), lg.stride(1), k=TOPK)
    if "vllm_c" in want:
        import vllm._custom_ops  # noqa: F401
        cands["vllm_c"] = lambda lg, ln, o: torch.ops._C.top_k_per_row_decode(lg, 1, ln, o, lg.shape[0], lg.stride(0), lg.stride(1), TOPK)
    if "triton" in want:
        import vllm_triton_topk_decode as tk
        cands["triton"] = lambda lg, ln, o: tk.triton_top_k_per_row_decode(lg, ln, o, TOPK)
    if "triton_v" in want and len(sys.argv) > 2:
        import vpd_triton_topk_decode as tv
        cands["triton_v"] = lambda lg, ln, o: tv.triton_top_k_per_row_decode(lg, ln, o, TOPK)
    print(f"device {torch.cuda.get_device_properties(0).gcnArchName}; dist {DIST}; CUDA-graph replay timing, {ITERS} calls", flush=True)
    print("scenario | rows | max / mean ctx | " + " | ".join(f"{n} us (bad)" for n in cands), flush=True)
    for name, n, fixed, seed in SCEN:
        ctx = ctx_list(n, fixed, seed)
        logits, lens = make(ctx, seed)
        rows = lens.numel()
        out = torch.empty((rows, TOPK), dtype=torch.int32, device=dev)
        cells = []
        for cn, fn in cands.items():
            try:
                out.fill_(-7); fn(logits, lens, out); torch.cuda.synchronize()
                bad = check(logits, lens, out)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(ITERS):
                        fn(logits, lens, out)
                g.replay(); torch.cuda.synchronize()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(5):
                    g.replay()
                e.record(); torch.cuda.synchronize()
                cells.append(f"{s.elapsed_time(e) * 1000 / (5 * ITERS):.1f} ({bad})")
            except Exception as ex:
                cells.append(f"error {type(ex).__name__}: {str(ex)[:100]}")
        print(f"{name} | {rows} | {max(ctx) // 1000}K / {sum(ctx) // len(ctx) // 1000}K | " + " | ".join(cells), flush=True)
        del logits
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
