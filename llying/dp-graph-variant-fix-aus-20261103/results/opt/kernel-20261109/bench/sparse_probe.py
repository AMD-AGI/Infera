#!/usr/bin/env python3
"""Two probes on the KF sparse-MLA decode kernel (see sparse_bench.py):
  splits: force the decode plan's KV-split count at 86-128 rows (production picks floor(num_cu / rows) = 2);
  span:   a 4.09M-slot pool with indices limited to its first N slots, against pools of N slots.
"""
import argparse
import statistics as st
import sys

import torch

sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import sparse_bench as sb  # noqa: E402

FP8 = torch.float8_e4m3fn


def timed(kf, qn, qr, kv, idx):
    def run():
        for i in idx:
            kf.triton_sparse_mla_fwd(qn, qr, kv, i, 0.0625, sb.DV)

    run()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t = []
    for _ in range(20):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        t.append(a.elapsed_time(b) * 1e3 / len(idx))
    return st.median(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=("splits", "span"), required=True)
    a = ap.parse_args()
    kf = sb.load_kf()
    g = torch.Generator(device="cuda").manual_seed(0)
    big = 4094720
    kv = (torch.randn(big, 1, sb.DV + sb.DR, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5).to(FP8)
    if a.probe == "splits":
        orig = kf._decode_splitk_splits
        for rows in (88, 96, 104, 112, 128):
            qn = (torch.randn(rows, sb.H, sb.DV, device="cuda", generator=g) * 0.5).to(FP8)
            qr = (torch.randn(rows, sb.H, sb.DR, device="cuda", generator=g) * 0.5).to(FP8)
            idx = [sb.make_indices(rows, big, "request", g) for _ in range(8)]
            ref = kf.triton_sparse_mla_fwd(qn, qr, kv, idx[0], 0.0625, sb.DV).float()
            res = []
            for s in (2, 3, 4, 6, 8):
                kf._decode_splitk_splits = lambda seq, topk, bk, ncu, _s=s: _s
                out = kf.triton_sparse_mla_fwd(qn, qr, kv, idx[0], 0.0625, sb.DV).float()
                err = (out - ref).abs().max().item()
                res.append(f"s{s} {timed(kf, qn, qr, kv, idx):5.1f}us (maxdiff {err:.2g})")
            kf._decode_splitk_splits = orig
            print(f"rows {rows:4d}: " + "  ".join(res), flush=True)
    else:
        rows = 88
        qn = (torch.randn(rows, sb.H, sb.DV, device="cuda", generator=g) * 0.5).to(FP8)
        qr = (torch.randn(rows, sb.H, sb.DR, device="cuda", generator=g) * 0.5).to(FP8)
        for span in (524288, 1048576, 2097152, 3093376, 4094720):
            idx = [sb.make_indices(rows, span, "request", g) for _ in range(8)]
            t_big = timed(kf, qn, qr, kv, idx)
            small = kv[:span].clone()
            t_small = timed(kf, qn, qr, small, idx)
            del small
            torch.cuda.empty_cache()
            print(f"span {span:8d} slots ({span * 576 / 2**30:4.2f} GiB): in 4.09M pool {t_big:5.1f} us, "
                  f"in its own pool {t_small:5.1f} us", flush=True)


if __name__ == "__main__":
    main()
