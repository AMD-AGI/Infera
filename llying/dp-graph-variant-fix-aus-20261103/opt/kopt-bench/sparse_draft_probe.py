#!/usr/bin/env python3
"""Which KF sparse MLA kernel each draft-like call runs, and whether it has the > 2 GiB cliff: triton_sparse_mla_fwd at
the TP4 (3.09M) and TP8 (4.09M) pool sizes, fp8 and bf16 KV, 8-96 rows. Kernel names from torch.profiler, time under
CUDA graph (8 index sets).

Usage (production image, overlay at /yihou_overlay): python3 sparse_draft_probe.py
"""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import sparse_bench as sb  # noqa: E402
import sparse_probe as sp  # noqa: E402


def kernels(kf, qn, qr, kv, idx):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        kf.triton_sparse_mla_fwd(qn, qr, kv, idx, 0.0625, sb.DV)
        torch.cuda.synchronize()
    names = [e.key for e in p.key_averages() if e.device_type.name == "CUDA" and "sparse" in e.key.lower()]
    return ",".join(sorted(set(n[:40] for n in names)))


def main():
    kf = sb.load_kf()
    g = torch.Generator(device="cuda").manual_seed(0)
    for dtype in (torch.float8_e4m3fn, torch.bfloat16):
        for pool in (3093376, 4094720):
            kv = (torch.randn(pool, 1, sb.DV + sb.DR, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5).to(dtype)
            gib = pool * (sb.DV + sb.DR) * kv.element_size() / 2**30
            for rows in (8, 24, 48, 96):
                qn = (torch.randn(rows, sb.H, sb.DV, device="cuda", generator=g) * 0.5).to(dtype)
                qr = (torch.randn(rows, sb.H, sb.DR, device="cuda", generator=g) * 0.5).to(dtype)
                idx = [sb.make_indices(rows, pool, "request", g) for _ in range(8)]
                try:
                    names = kernels(kf, qn, qr, kv, idx[0])
                    t = sp.timed(kf, qn, qr, kv, idx)
                    print(f"{str(dtype)[6:]:14s} pool {pool} ({gib:.2f} GiB) rows {rows:3d}: {t:7.1f} us  {names}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{str(dtype)[6:]:14s} pool {pool} rows {rows}: ERROR {type(e).__name__}: {str(e)[:150]}", flush=True)
            del kv
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
