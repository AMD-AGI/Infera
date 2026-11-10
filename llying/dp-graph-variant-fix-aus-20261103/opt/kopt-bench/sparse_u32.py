#!/usr/bin/env python3
"""KF sparse MLA with unsigned 32-bit page offsets (kf_sparse_mla_u32.py) against the production copy: time at the TP4
and TP8 pool sizes, and the output against the production kernel's (64-bit offsets above 2^31 B) with indices over the
whole pool, so pages past 2^31 B are read."""
import importlib.util
import sys

import torch

sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import sparse_bench as sb  # noqa: E402
import sparse_probe as sp  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


orig = load("/yihou_overlay/yihou_kf/kf_sparse_mla.py", "kf_orig")
u32 = load("/perf_apps/liyingli/bench_agentx/kopt-20261109/bench/kf_sparse_mla_u32.py", "kf_u32")
g = torch.Generator(device="cuda").manual_seed(0)
FP8 = torch.float8_e4m3fn
for pool in (3093376, 4094720):
    kv = (torch.randn(pool, 1, sb.DV + sb.DR, device="cuda", dtype=torch.bfloat16, generator=g) * 0.5).to(FP8)
    for rows in (72, 88, 96):
        qn = (torch.randn(rows, sb.H, sb.DV, device="cuda", generator=g) * 0.5).to(FP8)
        qr = (torch.randn(rows, sb.H, sb.DR, device="cuda", generator=g) * 0.5).to(FP8)
        idx = [sb.make_indices(rows, pool, "uniform", g) for _ in range(8)]
        ro = orig.triton_sparse_mla_fwd(qn, qr, kv, idx[0], 0.0625, sb.DV).float()
        ru = u32.triton_sparse_mla_fwd(qn, qr, kv, idx[0], 0.0625, sb.DV).float()
        diff = (ro - ru).abs().max().item()
        print(f"pool {pool:8d} rows {rows:3d}: production {sp.timed(orig, qn, qr, kv, idx):5.1f} us, "
              f"u32 {sp.timed(u32, qn, qr, kv, idx):5.1f} us, max |diff| {diff:.3g}", flush=True)
    del kv
    torch.cuda.empty_cache()
