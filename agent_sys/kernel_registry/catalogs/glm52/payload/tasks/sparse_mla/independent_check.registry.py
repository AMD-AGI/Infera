#!/usr/bin/env python3
"""Independent check of a forge candidate vs the pristine sparse MLA kernel (not the forge driver).

Fresh random inputs the campaign never saw: indices sampled uniformly without replacement from
random per-row contexts inside the 2.2M-slot pool, new q/KV seeds. Reports, per T:
  SNR(candidate vs fp32 ref), SNR(pristine vs fp32 ref), SNR(candidate vs pristine),
  per-call ms of both (CUDA events, 8 distinct KV pools per measurement, median of 50).
Usage (in the container): python3 independent_check.registry.py <candidate_kernel.py> [--seed N]
"""
import argparse
import importlib.util
import math
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
H, D_V, D_ROPE, TOPK, POOL, SCALE, N_POOLS = 64, 512, 64, 2048, 34_787 * 64, 0.0625, 8


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.triton_sparse_mla_fwd


def snr(ref, x):
    ref, x = ref.float(), x.float()
    n = torch.mean((x - ref) ** 2).item()
    return 100.0 if n == 0 else 10 * math.log10(torch.mean(ref ** 2).item() / n)


def ref_out(q_nope, q_rope, kv, idx):
    kv2 = kv.squeeze(1)
    out = torch.empty(idx.shape[0], H, D_V, device=kv.device)
    for r in range(idx.shape[0]):
        k = kv2[idx[r, 0].long()].float()
        q = torch.cat([q_nope[r].float(), q_rope[r].float()], -1)
        out[r] = torch.softmax(q @ k.t() * SCALE, -1) @ k[:, :D_V]
    return out


def time_ms(fn, args_per_pool, iters=50):
    for a in args_per_pool:
        fn(*a)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for a in args_per_pool:
            fn(*a)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / len(args_per_pool))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    cand = load(a.candidate, "cand")
    prist = load(os.path.join(HERE, "sparse_mla_kernel.pristine.py"), "prist")
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(a.seed)
    pools = [torch.randn(POOL, 1, D_V + D_ROPE, device=dev, generator=g, dtype=torch.bfloat16)
             .clamp_(-4, 4).to(torch.float8_e4m3fn) for _ in range(N_POOLS)]
    for bs in (8, 10, 12, 14):
        t = 6 * bs
        rows = []
        for _ in range(t):                       # each row: random context window, 2048 distinct slots
            start = int(torch.randint(0, POOL - 600_000, (1,), generator=g, device=dev).item())
            length = int(torch.randint(4096, 600_000, (1,), generator=g, device=dev).item())
            rows.append(start + torch.randperm(length, generator=g, device=dev)[:TOPK])
        idx = torch.stack(rows).to(torch.int32).unsqueeze(1).contiguous()
        q_nope = torch.randn(t, H, D_V, device=dev, generator=g).clamp_(-4, 4).to(torch.float8_e4m3fn)
        q_rope = torch.randn(t, H, D_ROPE, device=dev, generator=g).clamp_(-4, 4).to(torch.float8_e4m3fn)
        call = lambda f: (lambda qn, qr, kv, ix: f(qn, qr, kv, ix, SCALE, d_v=D_V, topk_length=None, max_topk_length=TOPK))
        oc = call(cand)(q_nope, q_rope, pools[0], idx).reshape(t, H, D_V)
        op = call(prist)(q_nope, q_rope, pools[0], idx).reshape(t, H, D_V)
        r = ref_out(q_nope, q_rope, pools[0], idx)
        args = [(q_nope, q_rope, kv, idx) for kv in pools]
        tc, tp = time_ms(call(cand), args), time_ms(call(prist), args)
        print(f"T{t}_bs{bs}: snr cand/ref {snr(r, oc):.2f}  prist/ref {snr(r, op):.2f}  cand/prist {snr(op, oc):.2f}"
              f" | ms cand {tc:.4f} prist {tp:.4f} speedup {tp / tc:.3f}x", flush=True)


if __name__ == "__main__":
    sys.exit(main())
