#!/usr/bin/env python3
"""GLM-5.2 MXFP4 routed MoE (257 experts incl. the fused shared one, top-9, hidden 6144) as sglang calls aiter's
fused_moe, timed per call under CUDA-graph replay. One graph runs the op on NSETS weight sets back to back, like
consecutive layers, so no call finds its weights in L2 / MALL.

Arms: `auto` = aiter's own tuned-table row (get_padded_M = nextPow2(M) below 32768); `kNNN` = force the token=NNN
row of the same table; `kf` = the yihou KF overlay entry (glm52_moe_mxfp4; run it in its own process, since
importing kf_moe sets aiter gemm2 code-gen env defaults).

Usage (in the production image): moe_bench.py --inter 256 --ms 384,512,704 --arms auto,k512,k256 [--out f.json]
"""
import argparse
import collections
import json
import os
import statistics as st

import torch

E, TOPK, H = 257, 9, 6144


def mxfp4_bytes(n):
    return n * 0.5 * (1 + 1 / 16)


def make_weights(inter, seed):
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randint(0, 256, (E, 2 * inter, H // 2), dtype=torch.uint8, device="cuda", generator=g)
    w2 = torch.randint(0, 256, (E, H, inter // 2), dtype=torch.uint8, device="cuda", generator=g)
    s13 = torch.randint(119, 124, (E, 2 * inter, H // 32), dtype=torch.uint8, device="cuda", generator=g)
    s2 = torch.randint(119, 124, (E, H, inter // 32), dtype=torch.uint8, device="cuda", generator=g)
    s13 = e8m0_shuffle(s13.reshape(E * 2 * inter, -1)).view(E, 2 * inter, -1)
    s2 = e8m0_shuffle(s2.reshape(E * H, -1)).view(E, H, -1)
    w13 = shuffle_weight(w13.contiguous(), (16, 16)).view(torch.float4_e2m1fn_x2)
    w2 = shuffle_weight(w2.contiguous(), (16, 16)).view(torch.float4_e2m1fn_x2)
    w13.is_shuffled = True
    w2.is_shuffled = True
    return w13, w2, s13, s2


def make_inputs(M, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda", generator=g)
    routed = torch.rand(M, 256, device="cuda", generator=g).argsort(dim=1)[:, :8].to(torch.int32)
    ids = torch.cat([routed, torch.full((M, 1), 256, dtype=torch.int32, device="cuda")], dim=1).contiguous()
    w = torch.rand(M, TOPK, device="cuda", generator=g).float()
    w[:, :8] = w[:, :8] / w[:, :8].sum(dim=1, keepdim=True) * 2.5
    w[:, 8] = 1.0
    return x, ids, w.contiguous()


def aiter_call(x, ws, ids, tw):
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType
    from aiter.ops.flydsl.moe_common import GateMode

    w13, w2, s13, s2 = ws
    return fm.fused_moe(x, w13, w2, tw, ids, expert_mask=None, activation=ActivationType.Silu,
                        quant_type=QuantType.per_1x32, doweight_stage1=False, w1_scale=s13, w2_scale=s2,
                        a1_scale=None, a2_scale=None, bias1=None, bias2=None, hidden_pad=0, intermediate_pad=0,
                        gate_mode=GateMode.SEPARATED.value, moe_sorting_dispatch_policy=2)


def set_key(arm):
    import aiter.fused_moe as fm

    orig = getattr(fm.get_padded_M, "_kopt_orig", fm.get_padded_M)
    if arm.startswith("k"):
        key = int(arm[1:])
        f = lambda M, _k=key: _k  # noqa: E731
    else:
        f = orig
    f._kopt_orig = orig
    fm.get_padded_M = f


def snr(ref, x):
    ref, x = ref.float(), x.float()
    n = torch.mean((x - ref) ** 2).item()
    return 100.0 if n == 0 else 10 * torch.log10(torch.mean(ref ** 2) / n).item()


def bench_arm(arm, M, inter, wsets, reps, kf=None):
    x, ids, tw = make_inputs(M, 1234 + M)
    if arm == "kf":
        call = lambda ws: kf.glm52_moe_mxfp4(x, ws[0], ws[1], ws[2], ws[3], tw, ids)  # noqa: E731
    else:
        set_key(arm)
        call = lambda ws: aiter_call(x, ws, ids, tw)  # noqa: E731
    out = call(wsets[0])
    for ws in wsets:
        call(ws)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for ws in wsets:
            call(ws)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [call(ws) for ws in wsets]
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / len(wsets))
    # Per-kernel breakdown of a few replays.
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(4):
            g.replay()
        torch.cuda.synchronize()
    per = collections.Counter()
    for e in prof.events():
        if e.device_type.name == "CUDA":
            per[e.name] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
    calls = 4 * len(wsets)
    kernels = [(k[:110], v / calls) for k, v in per.most_common(12)]
    nk = sum(1 for e in prof.events() if e.device_type.name == "CUDA") / calls
    ref = outs[0].clone()
    del g
    return {"arm": arm, "M": M, "us_median": st.median(times), "us_min": min(times),
            "kernels_us": kernels, "kernel_sum_us": sum(per.values()) / calls, "kernels_per_call": nk, "out": ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", type=int, default=256)
    ap.add_argument("--ms", default="384,448,512,576,640,704,768,832,896,1024")
    ap.add_argument("--arms", default="auto,k128,k256,k512,k1024")
    ap.add_argument("--nsets", type=int, default=4)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out")
    a = ap.parse_args()
    kf = None
    if "kf" in a.arms.split(","):
        import importlib.util

        spec = importlib.util.spec_from_file_location("kf_moe", "/yihou_overlay/yihou_kf/kf_moe.py")
        kf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kf)
    wsets = [make_weights(a.inter, 7 + i) for i in range(a.nsets)]
    gb = (mxfp4_bytes(E * 2 * a.inter * H) + mxfp4_bytes(E * H * a.inter)) / 1e9
    print(f"inter {a.inter}: weights {gb * 1e3:.1f} MB per call, {a.nsets} sets", flush=True)
    rows = []
    for M in [int(m) for m in a.ms.split(",")]:
        ref = None
        for arm in a.arms.split(","):
            try:
                r = bench_arm(arm, M, a.inter, wsets, a.reps, kf)
            except Exception as e:  # noqa: BLE001
                print(f"M {M:5d} {arm:6s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            out = r.pop("out")
            if ref is None:
                ref = out
            r["snr_vs_first_db"] = round(snr(ref, out), 1)
            r["tbps"] = round(gb * 1e3 / r["us_median"], 2)
            rows.append(r)
            ks = "; ".join(f"{k.split('(')[0][:60]} {v:.1f}" for k, v in r["kernels_us"][:6])
            print(f"M {M:5d} {arm:6s} {r['us_median']:7.1f} us ({r['tbps']:.2f} TB/s) snr {r['snr_vs_first_db']:5.1f} ksum {r['kernel_sum_us']:6.1f} n {r['kernels_per_call']:.1f} | {ks}",
                  flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
