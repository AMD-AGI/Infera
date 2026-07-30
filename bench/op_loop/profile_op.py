#!/usr/bin/env python3
"""Profile ring of the MoE-experts optimize loop (issue #40).

Roofline profile of an experts impl at a given shape: median latency, the
selected-expert weight traffic it must move, achieved effective HBM bandwidth,
%-of-peak, and a bottleneck verdict (bandwidth-bound ⇒ near optimal, cut traffic;
launch/occupancy-bound ⇒ headroom, tune tiling/warps). ``--kernels`` adds the
per-kernel device-time split (torch profiler) so you see which kernel to tune.

  python profile_op.py --impl infera_decode --tokens 1
  python profile_op.py --impl infera_decode --tokens 1 --kernels
  python profile_op.py --impl builtin --tokens 1        # profile the baseline
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from moe_experts_loop import build_inputs, timed  # noqa: E402


def weight_bytes(T, K, H, Dm, dbytes):
    # The decode kernel reads each (token, slot) expert's gate/up (2·Dm·H) + down
    # (H·Dm) weights once — the traffic floor for this regime.
    return T * K * (2 * Dm * H + H * Dm) * dbytes


def peak_gbps(override):
    if override:
        return override
    try:
        p = torch.cuda.get_device_properties(0)
        bw = p.memory_clock_rate * 1e3 * (p.memory_bus_width / 8) * 2 / 1e9
        if bw > 6000:  # trust only a plausibly-HBM3e-class figure
            return bw
    except Exception:  # noqa: BLE001
        pass
    return 8000.0  # MI355X HBM3e ~8 TB/s; override with --peak-gbps


def get_impl(name):
    from vllm.model_executor.layers.fused_moe import fused_experts as builtin

    if name in ("builtin", "baseline"):
        return builtin
    os.environ["INFERA_MOE_EXPERTS"] = name
    from infera.engine.vllm.ops.moe import infera_fused_experts

    return infera_fused_experts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="infera_decode")
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--peak-gbps", type=float, default=0.0)
    ap.add_argument("--kernels", action="store_true")
    args = ap.parse_args()

    dt = getattr(torch, args.dtype)
    E, H, Dm, K, T = args.experts, args.hidden, args.inter, args.topk, args.tokens
    x, w1, w2, tw, ti = build_inputs(E, H, Dm, K, T, dt, "cuda")
    impl = get_impl(args.impl)

    def call():
        return impl(x, w1, w2, tw, ti, global_num_experts=E)

    call()  # warm / JIT
    ms = timed(call)
    wb = weight_bytes(T, K, H, Dm, torch.finfo(dt).bits // 8)
    bw_gbps = (wb / 1e9) / (ms / 1e3)  # GB/s
    peak = peak_gbps(args.peak_gbps)  # GB/s
    pct = 100 * bw_gbps / peak
    verdict = (
        "bandwidth-bound — near peak; win by moving less traffic (dtype / dedup)"
        if pct >= 55
        else "launch/occupancy-bound — headroom; tune tiling/warps, cut launches"
    )
    print(f"impl={args.impl}  T={T} E={E} H={H} I={Dm} top_k={K} dtype={args.dtype}")
    print(f"  latency        : {ms:.4f} ms")
    print(f"  weight traffic : {wb / 1e6:.1f} MB (each selected expert read once)")
    print(
        f"  achieved BW    : {bw_gbps / 1e3:.2f} TB/s  ({pct:.0f}% of {peak / 1e3:.1f} TB/s peak)"
    )
    print(f"  bottleneck     : {verdict}")

    if args.kernels:
        from torch.profiler import ProfilerActivity, profile

        for _ in range(5):
            call()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                call()
            torch.cuda.synchronize()
        print("\n  per-kernel device time (top 6):")
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=6))


if __name__ == "__main__":
    main()
