#!/usr/bin/env python3
"""TP8 decode MoE (257 x i256, top-9, token=512 row of the tuned table = PP8 kopt) with the prologue variants aiter
already has, one variant per process (the sorting backend env vars are read at import):

  --policy P        moe_sorting_dispatch_policy passed to fused_moe (production: 2 = Opus multi-phase P0_v2 + P23);
                    with AITER_USE_FLYDSL_MOE_SORTING=1 (set in production), policy 0 selects the FlyDSL sort.
  --fused-quant     stage-1 activation quant + E8M0 scale sort as the single fused HIP kernel (aiter only uses it for
                    M <= 8 * 256 / topk = 227; production M 576-832 takes per_1x32 quant + mxfp4_moe_sort).

Prints us/call (CUDA-graph replay, NSETS weight sets back to back) and, with TOPN > 0, the per-kernel split (set
DEBUG_CLR_GRAPH_PACKET_CAPTURE=0, otherwise Kineto misattributes in-graph kernel times). --save / --ref write and
compare the outputs (SNR in dB; the stage-2 atomics make bit equality impossible across runs).

Usage (production image): prologue_bench.py --ms 576,704,832 --policy 2 [--fused-quant] [--save ref.pt | --ref ref.pt]
"""
import argparse
import collections
import os
import sys

import torch

sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import ep_moe_bench as eb  # noqa: E402
import moe_bench as mb  # noqa: E402


def force_fused_quant():
    import aiter.ops.quant as q
    from aiter import dtypes

    def fused(input, sorted_ids, num_valid_ids, token_num, topk, block_size, quant_dtype=dtypes.fp4x2,
              num_rows=None, group_size=32, sorted_weights=None, num_experts_upper_bound=None):
        M, N = input.view(-1, input.shape[-1]).shape
        out_cols = N // 2 if quant_dtype == dtypes.fp4x2 else N
        scale_n = ((N + group_size - 1) // group_size + 7) // 8 * 8
        scale = torch.empty((sorted_ids.shape[0] + 31) // 32 * 32, scale_n, dtype=dtypes.fp8_e8m0, device=input.device)
        out = torch.empty(M, out_cols, dtype=quant_dtype, device=input.device)
        common = (out, scale, input, sorted_ids, num_valid_ids, token_num, block_size)
        if num_experts_upper_bound is None:
            q.fused_dynamic_mx_quant_moe_sort_hip(*common, group_size, sorted_weights)
        else:
            q.fused_dynamic_mx_quant_moe_sort_hip_bounded(*common, token_num * topk, num_experts_upper_bound,
                                                          group_size, sorted_weights)
        return out, scale

    q.fused_dynamic_mx_quant_moe_sort = fused


def call(x, ws, ids, tw, policy):
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType
    from aiter.ops.flydsl.moe_common import GateMode

    w13, w2, s13, s2 = ws
    return fm.fused_moe(x, w13, w2, tw, ids, expert_mask=None, activation=ActivationType.Silu,
                        quant_type=QuantType.per_1x32, doweight_stage1=False, w1_scale=s13, w2_scale=s2,
                        a1_scale=None, a2_scale=None, bias1=None, bias2=None, hidden_pad=0, intermediate_pad=0,
                        gate_mode=GateMode.SEPARATED.value, moe_sorting_dispatch_policy=policy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="576,704,832")
    ap.add_argument("--policy", type=int, default=2)
    ap.add_argument("--fused-quant", action="store_true")
    ap.add_argument("--key", default="k512")
    ap.add_argument("--nsets", type=int, default=3)
    ap.add_argument("--save")
    ap.add_argument("--ref")
    a = ap.parse_args()
    if a.fused_quant:
        force_fused_quant()
    mb.set_key(a.key)
    wsets = [mb.make_weights(256, 7 + i) for i in range(a.nsets)]
    ref = torch.load(a.ref) if a.ref else {}
    saved = {}
    tag = f"policy={a.policy} fused_quant={int(a.fused_quant)} flydsl_sort={os.environ.get('AITER_USE_FLYDSL_MOE_SORTING', '0')}"
    for M in [int(m) for m in a.ms.split(",")]:
        x, ids, tw = mb.make_inputs(M, 1234 + M)
        out = call(x, wsets[0], ids, tw, a.policy).float()
        torch.cuda.synchronize()
        saved[M] = out.cpu()
        snr = mb.snr(ref[M].cuda(), out) if M in ref else float("nan")
        t = eb.timed(lambda: [call(x, w, ids, tw, a.policy) for w in wsets], len(wsets))
        print(f"RESULT {tag} M {M}: {t:6.1f} us/call  snr_vs_ref {snr:5.1f} dB", flush=True)
        topn = int(os.environ.get("TOPN", "0"))
        if topn:
            from torch.profiler import ProfilerActivity, profile
            fn = lambda: [call(x, w, ids, tw, a.policy) for w in wsets]  # noqa: E731
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                fn()
            g.replay()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(4):
                    g.replay()
                torch.cuda.synchronize()
            per = collections.Counter()
            for e in p.events():
                if e.device_type.name == "CUDA":
                    per[e.name.split("(")[0][:100]] += e.device_time_total
            for k, v in per.most_common(topn):
                print(f"   {v / (4 * len(wsets)):7.1f} us  {k}")
    if a.save:
        torch.save(saved, a.save)


if __name__ == "__main__":
    main()
