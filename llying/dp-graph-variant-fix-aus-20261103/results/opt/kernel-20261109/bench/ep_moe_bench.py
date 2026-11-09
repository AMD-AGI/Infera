#!/usr/bin/env python3
"""TP8 decode MoE, two ways to shard the 256 + 1 experts over 8 ranks (DP-attention all-gather / reduce-scatter
unchanged, so every rank sees all M tokens):
  tp:  every rank holds all 257 experts with inter 2048 / 8 = 256 (production);
  ep:  rank r holds routed experts [32r, 32r + 32) plus the shared expert, inter 2048 (aiter fused_moe with
       expert_mask over the 257 global ids, 33 local slots).
Same tokens and routing for both; the ep time is measured for every rank's mask (the step waits for the slowest).

Usage (production image): ep_moe_bench.py --ms 384,576,704,768 [--skew 0]
"""
import argparse
import statistics as st
import sys

import torch

sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import moe_bench as mb  # noqa: E402

E_GLOBAL, H = 257, 6144


def make_local_weights(n_local, inter, seed):
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randint(0, 256, (n_local, 2 * inter, H // 2), dtype=torch.uint8, device="cuda", generator=g)
    w2 = torch.randint(0, 256, (n_local, H, inter // 2), dtype=torch.uint8, device="cuda", generator=g)
    s13 = torch.randint(119, 124, (n_local, 2 * inter, H // 32), dtype=torch.uint8, device="cuda", generator=g)
    s2 = torch.randint(119, 124, (n_local, H, inter // 32), dtype=torch.uint8, device="cuda", generator=g)
    s13 = e8m0_shuffle(s13.reshape(n_local * 2 * inter, -1)).view(n_local, 2 * inter, -1)
    s2 = e8m0_shuffle(s2.reshape(n_local * H, -1)).view(n_local, H, -1)
    w13 = shuffle_weight(w13.contiguous(), (16, 16)).view(torch.float4_e2m1fn_x2)
    w2 = shuffle_weight(w2.contiguous(), (16, 16)).view(torch.float4_e2m1fn_x2)
    w13.is_shuffled = w2.is_shuffled = True
    return w13, w2, s13, s2


def call(x, ws, ids, tw, mask):
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType
    from aiter.ops.flydsl.moe_common import GateMode

    w13, w2, s13, s2 = ws
    return fm.fused_moe(x, w13, w2, tw, ids, expert_mask=mask, activation=ActivationType.Silu,
                        quant_type=QuantType.per_1x32, doweight_stage1=False, w1_scale=s13, w2_scale=s2,
                        a1_scale=None, a2_scale=None, bias1=None, bias2=None, hidden_pad=0, intermediate_pad=0,
                        gate_mode=GateMode.SEPARATED.value, moe_sorting_dispatch_policy=2)


def timed(fn, n, reps=20):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        t.append(a.elapsed_time(b) * 1e3 / n)
    return st.median(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="384,576,704,768")
    ap.add_argument("--nsets", type=int, default=3)
    ap.add_argument("--ranks", default="0,3,7")
    ap.add_argument("--skew", type=float, default=0.0, help="Zipf-like router skew: logits += skew * log-rank")
    a = ap.parse_args()
    tp_w = [mb.make_weights(256, 7 + i) for i in range(a.nsets)]
    ep_w = [make_local_weights(33, 2048, 70 + i) for i in range(a.nsets)]
    for M in [int(m) for m in a.ms.split(",")]:
        g = torch.Generator(device="cuda").manual_seed(M)
        x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda", generator=g)
        logits = torch.rand(M, 256, device="cuda", generator=g)
        if a.skew:
            logits = logits + a.skew * torch.log1p(torch.arange(256, device="cuda").flip(0).float()) / 6.0
        routed = logits.topk(8, dim=1).indices.to(torch.int32)
        ids = torch.cat([routed, torch.full((M, 1), 256, dtype=torch.int32, device="cuda")], 1).contiguous()
        tw = torch.rand(M, 9, device="cuda", generator=g).float().contiguous()
        t_tp = timed(lambda: [call(x, w, ids, tw, None) for w in tp_w], len(tp_w))
        res = []
        for r in [int(v) for v in a.ranks.split(",")]:
            mask = torch.zeros(E_GLOBAL, dtype=torch.int32, device="cuda")
            mask[32 * r:32 * r + 32] = 1
            mask[256] = 1
            local_routes = int(((ids[:, :8] >= 32 * r) & (ids[:, :8] < 32 * r + 32)).sum().item()) + M
            t_ep = timed(lambda: [call(x, w, ids, tw, mask) for w in ep_w], len(ep_w))
            res.append(f"rank{r} {t_ep:6.1f} us ({local_routes} routes)")
        print(f"M {M:4d} skew {a.skew}: tp (257 x i256) {t_tp:6.1f} us ({M * 9} routes) | ep (33 x i2048): " + "  ".join(res),
              flush=True)


if __name__ == "__main__":
    main()
