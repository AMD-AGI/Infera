#!/usr/bin/env python3
"""TP8 decode MoE layer on real GLM-5.2-MXFP4 weights (layer LAYER, this rank's TP8 slice of every expert's inter dim,
shared expert in slot 256, shuffled exactly as sglang's mxfp4 aiter path does), real router top-k on each rank's own
tokens; production path (all-gather + aiter fused_moe with the token=512 tuned row + reduce-scatter, torch.distributed
collectives) vs aiter's MegaMoeTP (one kernel, bf16 or fp8 comm). Prints the SNR of MegaMoeTP against the production
output (this also checks that sglang's weight layout is the one MegaMoeTP expects) and CUDA-graph us per layer call
(slowest rank). The production timing uses RCCL collectives, not sglang's custom all-gather / reduce-scatter kernels
(in-graph PP8 trace: 38.4 + 32.2 us), so compare its MoE part only.

Usage (8 GPUs, production image, HSA_ENABLE_IPC_MODE_LEGACY=1):
    torchrun --standalone --nproc_per_node=8 megatp_realw_ab.py [--layer 3] [--tokens 576,704,832] [--comm bf16]
"""
import argparse
import json
import os
import statistics as st
import sys

import torch
import torch.distributed as dist
from safetensors import safe_open

sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
import moe_bench as mb  # noqa: E402

M_DIR = "/perf_apps/data/models/GLM-5.2-MXFP4"
H, E, TOPK, INTER_FULL = 6144, 257, 9, 2048


def load_layer(layer, rank, world):
    """This rank's slice, as sglang loads and shuffles it (mxfp4.py process_weights_after_loading, non-interleaved)."""
    from aiter.ops.shuffle import shuffle_scale, shuffle_weight

    inter = INTER_FULL // world
    r0, r1 = rank * inter, (rank + 1) * inter
    p = f"model.layers.{layer}.mlp"
    experts = [f"{p}.experts.{e}" for e in range(256)] + [f"{p}.shared_experts"]
    wm = json.load(open(f"{M_DIR}/model.safetensors.index.json"))["weight_map"]
    want = {}
    for x in experts:
        for w in ("gate", "up"):
            for k in ("weight", "weight_scale"):
                want[f"{x}.{w}_proj.{k}"] = ("rows", r0, r1)
        want[f"{x}.down_proj.weight"] = ("cols", r0 // 2, r1 // 2)
        want[f"{x}.down_proj.weight_scale"] = ("cols", r0 // 32, r1 // 32)
    want[f"{p}.gate.weight"] = ("all",)
    want[f"{p}.gate.e_score_correction_bias"] = ("all",)
    t = {}
    for f in sorted({wm[n] for n in want}):
        with safe_open(f"{M_DIR}/{f}", framework="pt", device="cpu") as h:
            for n, how in want.items():
                if wm[n] != f:
                    continue
                s = h.get_slice(n)
                t[n] = s[how[1]:how[2]] if how[0] == "rows" else s[:, how[1]:how[2]] if how[0] == "cols" else s[:]
    w13 = torch.stack([torch.cat([t[f"{x}.gate_proj.weight"], t[f"{x}.up_proj.weight"]]) for x in experts])
    s13 = torch.stack([torch.cat([t[f"{x}.gate_proj.weight_scale"], t[f"{x}.up_proj.weight_scale"]]) for x in experts])
    w2 = torch.stack([t[f"{x}.down_proj.weight"] for x in experts])
    s2 = torch.stack([t[f"{x}.down_proj.weight_scale"] for x in experts])
    w13, s13, w2, s2 = (x.contiguous().cuda() for x in (w13, s13, w2, s2))
    w13 = shuffle_weight(w13, is_guinterleave=False, gate_up=True).view(torch.float4_e2m1fn_x2)
    w2 = shuffle_weight(w2, is_guinterleave=False, gate_up=False).view(torch.float4_e2m1fn_x2)
    s13 = shuffle_scale(s13.view(-1, s13.shape[-1]), experts_cnt=E, is_guinterleave=False, gate_up=True)
    s2 = shuffle_scale(s2.view(-1, s2.shape[-1]), experts_cnt=E, is_guinterleave=False, gate_up=False)
    w13.is_shuffled = w2.is_shuffled = True
    gate = t[f"{p}.gate.weight"].cuda().to(torch.bfloat16)
    bias = t[f"{p}.gate.e_score_correction_bias"].cuda().float()
    return (w13, w2, s13, s2), gate, bias, inter


def route(x, gate, bias):
    s = torch.sigmoid((x @ gate.t()).float())
    ids = (s + bias).topk(8, dim=1).indices.to(torch.int32)
    w = s.gather(1, ids.long())
    w = w / w.sum(1, keepdim=True) * 2.5
    ids = torch.cat([ids, torch.full((x.shape[0], 1), 256, dtype=torch.int32, device=x.device)], 1).contiguous()
    tw = torch.cat([w, torch.ones(x.shape[0], 1, device=x.device)], 1).contiguous()
    return tw, ids


def graph_us(fn, calls=10, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    # thread_local: the NCCL watchdog thread queries events of earlier collectives while this thread captures
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    dist.barrier()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / calls)
    t = torch.tensor([st.median(ts)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--tokens", default="576,704,832")
    ap.add_argument("--comm", default="bf16")
    a = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    from aiter import ActivationType
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

    ws, gate, bias, inter = load_layer(a.layer, rank, world)
    if rank == 0:
        print(f"layer {a.layer}: w13 {tuple(ws[0].shape)} s13 {tuple(ws[2].shape)} w2 {tuple(ws[1].shape)} "
              f"s2 {tuple(ws[3].shape)} inter/rank {inter}", flush=True)
    mb.set_key("k512")
    toks = [int(v) for v in a.tokens.split(",")]
    mega = MegaMoeTP(MegaMoeTPConfig(rank=rank, world_size=world, model_dim=H, inter_dim=inter, experts=E, topk=TOPK,
                                     max_local_tokens=max(128, max(toks) // world), activation=ActivationType.Silu,
                                     comm_mode="ag_rs", comm_dtype=a.comm),
                     w1=ws[0], w1_scale=ws[2], w2=ws[1], w2_scale=ws[3])
    mega.prepare([T // world for T in toks])
    for T in toks:
        m = T // world
        g = torch.Generator(device="cuda").manual_seed(1000 * T + rank)
        x = torch.randn(m, H, dtype=torch.bfloat16, device="cuda", generator=g)
        tw, ids = route(x, gate, bias)
        xg = torch.empty(T, H, dtype=x.dtype, device="cuda")
        twg = torch.empty(T, TOPK, dtype=tw.dtype, device="cuda")
        idsg = torch.empty(T, TOPK, dtype=ids.dtype, device="cuda")
        yl = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")

        def prod():
            dist.all_gather_into_tensor(xg, x)
            dist.all_gather_into_tensor(twg, tw)
            dist.all_gather_into_tensor(idsg, ids)
            y = mb.aiter_call(xg, ws, idsg, twg)
            dist.reduce_scatter_tensor(yl, y)
            return yl

        def moe_only():
            return mb.aiter_call(xg, ws, idsg, twg)

        ref = prod().float().clone()
        ref2 = prod().float().clone()
        out = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")
        mega.clear_errors()
        y = mega(x, tw, ids, out=out).float().clone()
        torch.cuda.synchronize()
        err = mega.poll_errors()
        snr = torch.tensor([mb.snr(ref, y), mb.snr(ref, ref2)], device="cuda")
        dist.all_reduce(snr, op=dist.ReduceOp.MIN)
        t_mega = graph_us(lambda: mega(x, tw, ids, out=out))
        t_moe = graph_us(moe_only)
        try:
            t_prod = graph_us(prod)
        except Exception as e:  # RCCL capture may be unsupported; the SNR and the other two timings still stand
            if rank == 0:
                print(f"production graph timing failed: {e!r}", flush=True)
            t_prod = float("nan")
        if rank == 0:
            print(f"RESULT layer {a.layer} tokens {T} (local {m}): MegaMoeTP-{a.comm} vs production {snr[0]:.1f} dB "
                  f"(production run-to-run {snr[1]:.1f} dB), watchdog {err} | us/call: production AG+MoE+RS (RCCL) "
                  f"{t_prod:.1f}, MoE only {t_moe:.1f}, MegaMoeTP {t_mega:.1f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
