#!/usr/bin/env python3
"""Where MegaMoeTP's time goes, without instrumenting the kernel: on real layer-3 weights, per global token count T,
  prod      production MoE GEMMs only (aiter fused_moe on this rank's inter slice, all T tokens; no communication)
  tp8       MegaMoeTP over the 8 ranks (all-gather + GEMMs + reduce-scatter in one persistent kernel)
  tp1       MegaMoeTP on one rank's slice with all T tokens local (world size 1: the same GEMM work, no cross-rank waits)
  rccl      RCCL all-gather + reduce-scatter of the T x H bf16 rows (a communication reference)
for MegaMoeTP's default knobs (mt 3, npp 1, q 1) and the tuned ones (mt 4, npp 2, q 2). tp1 vs prod is the GEMM
efficiency of the fused kernel; tp8 - tp1 is what the cross-rank part (gather, waits, scatter) costs inside it.
CUDA-graph us per call, slowest rank. tp1 is also checked against prod (SNR of the unreduced partial).

Usage (8 GPUs, production image, HSA_ENABLE_IPC_MODE_LEGACY=1):
    torchrun --standalone --nproc_per_node=8 megatp_phase.py [--tokens 384,640,768] [--out OUT.json]
"""
import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import megatp_realw_ab as ab  # noqa: E402
import moe_bench as mb  # noqa: E402

KNOBS = {"default": (3, 1, 1), "tuned": (4, 2, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--tokens", default="384,640,768")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    toks = [int(v) for v in a.tokens.split(",")]
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    from aiter import ActivationType
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

    solo = [dist.new_group([r]) for r in range(world)][rank]
    ws, gate, bias, inter, _ = ab.load_layer(a.layer, rank, world)
    mb.set_key("k512")

    def tmax(v):
        t = torch.tensor([v], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return round(t.item(), 1)

    def tmin(v):
        t = torch.tensor([v], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return round(t.item(), 1)

    data = {}
    for T in toks:
        m = T // world
        g = torch.Generator(device="cuda").manual_seed(1000 * T + rank)
        x = torch.randn(m, ab.H, dtype=torch.bfloat16, device="cuda", generator=g)
        tw, ids = ab.route(x, gate, bias)
        xg = torch.empty(T, ab.H, dtype=x.dtype, device="cuda")
        twg = torch.empty(T, ab.TOPK, dtype=tw.dtype, device="cuda")
        idsg = torch.empty(T, ab.TOPK, dtype=ids.dtype, device="cuda")
        dist.all_gather_into_tensor(xg, x)
        dist.all_gather_into_tensor(twg, tw)
        dist.all_gather_into_tensor(idsg, ids)
        yl = torch.empty(m, ab.H, dtype=torch.bfloat16, device="cuda")
        ref_part = mb.aiter_call(xg, ws, idsg, twg).float().clone()
        dist.reduce_scatter_tensor(yl, mb.aiter_call(xg, ws, idsg, twg))
        ref_red = yl.float().clone()
        part = torch.empty(T, ab.H, dtype=torch.bfloat16, device="cuda")

        def rccl():
            dist.all_gather_into_tensor(xg, x)
            dist.reduce_scatter_tensor(yl, part)

        row = {"prod": tmax(ab.graph_us(lambda: mb.aiter_call(xg, ws, idsg, twg)))}
        try:
            row["rccl"] = tmax(ab.graph_us(rccl))
        except Exception as e:  # noqa: BLE001 - RCCL capture may be unsupported
            row["rccl"] = None
            if rank == 0:
                print(f"rccl timing failed: {e!r}", flush=True)
        data[T] = (x, tw, ids, xg, twg, idsg, ref_part, ref_red, row)

    def instance(knob, tp, mmax):
        mt, npp, q = KNOBS[knob]
        cfg = MegaMoeTPConfig(rank=0 if tp == "tp1" else rank, world_size=1 if tp == "tp1" else world,
                              model_dim=ab.H, inter_dim=inter, experts=ab.E, topk=ab.TOPK, max_local_tokens=mmax,
                              activation=ActivationType.Silu, comm_mode="ag_rs", comm_dtype="bf16",
                              lb_mt=mt, lb_mt_small=mt, lb_npp=npp, lb_q=q)
        return MegaMoeTP(cfg, w1=ws[0], w1_scale=ws[2], w2=ws[1], w2_scale=ws[3], group=solo if tp == "tp1" else None)

    def measure(mega, tp, knob, T):
        x, tw, ids, xg, twg, idsg, ref_part, ref_red, row = data[T]
        xi, twi, idsi, ref = (xg, twg, idsg, ref_part) if tp == "tp1" else (x, tw, ids, ref_red)
        o = torch.empty(xi.shape[0], ab.H, dtype=torch.bfloat16, device="cuda")
        y = mega(xi, twi, idsi, out=o).float().clone()
        torch.cuda.synchronize()
        row[f"{tp}_{knob}"] = tmax(ab.graph_us(lambda: mega(xi, twi, idsi, out=o)))
        row[f"{tp}_{knob}_snr"] = tmin(mb.snr(ref, y))

    for knob in KNOBS:
        # tp8: one instance for all T (128 local rows); tp1: all T rows are local, one instance per T (LDS bound)
        plans = [("tp8", 128, toks)] + [("tp1", T, [T]) for T in toks]
        for tp, mmax, ts in plans:
            try:
                mega = instance(knob, tp, mmax)
                mega.prepare([T if tp == "tp1" else T // world for T in ts])
                for T in ts:
                    measure(mega, tp, knob, T)
                del mega
            except Exception as e:  # noqa: BLE001 - e.g. max_local_tokens beyond LDS at world size 1
                if rank == 0:
                    print(f"{tp} {knob} T={ts} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            torch.cuda.empty_cache()
            dist.barrier()
    if rank == 0:
        out = {T: data[T][-1] for T in toks}
        print(json.dumps(out, indent=1), flush=True)
        print("T     prod  rccl | tp1 def  tp8 def  (tp8-tp1) | tp1 tun  tp8 tun  (tp8-tp1)   us per layer call")
        for T in toks:
            r = out[T]
            f = lambda k: r.get(k) if r.get(k) is not None else float("nan")  # noqa: E731
            print(f"{T:4d}  {f('prod'):5.1f} {f('rccl'):5.1f} | {f('tp1_default'):7.1f} {f('tp8_default'):8.1f} "
                  f"({f('tp8_default') - f('tp1_default'):6.1f}) | {f('tp1_tuned'):7.1f} {f('tp8_tuned'):8.1f} "
                  f"({f('tp8_tuned') - f('tp1_tuned'):6.1f})", flush=True)
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
