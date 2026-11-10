#!/usr/bin/env python3
"""M2 step 1: MegaMoeTP's large-batch schedule knobs on real GLM-5.2-MXFP4 weights (megatp_realw_ab.py's layer
loading, routing and graph timing) at our TP8 decode global tokens 384 / 640 / 768 (per-rank batch 12 / 20 / 24 x 4
verify rows). In the decode graphs MegaMoeTP takes 181 / 222 / 236 us per layer against 130 / 151 / 165 us for the
production GEMMs alone (synthetic traces 11-10). Knobs: row tile 16 x mt (lb_mt and, up to lb_small_max = 512 tokens,
lb_mt_small: set together), lb_npp GEMM1 column blocks per A pass, lb_q output column chunks per GEMM2 unit; defaults
mt 3, npp 1, q 1. One instance per config (same weights), resumable: configs already in OUT are skipped.

Usage (8 GPUs, production image, HSA_ENABLE_IPC_MODE_LEGACY=1):
    torchrun --standalone --nproc_per_node=8 megatp_sweep.py --out OUT.jsonl [--mt 1,2,3,4,6,8] [--npp 1,2,4] [--q 1]
"""
import argparse
import gc
import itertools
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import megatp_realw_ab as ab  # noqa: E402
import moe_bench as mb  # noqa: E402

TOKENS = (384, 640, 768)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mt", default="1,2,3,4,6,8")
    ap.add_argument("--npp", default="1,2,4")
    ap.add_argument("--q", default="1")
    ap.add_argument("--tokens", default=",".join(map(str, TOKENS)), help="global tokens (TP4 4+4: 256,320,384)")
    a = ap.parse_args()
    tokens = [int(v) for v in a.tokens.split(",")]
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    from aiter import ActivationType
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                if r.get("world", 8) == world:
                    done.add(tuple(r["cfg"]))
            except (ValueError, KeyError):
                pass
    ws, gate, bias, inter, _ = ab.load_layer(a.layer, rank, world)
    mb.set_key("k512")
    inputs = {}
    for T in tokens:
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
        y = mb.aiter_call(xg, ws, idsg, twg)
        dist.reduce_scatter_tensor(yl, y)
        t_moe = ab.graph_us(lambda: mb.aiter_call(xg, ws, idsg, twg))
        inputs[T] = (x, tw, ids, yl.float().clone(), t_moe)
        if rank == 0:
            print(f"tokens {T}: production MoE only (no comm) {t_moe:.1f} us", flush=True)
    out = open(a.out, "a") if rank == 0 else None
    for mt, npp, q in itertools.product(*[[int(v) for v in s.split(",")] for s in (a.mt, a.npp, a.q)]):
        if (mt, npp, q) in done:
            continue
        res = {"cfg": [mt, npp, q], "world": world}
        try:
            mega = MegaMoeTP(MegaMoeTPConfig(rank=rank, world_size=world, model_dim=ab.H, inter_dim=inter, experts=ab.E,
                                             topk=ab.TOPK, max_local_tokens=128, activation=ActivationType.Silu,
                                             comm_mode="ag_rs", comm_dtype="bf16", lb_mt=mt, lb_mt_small=mt,
                                             lb_npp=npp, lb_q=q),
                             w1=ws[0], w1_scale=ws[2], w2=ws[1], w2_scale=ws[3])
            mega.prepare([T // world for T in tokens])
            for T in tokens:
                x, tw, ids, ref, t_moe = inputs[T]
                o = torch.empty(x.shape[0], ab.H, dtype=torch.bfloat16, device="cuda")
                mega.clear_errors()
                y = mega(x, tw, ids, out=o).float().clone()
                torch.cuda.synchronize()
                snr = torch.tensor([mb.snr(ref, y)], device="cuda")
                dist.all_reduce(snr, op=dist.ReduceOp.MIN)
                t = torch.tensor([ab.graph_us(lambda: mega(x, tw, ids, out=o))], device="cuda")
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                res[str(T)] = {"us": round(t.item(), 1), "snr": round(snr.item(), 1), "prod_moe_us": round(t_moe, 1),
                               "watchdog": str(mega.poll_errors())}
            del mega
        except Exception as e:  # noqa: BLE001 - an invalid tile for this shape: record and go on
            res["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            print(json.dumps(res), flush=True)
            out.write(json.dumps(res) + "\n")
            out.flush()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
