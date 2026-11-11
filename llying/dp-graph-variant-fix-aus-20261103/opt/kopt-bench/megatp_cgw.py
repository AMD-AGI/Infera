#!/usr/bin/env python3
"""MegaMoeTP with tapered GEMM2 column groups (needs megatp_cgw_patch.py applied for explicit widths): real layer-3
weights, CUDA-graph us and SNR against production per token count, for several (q, widths) variants. Also the
production MoE part (GEMMs only) as the reference. One instance per variant, with the compile cache cleared (the widths
are read when the kernel is compiled). H 6144 has 12 GEMM2 chunks of 512 columns; q 3 alone gives aiter's [3,3,3,2,1].

Usage (8 or 4 GPUs, patched image, HSA_ENABLE_IPC_MODE_LEGACY=1):
    torchrun --standalone --nproc_per_node=8 megatp_cgw.py --out OUT.jsonl [--mt 4 --npp 2 --lb-min 256
        --tokens 384,640,768 --variants "q2:2:;q3:3:;3322:2:3,3,2,2,1,1"]
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

# name:q:widths (widths in 512-column chunks, empty = aiter's own for that q)
DEFAULT_VARIANTS = ";".join([
    "uniform_q2:2:",
    "uniform_q1:2:" + ",".join(["1"] * 12),
    "2x5_1x2:2:2,2,2,2,2,1,1",
    "2x4_1x4:2:2,2,2,2,1,1,1,1",
    "2x3_1x6:2:2,2,2,1,1,1,1,1,1",
    "3x2_2x2_1x2:2:3,3,2,2,1,1",
    "3x3_2x1_1x1:2:3,3,3,2,1",
    "4x1_2x3_1x2:2:4,2,2,2,1,1",
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mt", type=int, default=4)
    ap.add_argument("--npp", type=int, default=2)
    ap.add_argument("--lb-min", type=int, default=0, help="AITER_MEGAMOE_TP_LB_MIN (0 = aiter default)")
    ap.add_argument("--tokens", default="384,640,768")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    variants = [v.split(":") for v in a.variants.split(";")]
    if a.lb_min:
        os.environ["AITER_MEGAMOE_TP_LB_MIN"] = str(a.lb_min)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    from aiter import ActivationType
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig
    # compile_mega_moe_tp is functools.cache'd on its arguments, which do not include the widths
    from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp_kernel import compile_mega_moe_tp

    ws, gate, bias, inter, _ = ab.load_layer(a.layer, rank, world)
    mb.set_key("k512")
    inputs = {}
    prod = {}
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
        dist.reduce_scatter_tensor(yl, mb.aiter_call(xg, ws, idsg, twg))
        inputs[T] = (x, tw, ids, yl.float().clone())
        t = torch.tensor([ab.graph_us(lambda: mb.aiter_call(xg, ws, idsg, twg))], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        prod[T] = round(t.item(), 1)
    out = open(a.out, "a") if rank == 0 else None
    if rank == 0:
        out.write(json.dumps({"world": world, "mt": a.mt, "npp": a.npp, "lb_min": a.lb_min, "prod_moe_us": prod}) + "\n")
        print(f"world {world} mt {a.mt} npp {a.npp} lb_min {a.lb_min}; production MoE (GEMMs only):", prod, flush=True)
    for pname, q, cgw in variants:
        os.environ["AITER_MEGAMOE_TP_LB_CGW"] = cgw
        compile_mega_moe_tp.cache_clear()
        res = {"pattern": pname, "q": int(q), "cgw": cgw}
        try:
            cfg = MegaMoeTPConfig(rank=rank, world_size=world, model_dim=ab.H, inter_dim=inter, experts=ab.E,
                                  topk=ab.TOPK, max_local_tokens=max(128, max(tokens) // world), activation=ActivationType.Silu,
                                  comm_mode="ag_rs", comm_dtype="bf16", lb_mt=a.mt, lb_mt_small=a.mt, lb_npp=a.npp,
                                  lb_q=int(q))
            mega = MegaMoeTP(cfg, w1=ws[0], w1_scale=ws[2], w2=ws[1], w2_scale=ws[3])
            mega.prepare([T // world for T in tokens])
            for T in tokens:
                x, tw, ids, ref = inputs[T]
                o = torch.empty(x.shape[0], ab.H, dtype=torch.bfloat16, device="cuda")
                y = mega(x, tw, ids, out=o).float().clone()
                torch.cuda.synchronize()
                snr = torch.tensor([mb.snr(ref, y)], device="cuda")
                dist.all_reduce(snr, op=dist.ReduceOp.MIN)
                t = torch.tensor([ab.graph_us(lambda: mega(x, tw, ids, out=o))], device="cuda")
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                res[str(T)] = {"us": round(t.item(), 1), "snr": round(snr.item(), 1)}
            del mega
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
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
