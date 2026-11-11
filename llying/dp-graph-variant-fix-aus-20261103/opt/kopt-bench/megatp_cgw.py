#!/usr/bin/env python3
"""MegaMoeTP with tapered GEMM2 column groups (needs megatp_cgw_patch.py applied): real layer-3 weights, TP8, the tuned
knobs (lb_mt 4, npp 2, q 2), CUDA-graph us and SNR against production at 384 / 640 / 768 global tokens, for the
uniform widths (q 2: twelve groups of 2 chunks) and tapered ones. Also the production MoE part (GEMMs only) as the
reference. One instance per pattern (the widths are read when the kernel is compiled).

Usage (8 GPUs, patched image, HSA_ENABLE_IPC_MODE_LEGACY=1):
    torchrun --standalone --nproc_per_node=8 megatp_cgw.py [--out OUT.jsonl]
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

TOKENS = (384, 640, 768)
PATTERNS = {
    "uniform_q2": "",
    "2x10_1x4": ",".join(["2"] * 10 + ["1"] * 4),
    "2x11_1x2": ",".join(["2"] * 11 + ["1"] * 2),
    "2x8_1x8": ",".join(["2"] * 8 + ["1"] * 8),
    "4x4_2x3_1x2": ",".join(["4"] * 4 + ["2"] * 3 + ["1"] * 2),
    "4x3_2x4_1x4": ",".join(["4"] * 3 + ["2"] * 4 + ["1"] * 4),
    "4x5_2x1_1x2": ",".join(["4"] * 5 + ["2"] + ["1"] * 2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    from aiter import ActivationType
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

    ws, gate, bias, inter, _ = ab.load_layer(a.layer, rank, world)
    mb.set_key("k512")
    inputs = {}
    prod = {}
    for T in TOKENS:
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
        out.write(json.dumps({"prod_moe_us": prod}) + "\n")
        print("production MoE (GEMMs only):", prod, flush=True)
    for pname, cgw in PATTERNS.items():
        os.environ["AITER_MEGAMOE_TP_LB_CGW"] = cgw
        res = {"pattern": pname, "cgw": cgw}
        try:
            cfg = MegaMoeTPConfig(rank=rank, world_size=world, model_dim=ab.H, inter_dim=inter, experts=ab.E,
                                  topk=ab.TOPK, max_local_tokens=128, activation=ActivationType.Silu,
                                  comm_mode="ag_rs", comm_dtype="bf16", lb_mt=4, lb_mt_small=4, lb_npp=2, lb_q=2)
            mega = MegaMoeTP(cfg, w1=ws[0], w1_scale=ws[2], w2=ws[1], w2_scale=ws[3])
            mega.prepare([T // world for T in TOKENS])
            for T in TOKENS:
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
