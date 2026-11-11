#!/usr/bin/env python3
"""Per-CTA phase times inside MegaMoeTP (needs megatp_dbg_patch.py applied and AITER_MEGAMOE_TP_DBG=1): real layer-3
weights, TP8 default / tuned knobs and TP1 tuned, at 384 and 768 global tokens. Eager launches (the timestamps are
read back after each); per CTA, relative to the earliest kernel entry on that GPU:
  t1 own rows quantized, t2 compute warps done with their GEMM units, t3 all roles done, t4 finish done.
Prints, per instance and T, the median / p90 / max over CTAs (slowest rank) and the compute-done time per XCD
(cta % 8), to tell load imbalance (a few late CTAs) from uniformly slow units.

Usage (8 GPUs, patched image, HSA_ENABLE_IPC_MODE_LEGACY=1 AITER_MEGAMOE_TP_DBG=1):
    torchrun --standalone --nproc_per_node=8 megatp_dbg.py [--tokens 384,768] [--reps 5] [--out OUT.json]
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


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--tokens", default="384,768")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    assert os.environ.get("AITER_MEGAMOE_TP_DBG") == "1", "set AITER_MEGAMOE_TP_DBG=1"
    toks = [int(v) for v in a.tokens.split(",")]
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    from aiter import ActivationType
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

    solo = [dist.new_group([r]) for r in range(world)][rank]
    ws, gate, bias, inter, _ = ab.load_layer(a.layer, rank, world)
    inputs = {}
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
        inputs[T] = (x, tw, ids, xg, twg, idsg)

    results = {}
    plans = [("tp8", k) for k in KNOBS] + [("tp1", "tuned")]
    for tp, knob in plans:
        mt, npp, q = KNOBS[knob]
        for T in toks:
            key = f"{tp}_{knob}_{T}"
            try:
                mmax = T if tp == "tp1" else 128
                cfg = MegaMoeTPConfig(rank=0 if tp == "tp1" else rank, world_size=1 if tp == "tp1" else world,
                                      model_dim=ab.H, inter_dim=inter, experts=ab.E, topk=ab.TOPK, max_local_tokens=mmax,
                                      activation=ActivationType.Silu, comm_mode="ag_rs", comm_dtype="bf16",
                                      lb_mt=mt, lb_mt_small=mt, lb_npp=npp, lb_q=q)
                mega = MegaMoeTP(cfg, w1=ws[0], w1_scale=ws[2], w2=ws[1], w2_scale=ws[3],
                                 group=solo if tp == "tp1" else None)
                n_cta = int(mega.engine.n_cta)
                buf = torch.zeros(n_cta * 8, dtype=torch.int64, device="cuda")
                mega.engine.dbg_buf = buf
                mlocal = T if tp == "tp1" else T // world
                mega.prepare([mlocal])
                x, tw, ids, xg, twg, idsg = inputs[T]
                xi, twi, idsi = (xg, twg, idsg) if tp == "tp1" else (x, tw, ids)
                o = torch.empty(xi.shape[0], ab.H, dtype=torch.bfloat16, device="cuda")
                samples = []
                for _ in range(a.reps + 2):
                    buf.zero_()
                    torch.cuda.synchronize()
                    dist.barrier()
                    mega(xi, twi, idsi, out=o)
                    torch.cuda.synchronize()
                    samples.append(buf.view(n_cta, 8)[:, :5].cpu())
                samples = samples[2:]
                ok = all((s[:, 0] > 0).all() for s in samples)
                st = {}
                for k, name in ((1, "quant"), (2, "compute"), (3, "roles"), (4, "finish")):
                    per = []
                    for s in samples:
                        rel = (s[:, k] - s[:, 0].min()).float() / 100.0  # 100 MHz ticks -> us
                        per.append([pct(rel.tolist(), 0.5), pct(rel.tolist(), 0.9), rel.max().item()])
                    st[name] = [round(sum(p[i] for p in per) / len(per), 1) for i in range(3)]
                xcd = []
                s = samples[-1]
                rel = (s[:, 2] - s[:, 0].min()).float() / 100.0
                for xc in range(8):
                    v = rel[xc::8].tolist()
                    xcd.append([round(sum(v) / len(v), 1), round(max(v), 1)])
                rec = {"ok": ok, "n_cta": n_cta, "phases_us_p50_p90_max": st, "compute_done_by_xcd_mean_max": xcd}
                allr = [None] * world
                dist.all_gather_object(allr, rec)
                worst = max(range(world), key=lambda r: allr[r]["phases_us_p50_p90_max"]["finish"][2])
                results[key] = {"worst_rank": worst, **allr[worst]}
                del mega
            except Exception as e:  # noqa: BLE001
                results[key] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            torch.cuda.empty_cache()
            dist.barrier()
            if rank == 0:
                print(key, json.dumps(results[key]), flush=True)
    if rank == 0 and a.out:
        json.dump(results, open(a.out, "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
