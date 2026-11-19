# ruff: noqa: B023, F821 - each closure is timed in the iteration that builds it
"""RCCL at refinement-step payloads. torchrun --nproc_per_node 8 comm_bench.py out.json"""

import json
import sys

import torch
import torch.distributed as dist

H = 4096
TOPK = 8
FP8 = (
    torch.float8_e4m3fnuz
    if "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
    else torch.float8_e4m3fn
)


def timeit(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    t = torch.tensor([s.elapsed_time(e) / iters], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def pack(x):
    amax = x.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-6)
    scale = amax / 240.0
    q = (x.float() / scale).to(FP8).view(torch.uint8)
    return torch.cat([q, scale.view(torch.uint8).reshape(x.shape[0], 4)], dim=1)


def unpack(b, h):
    q = b[:, :h].contiguous().view(FP8).float()
    scale = b[:, h:].contiguous().view(torch.float32)
    return q * scale


def main():
    dist.init_process_group("nccl")
    r, w = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r % torch.cuda.device_count())
    groups = {tp: dist.new_group(list(range(tp))) for tp in (2, 4, 8) if tp <= w}
    out = {"allreduce": [], "quant_allreduce": [], "alltoall": [], "sendrecv": [], "pack": []}
    for tok in (32, 256, 1024, 4096, 16384, 65536):
        iters = 20 if tok <= 4096 else 5
        x = torch.randn(tok, H, device="cuda", dtype=torch.bfloat16)
        for tp, g in groups.items():
            if r < tp:
                ms = timeit(lambda: dist.all_reduce(x, group=g), iters)
            else:
                ms = timeit(lambda: None, iters)
            if r == 0:
                out["allreduce"].append({"tp": tp, "tokens": tok, "bytes": x.numel() * 2, "ms": ms})
        # FP8 two-shot all-reduce over all ranks: pack, all-to-all, reduce, repack, all-gather, unpack.
        if tok % w == 0:
            shard = tok // w
            packed = pack(x)
            recv = torch.empty_like(packed)
            gath = torch.empty_like(packed)

            def qar():
                p = pack(x)
                dist.all_to_all_single(recv, p)
                red = unpack(recv, H).view(w, shard, H).sum(0).to(torch.bfloat16)
                dist.all_gather_into_tensor(gath, pack(red))
                return unpack(gath, H).to(torch.bfloat16)

            def wire():
                dist.all_to_all_single(recv, packed)
                dist.all_gather_into_tensor(gath, packed[:shard])

            def kernels():
                p = pack(x)
                red = unpack(p, H).view(w, shard, H).sum(0).to(torch.bfloat16)
                unpack(pack(red).repeat(w, 1), H).to(torch.bfloat16)

            row = {
                "tp": w,
                "tokens": tok,
                "total_ms": timeit(qar, iters),
                "wire_ms": timeit(wire, iters),
                "kernel_ms": timeit(kernels, iters),
            }
            if r == 0:
                out["quant_allreduce"].append(row)
        pk = timeit(lambda: unpack(pack(x), H).to(torch.bfloat16), iters)
        if r == 0:
            out["pack"].append({"tokens": tok, "ms": pk, "elements": x.numel()})
        # Pipeline hop 0 -> 1.
        if r in (0, 1):
            buf = x.clone()
            ms = timeit(
                (lambda: dist.send(buf, 1)) if r == 0 else (lambda: dist.recv(buf, 0)), iters
            )
        else:
            ms = timeit(lambda: None, iters)
        if r == 0:
            out["sendrecv"].append({"tokens": tok, "bytes": x.numel() * 2, "ms": ms})
        del x
        torch.cuda.empty_cache()
    # MoE dispatch: tokens x hidden x topk per rank, one direction.
    for tok in (32, 256, 1024, 4096, 8192):
        iters = 20 if tok <= 1024 else 5
        a = torch.randn(tok * TOPK, H, device="cuda", dtype=torch.bfloat16)
        b = torch.empty_like(a)
        ms = timeit(lambda: dist.all_to_all_single(b, a), iters)
        if r == 0:
            out["alltoall"].append({"ep": w, "tokens": tok, "bytes": a.numel() * 2, "ms": ms})
        del a, b
        torch.cuda.empty_cache()
    if r == 0:
        json.dump(out, open(sys.argv[1], "w"), indent=1)
        for k, rows in out.items():
            for row in rows:
                print(k, row)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
