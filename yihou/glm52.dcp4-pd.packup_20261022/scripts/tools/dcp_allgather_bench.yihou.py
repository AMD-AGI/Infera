#!/usr/bin/env python3
"""All-gather microbench for the page-split DSA decode indexer (Variant B).

Each rank sends one [2, rows, topk] int32 pack (fp32 score bits + int32
position) and receives W of them, once per indexer layer (21 for GLM-5.2) per
forward. Times eager and CUDA-graph replay of that all-gather over a W-rank
group, plus the merge (torch.topk over W * topk candidates).

Run inside the sglang ROCm image on one node (GPUs are picked by
HIP_VISIBLE_DEVICES; no model needed), e.g. on the decode GPUs:
  HIP_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 \
      dcp_allgather_bench.yihou.py --rows 40,80,120,240,400 --topk 2048
Prints one line per rows value from rank 0: us per all-gather (eager, graph),
us per merge, and the derived per-step cost for --layers indexer layers.
"""
import argparse
import os

import torch
import torch.distributed as dist


def timed(fn, iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="40,80,120,160,240,320,400")
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=21)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    w = dist.get_world_size()
    rank = dist.get_rank()
    dev = torch.device("cuda", local_rank)

    for rows in [int(x) for x in args.rows.split(",")]:
        send = torch.randint(
            -(2**31), 2**31 - 1, (2, rows, args.topk), dtype=torch.int32, device=dev
        )
        recv = torch.empty((w * 2, rows, args.topk), dtype=torch.int32, device=dev)

        def gather():
            dist.all_gather_into_tensor(recv, send)

        scores = torch.randn(rows, w * args.topk, device=dev)

        def merge():
            torch.topk(scores, args.topk, dim=1)

        for _ in range(args.warmup):
            gather()
            merge()
        eager_us = timed(gather, args.iters)
        merge_us = timed(merge, args.iters)

        graph_us = float("nan")
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    gather()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gather()
            for _ in range(args.warmup):
                graph.replay()
            graph_us = timed(graph.replay, args.iters)
        except Exception as e:  # noqa: BLE001 -- report and keep the eager number
            if rank == 0:
                print(f"rows={rows}: graph capture failed: {e!r}")

        if rank == 0:
            mb = recv.numel() * 4 / 1e6
            best = min(eager_us, graph_us) if graph_us == graph_us else eager_us
            print(
                f"W={w} rows={rows} topk={args.topk} recv={mb:.1f}MB "
                f"allgather eager={eager_us:.1f}us graph={graph_us:.1f}us "
                f"({mb / (best / 1e6) / 1e3:.1f} GB/s) merge={merge_us:.1f}us "
                f"per_step({args.layers} layers)={(best + merge_us) * args.layers / 1000:.2f}ms",
                flush=True,
            )
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
