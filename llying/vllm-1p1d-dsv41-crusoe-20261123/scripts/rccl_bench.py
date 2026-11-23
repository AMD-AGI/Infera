"""All-gather / reduce-scatter between 2 GPUs at prefill's MoE message size, under the caller's NCCL_* / RCCL_* env.

usage: python3 rccl_bench.py LABEL [TOKENS=16384] [HIDDEN=5120]
Each rank holds TOKENS x HIDDEN bf16 (the prefill leg's full 16,384-token step): all-gather receives that much from the
peer, reduce-scatter sends that much. Also times a plain peer copy (hipMemcpyPeer) of the same size for comparison.
Prints one line per op on rank 0: ms per call and GB/s per direction.
"""
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, ws, port, label, n, h):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=ws, device_id=torch.device("cuda", rank))
    x = torch.randn(n, h, dtype=torch.bfloat16, device="cuda")
    ag_out = torch.empty(n * ws, h, dtype=torch.bfloat16, device="cuda")
    rs_in = torch.randn(n * ws, h, dtype=torch.bfloat16, device="cuda")
    rs_out = torch.empty(n, h, dtype=torch.bfloat16, device="cuda")
    nbytes = n * h * 2
    ops = [("allgather", lambda: dist.all_gather_into_tensor(ag_out, x)),
           ("reducescatter", lambda: dist.reduce_scatter_tensor(rs_out, rs_in))]
    for name, fn in ops:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        t = time.perf_counter()
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 30 * 1e3
        if rank == 0:
            print(f"{label:<28} {name:<14} {ms:7.3f} ms  {nbytes / ms / 1e6:6.1f} GB/s per direction", flush=True)
    if rank == 0:
        peer = torch.empty(n, h, dtype=torch.bfloat16, device="cuda:1")
        for _ in range(5):
            peer.copy_(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(30):
            peer.copy_(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / 30 * 1e3
        print(f"{label:<28} {'peer_copy':<14} {ms:7.3f} ms  {nbytes / ms / 1e6:6.1f} GB/s one direction", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    label = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 5120
    mp.spawn(run, args=(2, 29555, label, n, h), nprocs=2, join=True)
