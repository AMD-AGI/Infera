#!/usr/bin/env python3
"""How much of a shared-expert FP8 GEMM hides under prefill's MoE all-gather? (EXPERIMENT 4.11)

Two ranks (torchrun --nproc_per_node 2). Each step mimics one prefill MoE layer: an all-gather of the 16384 x 5120 bf16
hidden states (RCCL, ~3.3 ms in the engine) on the main stream, and the shared expert's two FP8 GEMMs (gate_up 4608 x 5120,
down 5120 x 2304) either after it on the main stream or on a side stream while it runs. stream_cost_test.py showed two
compute-bound kernels gain nothing from running together; RCCL is the case the overlap is meant for. RESULT lines give
ms per layer for: all-gather alone, GEMMs alone, serial, overlapped.
"""
import os
import time

import torch
import torch.distributed as dist

M, H, I = 16384, 5120, 2304


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    from aiter import gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.shuffle import shuffle_weight

    dev = torch.device("cuda", torch.cuda.current_device())
    hid = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    gathered = torch.empty(2 * M, H, device=dev, dtype=torch.bfloat16)

    def fp8_gemm_inputs(m, n, k):
        x = torch.randn(m, k, device=dev).to(torch.float8_e4m3fn)
        w = shuffle_weight(torch.randn(n, k, device=dev).to(torch.float8_e4m3fn), (16, 16))
        xs = torch.full((m, k // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
        ws = torch.full((n, k // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
        return x, w, xs, ws

    gu = fp8_gemm_inputs(M, 2 * I, H)
    dn = fp8_gemm_inputs(M, H, I)

    def shared_expert():
        gemm_a8w8_blockscale_bpreshuffle(*gu, dtype=torch.bfloat16)
        gemm_a8w8_blockscale_bpreshuffle(*dn, dtype=torch.bfloat16)

    def allgather():
        dist.all_gather_into_tensor(gathered, hid)

    main_s = torch.cuda.current_stream()
    side = torch.cuda.Stream()

    def serial():
        allgather()
        shared_expert()

    def overlapped():
        side.wait_stream(main_s)
        with torch.cuda.stream(side):
            shared_expert()
        allgather()
        main_s.wait_stream(side)

    def timed(fn, n=40):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000

    res = {name: timed(fn) for name, fn in (("all-gather alone", allgather), ("shared-expert GEMMs alone", shared_expert),
                                             ("serial", serial), ("overlapped (GEMMs on a side stream)", overlapped))}
    if rank == 0:
        for k, v in res.items():
            print(f"RESULT {k}: {v:.3f} ms per layer", flush=True)
        saved = res["serial"] - res["overlapped (GEMMs on a side stream)"]
        print(f"RESULT saved per layer {saved:.3f} ms of {res['shared-expert GEMMs alone']:.3f} ms of GEMMs "
              f"({100 * saved / max(res['serial'], 1e-9):.1f}% of the serial layer); x40 layers per 16k step: "
              f"{40 * saved:.1f} ms", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
