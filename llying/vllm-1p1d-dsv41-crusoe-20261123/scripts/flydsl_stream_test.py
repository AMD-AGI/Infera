#!/usr/bin/env python3
"""Does AITER's FlyDSL MXFP8 GEMM (the shared expert's GEMM on prefill) run on the caller's current stream?

R15 asked vLLM to put the shared expert on its auxiliary stream; the runner decided MULTI_STREAM_OVERLAPPED (shexp_probe),
yet the profile shows every shared-expert GEMM (flydsl_bpreshuffle_mxfp8_*) on the main stream (EXPERIMENT 4.11).
This launches gemm_a8w8_blockscale_bpreshuffle at the shared expert's gate_up shape (M 16384, N 4608, K 5120) on the
default stream and inside torch.cuda.stream(side), records which stream the profiler sees each kernel on, and times the
side-stream GEMM against a long main-stream matmul to see whether the two overlap. Prints one RESULT line per check.
"""
import collections
import time

import torch
from torch.profiler import ProfilerActivity, profile

import os

M = int(os.environ.get("FLY_M", "8192"))  # 8192 is in AITER's tuned table for the shared expert, as in the engine
N, K = int(os.environ.get("FLY_N", "4608")), int(os.environ.get("FLY_K", "5120"))


def main():
    from aiter import gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.shuffle import shuffle_weight

    dev = torch.device("cuda", 0)
    x = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
    w = shuffle_weight(torch.randn(N, K, device=dev).to(torch.float8_e4m3fn), (16, 16))
    xs = torch.full((M, K // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    ws = torch.full((N, K // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    gemm = lambda: gemm_a8w8_blockscale_bpreshuffle(x, w, xs, ws, dtype=torch.bfloat16)  # noqa: E731
    a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    big = lambda: a @ a  # noqa: E731
    side = torch.cuda.Stream()
    for _ in range(3):
        gemm()
        big()
    torch.cuda.synchronize()

    def timed(fn, n=20):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1000

    t_gemm, t_big = timed(gemm), timed(big)

    def both():
        big()
        with torch.cuda.stream(side):
            gemm()
        torch.cuda.current_stream().wait_stream(side)

    t_both = timed(both)
    print(f"RESULT times: gemm {t_gemm:.3f} ms, matmul {t_big:.3f} ms, matmul + side-stream gemm {t_both:.3f} ms "
          f"(serial would be {t_gemm + t_big:.3f}, full overlap {max(t_gemm, t_big):.3f})", flush=True)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        gemm()
        with torch.cuda.stream(side):
            gemm()
            big()
        torch.cuda.synchronize()
    import json
    import tempfile

    path = tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(path)
    streams = collections.defaultdict(collections.Counter)
    for e in json.load(open(path))["traceEvents"]:
        if e.get("cat") != "kernel":
            continue
        streams[e["name"][:70]][e.get("args", {}).get("stream")] += 1
    print(f"RESULT profiler streams: kernels by stream {dict((k, dict(v)) for k, v in streams.items())} "
          f"(torch stream ids: main {torch.cuda.current_stream().stream_id}, side {side.stream_id})", flush=True)
    print(f"RESULT expectation: the first flydsl GEMM on the main stream, the second flydsl GEMM and the matmul on the "
          f"side stream; flydsl all on one stream means it ignores the current stream", flush=True)


if __name__ == "__main__":
    main()
