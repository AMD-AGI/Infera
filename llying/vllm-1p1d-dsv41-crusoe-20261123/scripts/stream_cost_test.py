#!/usr/bin/env python3
"""What does a second stream cost in the engine image? (EXPERIMENT 4.11, after flydsl_stream_test.py)

flydsl_stream_test.py found a 0.73 ms matmul on the main stream plus a 0.46 ms FlyDSL GEMM on a side stream taking
5.0 ms per iteration, 4x the serial 1.19 ms. Every engine log has "[queue-intercept] ... falling back to system-memory
ring", so a queue-interception layer sits under every HIP queue. This separates the cost: an event round trip with no
work on the side stream; the GEMM alone on the side stream; main and side work with and without the cross-stream wait;
and prints the environment that could enable the interception. One RESULT line per measurement (ms per iteration).
"""
import os
import time

import torch

M, N, K = 16384, 4608, 5120


def timed(fn, n=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    keys = ("HSA", "ROCP", "ROCPROF", "INTERCEPT", "TOOLS", "HIP_", "GPU_MAX", "AMD_", "LD_PRELOAD")
    env = {k: v for k, v in os.environ.items() if any(s in k for s in keys)}
    print(f"RESULT env: {env}", flush=True)
    from aiter import gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.shuffle import shuffle_weight

    dev = torch.device("cuda", 0)
    x = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
    w = shuffle_weight(torch.randn(N, K, device=dev).to(torch.float8_e4m3fn), (16, 16))
    xs = torch.full((M, K // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    ws = torch.full((N, K // 32), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
    a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    gemm = lambda: gemm_a8w8_blockscale_bpreshuffle(x, w, xs, ws, dtype=torch.bfloat16)  # noqa: E731
    big = lambda: a @ a  # noqa: E731
    main_s = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    ev1, ev2 = torch.cuda.Event(), torch.cuda.Event()

    def event_round_trip():
        ev1.record(main_s)
        side.wait_event(ev1)
        ev2.record(side)
        main_s.wait_event(ev2)

    def side_gemm_only():
        side.wait_stream(main_s)
        with torch.cuda.stream(side):
            gemm()
        main_s.wait_stream(side)

    def both_with_wait():
        side.wait_stream(main_s)
        big()
        with torch.cuda.stream(side):
            gemm()
        main_s.wait_stream(side)

    def both_no_wait():
        big()
        with torch.cuda.stream(side):
            gemm()

    def serial():
        big()
        gemm()

    for name, fn in (("gemm on main", gemm), ("matmul on main", big), ("serial matmul + gemm", serial),
                     ("event round trip, no work", event_round_trip), ("gemm on side + waits", side_gemm_only),
                     ("matmul main + gemm side + waits", both_with_wait),
                     ("matmul main + gemm side, no cross-stream wait", both_no_wait)):
        print(f"RESULT {name}: {timed(fn):.3f} ms", flush=True)


if __name__ == "__main__":
    main()
