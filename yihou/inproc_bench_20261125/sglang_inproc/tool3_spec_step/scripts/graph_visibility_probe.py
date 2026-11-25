#!/usr/bin/env python3
"""Minimal probe: can torch.profiler attribute per-kernel GPU time inside a
replayed HIP graph, and does DEBUG_CLR_GRAPH_PACKET_CAPTURE change the answer?

No model, no SGLang. Runs in seconds on one GPU. The point is to settle the
core uncertainty of the profiling design before paying a GLM-5.2 startup.

Usage:
    python3 graph_visibility_probe_yihou.py --mode eager
    python3 graph_visibility_probe_yihou.py --mode graph
    DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 python3 graph_visibility_probe_yihou.py --mode graph

We build a body out of THREE kernels with deliberately distinct signatures so
they are individually identifiable in the trace:
  * a large matmul            -> gemm kernel
  * an elementwise chain      -> vectorized elementwise kernel
  * a reduction (sum)         -> reduce kernel
If the profiler can see inside the graph we expect >= 3 distinct GPU kernels
with non-zero self time. If it cannot, we expect one opaque graph-launch entry.
"""
import argparse
import json
import os
import sys

import torch


def build_body(a, b, c):
    """Three structurally different ops so their kernels are distinguishable."""
    def body():
        m = a @ b                      # gemm
        e = torch.tanh(m * 1.5 + 0.25)  # elementwise chain
        c.copy_(e.sum(dim=0))           # reduction
    return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dev = torch.device("cuda:0")
    torch.manual_seed(1234)
    a = torch.randn(args.size, args.size, device=dev, dtype=torch.bfloat16)
    b = torch.randn(args.size, args.size, device=dev, dtype=torch.bfloat16)
    c = torch.zeros(args.size, device=dev, dtype=torch.bfloat16)
    body = build_body(a, b, c)

    # Warm up on a side stream, exactly as torch's graph docs require.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = None
    if args.mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        torch.cuda.synchronize()

    run = (lambda: graph.replay()) if graph is not None else body

    # A few unprofiled reps so any lazy init is out of the window.
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(args.iters):
            run()
        torch.cuda.synchronize()

    events = prof.key_averages()
    gpu_events = []
    for e in events:
        # device_time_total is the modern name; fall back for older torch.
        self_dev = getattr(e, "self_device_time_total", None)
        if self_dev is None:
            self_dev = getattr(e, "self_cuda_time_total", 0.0)
        if self_dev and self_dev > 0:
            gpu_events.append({"name": e.key, "self_gpu_us": round(self_dev, 2),
                               "count": e.count})
    gpu_events.sort(key=lambda x: -x["self_gpu_us"])

    report = {
        "mode": args.mode,
        "iters": args.iters,
        "size": args.size,
        "DEBUG_CLR_GRAPH_PACKET_CAPTURE": os.environ.get(
            "DEBUG_CLR_GRAPH_PACKET_CAPTURE", "<unset>"),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0),
        "num_distinct_gpu_entries": len(gpu_events),
        "total_self_gpu_us": round(sum(e["self_gpu_us"] for e in gpu_events), 2),
        "gpu_entries": gpu_events[:20],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
