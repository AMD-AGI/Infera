#!/usr/bin/env python3
"""GPU smoke test for ROCm graph trace completeness; execute inside the image.

Use DEBUG_CLR_GRAPH_PACKET_CAPTURE=false for the validated ROCm 7.2 image.
The default packet path can silently omit the large graph from the trace.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import torch
    x = torch.ones((512, 512), device="cuda")
    y = torch.ones_like(x)
    for _ in range(3):
        z = x @ y
    large, small = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(large):
        for _ in range(1200):
            z = x @ y
    with torch.cuda.graph(small):
        z = x @ y
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            large.replay()
            small.replay()
            torch.cuda.synchronize()
        time.sleep(1)  # Outside the workload; allow asynchronous tracer delivery.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.output))
    events = json.loads(args.output.read_text())["traceEvents"]
    counts = Counter(e["args"].get("correlation") for e in events if e.get("cat") == "kernel")
    result = {"packet_capture": os.getenv("DEBUG_CLR_GRAPH_PACKET_CAPTURE", "default"),
              "expected_kernels": 3603, "observed_kernels": sum(counts.values()),
              "kernels_per_graph_launch": [counts[e["args"]["correlation"]] for e in events
                  if e.get("name") == "hipGraphLaunch"]}
    args.output.with_suffix(".validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if sorted(result["kernels_per_graph_launch"]) != [1, 1, 1, 1200, 1200, 1200]:
        raise RuntimeError("Incomplete HIP graph trace")


if __name__ == "__main__":
    main()
