#!/usr/bin/env python3
"""Single-GPU smoke probe for the fork json-tool, mirroring what the Muse server window does.

Run with the same env as the server (ROCP_TOOL_LIBRARIES, ROCPROFILER_REGISTER_LIBRARY, ...) and
ROCPROFILER_TOOL_OUTPUT_FILE=<out>/trace-<mode>.json. Writes <out>/sidecar-<mode>.json.

  --mode window : capture is NOT traced (server mode "window"); tracing only around 20 replays + eager ops.
  --mode capture: capture is traced, tracing stopped, then restarted for the replays (open question 3:
                  does capture attribution survive stop/start?).
Sequence: pause -> eager warmup -> capture graph (5-6 kernel nodes) -> [start] -> 20 replays (each bracketed
by CLOCK_BOOTTIME host stamps and torch.cuda.Event) -> 10 eager matmuls -> sync -> [stop] -> 5 more replays
(must NOT appear) -> finalize.
"""
import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import torch


def now():
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["window", "capture"], required=True)
    p.add_argument("--sdk-build", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--replays", type=int, default=20)
    a = p.parse_args()
    tool = ctypes.CDLL(str(Path(a.sdk_build) / "lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so"))
    tool.json_tool_pause_until_graph.restype = None
    tool.json_tool_finalize_idle_worker.restype = None
    side = {"mode": a.mode, "pid": os.getpid(), "clock": "CLOCK_BOOTTIME",
            "sdk_maps": sorted({l.split()[-1] for l in open("/proc/self/maps") if "rocprofiler-sdk" in l}),
            "torch": torch.__version__, "hip": torch.version.hip, "device": torch.cuda.get_device_name(0)}
    tool.json_tool_pause_until_graph()

    dev = torch.device("cuda:0")
    x = torch.randn(512, 1024, device=dev, dtype=torch.bfloat16)
    w = torch.randn(1024, 2048, device=dev, dtype=torch.bfloat16)
    dst = torch.empty(512, 2048, device=dev, dtype=torch.bfloat16)
    acc = torch.empty((), device=dev, dtype=torch.float32)

    def body():
        y = x @ w
        y = y + 1.0
        y = torch.relu(y)
        dst.copy_(y)
        acc.copy_(y.float().sum())

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    if a.mode == "capture":
        side["capture_start_rc"] = tool.json_tool_start()
    side["capture_t0"] = now()
    with torch.cuda.graph(g):
        body()
    torch.cuda.synchronize()
    side["capture_t1"] = now()
    if a.mode == "capture":
        side["capture_stop_rc"] = tool.json_tool_stop()

    # window
    side["window_start_rc"] = tool.json_tool_start()
    side["window_t0"] = now()
    reps = []
    for i in range(a.replays):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        h0 = now()
        g.replay()
        h1 = now()
        e1.record()
        reps.append([h0, h1, e0, e1])
    torch.cuda.synchronize()
    eager_t0 = now()
    for _ in range(10):
        _ = x @ w
    torch.cuda.synchronize()
    eager_t1 = now()
    side["window_t1"] = now()
    side["window_stop_rc"] = tool.json_tool_stop()
    side["replays"] = [{"host_t0": h0, "host_t1": h1, "event_ms": e0.elapsed_time(e1)} for h0, h1, e0, e1 in reps]
    side["eager"] = {"host_t0": eager_t0, "host_t1": eager_t1, "count": 10}

    # after stop: must not be traced
    side["post_t0"] = now()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    side["post_t1"] = now()

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"sidecar-{a.mode}.json").write_text(json.dumps(side, indent=1))
    tool.json_tool_finalize_idle_worker()
    print(json.dumps({k: v for k, v in side.items() if k not in ("replays",)}, indent=1))


if __name__ == "__main__":
    main()
