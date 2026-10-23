#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""AgentX 结果的"每路会话"视角：每卡并发、每卡每秒请求、平均输入 / 输出长度、每张 decode 卡的输出。

per-GPU 输入吞吐 = 每秒完成请求 x 平均输入长度 / 卡数，所以这几列能把吞吐拆开看。

用法：lane_metrics.py LABEL=path/to/agentx_conc<C>.json ...
"""

from __future__ import annotations

import json
import sys

HEADER = (
    "| 点 | P+D 卡 | C | 每卡并发 | 每 decode 卡并发 | 每卡 req/s | 平均输入 | 平均输出 "
    "| 输入 tok/s/GPU | 每 decode 卡输出 tok/s | ITL p50 ms | TTFT p50 s | intvty p90 |\n"
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)


def row(label: str, path: str) -> str:
    with open(path) as fh:
        d = json.load(fh)
    r = d["request_metrics"]
    t = r["throughput"]
    lat = r["latency"]
    p, dg = d["num_prefill_gpu"], d["num_decode_gpu"]
    g = p + dg
    rps = d["num_requests_successful"] / t["duration_seconds"]
    return (
        f"| {label} | {p}+{dg} | {d['conc']} | {d['conc'] / g:.1f} | {d['conc'] / dg:.0f} "
        f"| {rps / g:.3f} | {r['tokens']['input']['mean'] / 1e3:.1f}K "
        f"| {r['tokens']['output_actual']['mean']:.0f} | {t['per_gpu']['input_tput_tps']:,.0f} "
        f"| {t['output']['tokens_per_second'] / dg:.0f} | {lat['itl']['p50'] * 1e3:.1f} "
        f"| {lat['ttft']['p50']:.2f} | {lat['intvty']['p90']:.1f} |"
    )


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit(__doc__)
    print(HEADER)
    for arg in argv:
        label, _, path = arg.partition("=")
        print(row(label, path))


if __name__ == "__main__":
    main(sys.argv[1:])
