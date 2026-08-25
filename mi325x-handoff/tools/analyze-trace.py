#!/usr/bin/env python3
"""Summarise a torch profiler chrome trace into where a decode step actually goes.

Reports GPU kernel time grouped by a coarse category (collective / GEMM / MoE /
attention / elementwise), because the question this answers is which of those
dominates a step, not which individual kernel is slowest.

    python3 analyze-trace.py <trace.json.gz> [--top 25]
"""
import argparse
import gzip
import json
import re
from collections import defaultdict

# Ordered: first match wins, so the specific patterns must precede the generic ones.
CATEGORIES = [
    ("collective", r"rccl|nccl|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all|broadcast"),
    ("moe",        r"moe|expert|topk_softmax|grouped_gemm|fused_moe|silu_and_mul"),
    ("attention",  r"attn|attention|flash|paged|mla|dsa|indexer|rope|rotary"),
    ("gemm",       r"gemm|matmul|Cijk|cutlass|hipblas|rocblas|wvSplitK|skinny"),
    ("norm",       r"rms_?norm|layer_?norm|norm_kernel"),
    ("quant",      r"quant|scaled_fp8|fp8_|cast"),
    ("sampling",   r"sample|argmax|topk|softmax|logits|verify|draft|eagle"),
    ("copy",       r"memcpy|copy|cat_|index_|gather|scatter"),
    ("elementwise", r"elementwise|vectorized|add|mul|fill|zero"),
]


def categorize(name: str) -> str:
    low = name.lower()
    for cat, pat in CATEGORIES:
        if re.search(pat, low):
            return cat
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as fh:
        events = json.load(fh)["traceEvents"]

    by_kernel = defaultdict(lambda: [0.0, 0])
    by_cat = defaultdict(float)
    gpu_us = 0.0
    span_lo, span_hi = float("inf"), 0.0

    for ev in events:
        if ev.get("ph") != "X":
            continue
        ts, dur = ev.get("ts", 0), ev.get("dur", 0)
        span_lo, span_hi = min(span_lo, ts), max(span_hi, ts + dur)
        if ev.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        name = ev.get("name", "?")
        by_kernel[name][0] += dur
        by_kernel[name][1] += 1
        by_cat[categorize(name)] += dur
        gpu_us += dur

    wall_us = span_hi - span_lo if span_hi > span_lo else 0.0
    print(f"trace span      {wall_us/1000:10.1f} ms")
    print(f"GPU busy        {gpu_us/1000:10.1f} ms   ({gpu_us/wall_us*100:.1f}% of span)"
          if wall_us else "")
    print(f"GPU idle        {(wall_us-gpu_us)/1000:10.1f} ms   "
          f"({(wall_us-gpu_us)/wall_us*100:.1f}% of span)" if wall_us else "")

    print("\n--- GPU time by category ---")
    for cat, us in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:14} {us/1000:9.2f} ms  {us/gpu_us*100:5.1f}%")

    print(f"\n--- top {args.top} kernels ---")
    print(f"  {'ms':>9} {'calls':>7} {'us/call':>9}  kernel")
    ranked = sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[: args.top]
    for name, (us, n) in ranked:
        print(f"  {us/1000:9.2f} {n:7d} {us/n:9.1f}  {name[:96]}")


if __name__ == "__main__":
    main()
