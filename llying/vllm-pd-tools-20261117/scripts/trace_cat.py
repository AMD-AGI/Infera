#!/usr/bin/env python3
"""GPU time per decode step by kernel category, from one torch-profiler trace (Chrome JSON, .json or .json.gz).
  trace_cat.py TRACE [--steps-kernel REGEX] [--top N]
Steps are counted by the kernel matched with --steps-kernel (default: the final-norm / lm-head GEMM is not unique, so
count the MoE routing top-k launches and divide by the number of MoE layers, --moe-layers 75 + 1 MTP layer per draft
step). Categories follow the sglang breakdowns in dp-graph-variant-fix-aus-20261103/RESULT.zh-CN.md 4.1 / 4.6.
"""
import argparse
import gzip
import json
import re
from collections import defaultdict

CATS = [
    ("moe_gemm", r"moe.*(gemm|fp4|a4w4|mxfp4|ck_moe|fmoe|stage[12])|fused_moe|mfma_moe|gemm[12]_a4w4"),
    ("moe_misc", r"moe_sort|moe_align|topk|grouped_topk|biased_grouped|moe_sum|moe_reduction|opus_moe"),
    ("sparse_mla", r"mla|sparse_attn|sparse_mla|pa_decode_sparse|decode_fwd"),
    ("indexer", r"mqa_logits|indexer|top_k_per_row|topk_transform|paged_mqa|deep_select"),
    ("comm", r"allreduce|all_reduce|allgather|all_gather|reduce_scatter|rccl|nccl|cross_device|custom_ar"),
    ("dense_gemm", r"gemm|gemv|matmul|Cijk|bf16gemm|hipblaslt|wvSplitK|skinny|bpreshuffle|f8f8bf16"),
    ("norm_rope", r"rms|norm|rope|rotary"),
    ("quant", r"quant|scaled_fp8|cvt|fp8"),
]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--steps", type=float, default=0, help="number of decode steps in the window (else estimated)")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    ev = load(a.trace)
    ev = ev.get("traceEvents", ev)
    k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not k:
        print("no GPU kernel events")
        return
    t0 = min(e["ts"] for e in k)
    t1 = max(e["ts"] + e.get("dur", 0) for e in k)
    by_name = defaultdict(lambda: [0.0, 0])
    for e in k:
        by_name[e["name"]][0] += e.get("dur", 0)
        by_name[e["name"]][1] += 1
    cats = defaultdict(float)
    for name, (dur, _) in by_name.items():
        low = name.lower()
        for c, rx in CATS:
            if re.search(rx, low):
                cats[c] += dur
                break
        else:
            cats["other"] += dur
    busy = sum(cats.values())
    steps = a.steps
    if not steps:
        # One sampler / argmax-like kernel per step is not guaranteed; fall back to graph launches if present.
        launches = [e for e in ev if e.get("ph") == "X" and "hipGraphLaunch" in e.get("name", "")]
        steps = max(1, len(launches) / 2) if launches else 1
    span = (t1 - t0) / 1e3
    print(f"window {span:.1f} ms, GPU busy {busy / 1e3:.1f} ms ({100 * busy / 1e3 / span:.0f}%), steps ~{steps:.0f}")
    for c, d in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c:12s} {d / 1e3:9.1f} ms  {d / 1e3 / steps:7.2f} ms/step")
    print("top kernels:")
    for name, (dur, n) in sorted(by_name.items(), key=lambda x: -x[1][0])[: a.top]:
        print(f"  {dur / 1e3:9.1f} ms {n:7d}x {dur / max(n, 1):8.1f} us  {name[:110]}")


if __name__ == "__main__":
    main()
