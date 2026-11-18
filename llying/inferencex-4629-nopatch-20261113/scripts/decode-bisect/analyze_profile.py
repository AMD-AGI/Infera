#!/usr/bin/env python3
"""Per-kernel GPU time from SGLang torch-profiler traces, side by side for two variants.

usage: analyze_profile.py <out/variantA> <out/variantB> [--rank 0] [--top 40]
Sums kernel durations per (normalized) kernel name over all decode steps in one rank's trace,
divides by the number of profiled steps, and prints both variants and the delta.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re

CATS = [
    ("indexer.paged_mqa_logits", r"mqa|paged_mqa|indexer_logits|fp8_mqa|deepgemm.*mqa|mqa_logits"),
    ("indexer.topk", r"topk|top_k|radix_select|fast_topk"),
    ("sparse_mla/attn", r"mla|flash_attn|attention|sparse_attn|fwd_kernel|decode_attn|_fwd_|paged_attention|unified_attention"),
    ("moe", r"moe|fused_experts|expert|topk_softmax|grouped_gemm|flydsl|ck_moe|fmoe"),
    ("gemm", r"gemm|Cijk|matmul|bmm|hipblaslt|mfma|wvSplitK|ck_tile.*Gemm|_gemm_"),
    ("comm", r"allreduce|all_reduce|allgather|all_gather|reduce_scatter|rccl|nccl|cross_device|custom_ar|gatherv"),
    ("norm/rope/act", r"norm|rope|rotary|silu|gelu|act_and_mul|swiglu"),
    ("quant", r"quant|fp8|mxfp4|scale"),
    ("copy/elementwise", r"copy|elementwise|fill|cat|index|gather|scatter|memset|Memcpy|vectorized|reduce"),
]


def norm(name):
    name = re.sub(r"<.*", "", name)
    name = re.sub(r"\(.*", "", name)
    name = re.sub(r"_\d+(_\d+)*$", "", name)
    return name[:110]


def cat_of(name):
    low = name.lower()
    for c, pat in CATS:
        if re.search(pat, low, re.I):
            return c
    return "other"


def load(vdir, rank):
    files = sorted(glob.glob(os.path.join(vdir, "profile", "**", "*.json*"), recursive=True))
    pick = [f for f in files if re.search(rf"(TP|DP|rank)[-_]?{rank}\b|-{rank}\.trace", os.path.basename(f))] or files[:1]
    if not pick:
        raise SystemExit(f"no trace under {vdir}/profile")
    f = pick[0]
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh:
        d = json.load(fh)
    ev = d["traceEvents"] if isinstance(d, dict) else d
    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    steps = [e for e in ev if e.get("ph") == "X" and re.match(r"ProfilerStep#|decode|forward", str(e.get("name")))]
    per = collections.defaultdict(float)
    cnt = collections.Counter()
    for e in kern:
        n = norm(e["name"])
        per[n] += e.get("dur", 0)
        cnt[n] += 1
    span = (max(e["ts"] + e.get("dur", 0) for e in kern) - min(e["ts"] for e in kern)) if kern else 0
    return f, per, cnt, span, len([s for s in steps if str(s.get("name")).startswith("ProfilerStep#")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    fa, pa, ca, sa, na = load(a.a, a.rank)
    fb, pb, cb, sb, nb = load(a.b, a.rank)
    print(f"A: {fa}\n   kernels {sum(ca.values())}, GPU busy {sum(pa.values())/1e3:.1f} ms, span {sa/1e3:.1f} ms, steps {na}")
    print(f"B: {fb}\n   kernels {sum(cb.values())}, GPU busy {sum(pb.values())/1e3:.1f} ms, span {sb/1e3:.1f} ms, steps {nb}")
    ta, tb = sum(pa.values()), sum(pb.values())
    print("\nby category (ms, share of busy time)")
    catA, catB = collections.defaultdict(float), collections.defaultdict(float)
    for n, v in pa.items():
        catA[cat_of(n)] += v
    for n, v in pb.items():
        catB[cat_of(n)] += v
    for c in sorted(set(catA) | set(catB), key=lambda c: -(catB.get(c, 0) - catA.get(c, 0))):
        va, vb = catA.get(c, 0) / 1e3, catB.get(c, 0) / 1e3
        print(f"  {c:26s} A {va:9.1f} ({va*1e3/ta:5.1%})   B {vb:9.1f} ({vb*1e3/tb:5.1%})   delta {vb-va:+9.1f}")
    print(f"\ntop {a.top} kernels by |delta| (ms total in profile; calls)")
    names = set(pa) | set(pb)
    for n in sorted(names, key=lambda n: -abs(pb.get(n, 0) - pa.get(n, 0)))[: a.top]:
        va, vb = pa.get(n, 0) / 1e3, pb.get(n, 0) / 1e3
        print(f"  {vb-va:+9.1f}  A {va:8.1f} ({ca.get(n,0):6d})  B {vb:8.1f} ({cb.get(n,0):6d})  [{cat_of(n)}] {n}")


if __name__ == "__main__":
    main()
