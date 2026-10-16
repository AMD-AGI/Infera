#!/usr/bin/env python3
"""GPU time breakdown of sglang torch-profiler traces (ROCm), per file, by kernel category.

Also reports completeness signals: kernel / memcpy event counts, GPU busy time vs. trace span,
ProfilerStep count, and the largest gap between consecutive GPU events on the main stream.

Usage: trace_breakdown.py TRACE.json.gz [TRACE.json.gz ...]
"""

import gzip
import json
import re
import sys
from collections import defaultdict

CATEGORIES = [
    ("comm", r"nccl|rccl|all_?reduce|all_?gather|reduce_?scatter|cross_device|custom_ar|quick_?reduce|quickreduce|allreduce|ncclDevKernel|mscclpp"),
    ("moe", r"moe|a4w4|ne257|fused_moe|topk_|grouped_topk|expert|ck_moe|fmoe|flydsl|moe_sorting|scatter_reduce"),
    ("attn_dsa", r"mla|attention|attn|flash|paged|decode_attention|extend_attention|sparse|indexer|nsa|dsa|topk_transform|fp8_index|deepgemm_fp8_paged|lightning"),
    ("gemm", r"gemm|Cijk|hipblaslt|matmul|mm_|wvSplitK|wv_splitk|bf16_|_gemm|hgemm|sgemm|tensile|ck_gemm|batched_gemm|bmm"),
    ("kv_copy", r"kv_cache|transfer_kv|store_kv|set_kv|copy_kv|load_kv|kvcache|hicache|page_first"),
    ("norm_rope_act", r"norm|rope|rotary|silu|gelu|act_and_mul|swiglu|softmax"),
    ("sampling_spec", r"sample|sampling|argmax|eagle|draft|verify|tree|accept|logits|top_p|top_k"),
    ("elementwise", r"elementwise|vectorized|copy|fill|cat|index|scatter|gather|reduce|unrolled|where|add|mul"),
]
PATTERNS = [(name, re.compile(rx, re.I)) for name, rx in CATEGORIES]


def category(name: str) -> str:
    for cat, rx in PATTERNS:
        if rx.search(name):
            return cat
    return "other"


def analyze(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_op")]
    memcpy = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("gpu_memcpy", "gpu_memset")]
    steps = [e for e in events if e.get("ph") == "X" and str(e.get("name", "")).startswith("ProfilerStep")]
    by_cat = defaultdict(float)
    by_name = defaultdict(float)
    for e in kernels:
        by_cat[category(e.get("name", ""))] += e.get("dur", 0)
        by_name[e.get("name", "")[:90]] += e.get("dur", 0)
    copy_kind = defaultdict(float)
    for e in memcpy:
        by_cat["memcpy/memset"] += e.get("dur", 0)
        copy_kind[str(e.get("name", ""))[:40]] += e.get("dur", 0)
    gpu = kernels + memcpy
    span = (max(e["ts"] + e.get("dur", 0) for e in gpu) - min(e["ts"] for e in gpu)) if gpu else 0
    busy = sum(by_cat.values())
    streams = defaultdict(list)
    for e in kernels:
        streams[(e.get("pid"), e.get("tid"))].append(e)
    main = max(streams.values(), key=len) if streams else []
    main.sort(key=lambda e: e["ts"])
    main_cat = defaultdict(float)
    for e in main:
        main_cat[category(e.get("name", ""))] += e.get("dur", 0)
    gaps = [b["ts"] - (a["ts"] + a.get("dur", 0)) for a, b in zip(main, main[1:])]
    # Per scheduler.run_batch (GPU-side annotation): batch type from the inner step[...] annotation.
    ann = sorted((e for e in events if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation"), key=lambda e: e["ts"])
    runs = [e for e in ann if e.get("name") == "scheduler.run_batch"]
    inner = [e for e in ann if str(e.get("name", "")).startswith("step[")]
    kernels_sorted = sorted(main, key=lambda e: e["ts"])
    import bisect
    kts = [e["ts"] for e in kernels_sorted]
    batch = defaultdict(lambda: {"n": 0, "dur": 0.0, "cat": defaultdict(float), "bs": []})
    for r in runs:
        t0, t1 = r["ts"], r["ts"] + r.get("dur", 0)
        names = [e["name"] for e in inner if t0 <= e["ts"] < t1]
        kind = ("EXTEND" if any(n.startswith("step[EXTEND") for n in names) else
                "VERIFY" if any(n.startswith("step[VERIFY") for n in names) else
                "IDLE" if names and all(n.startswith(("step[IDLE", "step[DRAFT bs=0")) for n in names) else "OTHER")
        b = batch[kind]
        b["n"] += 1
        b["dur"] += r.get("dur", 0)
        for n in names:
            m = re.search(r"bs=(\d+)", n)
            if m and n.startswith(("step[VERIFY", "step[EXTEND")):
                b["bs"].append(int(m.group(1)))
        i = bisect.bisect_left(kts, t0)
        while i < len(kts) and kts[i] < t1:
            e = kernels_sorted[i]
            b["cat"][category(e.get("name", ""))] += e.get("dur", 0)
            i += 1
    return {"batches": {k: {"n": v["n"], "dur": v["dur"], "cat": dict(v["cat"]), "bs": v["bs"]} for k, v in batch.items()},
            "file": path.rsplit("/", 1)[-1], "kernels": len(kernels), "memcpy": len(memcpy), "steps": len(steps),
            "span_ms": span / 1000, "busy_ms": busy / 1000, "streams": len(streams),
            "max_gap_ms": (max(gaps) / 1000) if gaps else 0, "by_cat": dict(by_cat), "main_cat": dict(main_cat),
            "main_busy_ms": sum(main_cat.values()) / 1000, "main_span_ms": ((main[-1]["ts"] + main[-1].get("dur", 0) - main[0]["ts"]) / 1000) if main else 0,
            "copy_kind": dict(copy_kind),
            "top": sorted(by_name.items(), key=lambda kv: -kv[1])[:12]}


def main() -> int:
    for path in sys.argv[1:]:
        r = analyze(path)
        busy = r["busy_ms"] or 1
        print(f"== {r['file']}: {r['kernels']} kernels, {r['memcpy']} memcpy, {r['steps']} ProfilerStep, "
              f"{r['streams']} kernel streams, GPU busy {r['busy_ms']:.1f} ms over span {r['span_ms']:.1f} ms, "
              f"max gap on main stream {r['max_gap_ms']:.1f} ms")
        mb = r["main_busy_ms"] or 1
        print(f"   main stream: busy {r['main_busy_ms']:.1f} ms of span {r['main_span_ms']:.1f} ms -> "
              + "  ".join(f"{k} {100 * v / 1000 / mb:.1f}%" for k, v in sorted(r["main_cat"].items(), key=lambda kv: -kv[1])))
        if r["copy_kind"]:
            print("   copies (side streams): " + ", ".join(f"{k} {v / 1000:.1f} ms" for k, v in sorted(r["copy_kind"].items(), key=lambda kv: -kv[1])[:4]))
        for kind, b in sorted(r["batches"].items(), key=lambda kv: -kv[1]["dur"]):
            k_busy = sum(b["cat"].values()) or 1
            bs = (sum(b["bs"]) / len(b["bs"])) if b["bs"] else 0
            print(f"   batch {kind:6s} x{b['n']:4d}: mean {b['dur'] / max(b['n'], 1) / 1000:7.1f} ms (bs {bs:.1f}), {b['dur'] / 1000:8.1f} ms total; "
                  + "  ".join(f"{k} {100 * v / k_busy:.0f}%" for k, v in sorted(b["cat"].items(), key=lambda kv: -kv[1])[:6]))
        for name, dur in r["top"][:8]:
            print(f"     {dur / 1000:8.1f} ms  {category(name):13s} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
