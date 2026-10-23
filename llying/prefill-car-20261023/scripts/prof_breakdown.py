"""GPU time breakdown of torch-profiler traces (one per pipeline stage / rank): kernel time by category, GPU busy
share of the profiled span (union of kernel intervals), and the top kernels. Categories are matched on the kernel
name, first match wins (see CATS).
  python3 prof_breakdown.py TRACE.json[.gz] ... [--top 25]"""

import argparse
import collections
import gzip
import json
import re

CATS = [
    ("comm", r"nccl|rccl|all_?reduce|all_?gather|reduce_?scatter|sendrecv|cross_device|quick_?reduce"),
    ("moe", r"moe|fused_experts|expert|topk_softmax|grouped_topk|moe_sorting|biased_grouped"),
    ("dsa_indexer", r"index|fp8_mqa|mqa_logits|paged_mqa|topk"),
    ("attention", r"mla|attn|attention|flash|sparse|decode_|prefill_|merge_state|lse"),
    ("gemm", r"gemm|matmul|cijk|hipblaslt|wvsplitk|ck_tile|_mm_|bmm|gemv|splitk|a8w8|afp4|mxfp4|fp4"),
    ("norm_rope_act", r"rms|norm|rope|rotary|silu|gelu|act_and_mul|swiglu|softmax"),
    ("copy_cast_misc", r"copy|cast|cat|fill|elementwise|vectorized|unrolled|reduce|scatter|gather|arange|index"),
]
RX = [(name, re.compile(rx, re.I)) for name, rx in CATS]


def category(kernel):
    for name, rx in RX:
        if rx.search(kernel):
            return name
    return "other"


def load(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)["traceEvents"]


ap = argparse.ArgumentParser()
ap.add_argument("traces", nargs="+")
ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()
for path in a.traces:
    ev = [e for e in load(path) if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not ev:
        print(f"{path}: no GPU kernel events")
        continue
    ivs = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev)
    span = ivs[-1][1] - ivs[0][0]
    busy, cur_s, cur_e = 0.0, *ivs[0]
    for s, e in ivs[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    by_cat, by_name = collections.Counter(), collections.Counter()
    for e in ev:
        by_cat[category(e["name"])] += e["dur"]
        by_name[e["name"]] += e["dur"]
    total = sum(by_cat.values())
    print(f"\n== {path}\nspan {span / 1e3:.0f} ms, GPU busy {busy / 1e3:.0f} ms ({100 * busy / span:.0f}%), "
          f"kernel time {total / 1e3:.0f} ms, {len(ev)} kernels")
    for c, t in by_cat.most_common():
        print(f"  {c:16s} {t / 1e3:9.1f} ms  {100 * t / total:5.1f}%")
    print(f"  top {a.top} kernels:")
    for n, t in by_name.most_common(a.top):
        print(f"    {t / 1e3:8.1f} ms  {100 * t / total:5.1f}%  [{category(n)}] {n[:120]}")
