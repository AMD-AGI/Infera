#!/usr/bin/env python3
"""Per-step kernel-time breakdown of an sglang torch-profiler trace (one GPU).

Steps are the sglang span annotations ``step[VERIFY bs=..]``, ``step[DRAFT ..]``,
``step[DECODE ..]``, ``step[EXTEND ..]`` (utils/profile_utils.py
build_step_span_name). A kernel belongs to the step whose CPU span contains its
launch (kernel ``args.correlation`` -> runtime call -> span on the same thread);
kernels of a CUDA-graph replay share the graph launch's correlation. If GPU-side
step annotations (``gpu_user_annotation``) exist, a time-overlap attribution is
reported as a cross-check.

Categories are first-match regexes on the lower-cased kernel name; check them
against the "uncategorized top kernels" list and override with
``--cat NAME=REGEX`` (prepended, so it wins).

Usage: trace_step_breakdown.yihou.py TRACE.json.gz [--cat k=re ...] [--top 15]
"""
import argparse
import collections
import gzip
import json
import re
import statistics as st

DEFAULT_CATS = [
    # gfx950 fused DSA indexer (jit/csrc/dsa_gfx950): decode / target verify, rows <= 48.
    ("idx_fused_proj", r"dual_gemv_kernel|qk_rope_hadamard_quant"),
    ("idx_fused_logits", r"logits_hist"),
    ("idx_fused_topk", r"k_scatter"),
    ("moe_route", r"grouped_topk|topk_softmax|moe_align|moe_sort|biased_grouped|routing"),
    # Standard (unfused) indexer path: aiter paged MQA + top-k / transform kernels.
    ("idx_unfused_logits", r"paged_mqa|mqa_logits|fp8_mqa"),
    ("idx_unfused_topk", r"top_?k|radix_select|topk_transform"),
    ("idx_unfused_prep", r"indexer|index_k|k_quant_and_cache|act_quant|hadamard"),
    ("attention", r"sparse_mla|mla_decode|flash_mla|flashmla|splitk|decode_attention|mla_fwd|paged_attention|fmha"),
    ("dcp_merge", r"_lse|lse_|combine|transform_index|localize|owned|dcp"),
    ("comm", r"nccl|rccl|all_?gather|all_?to_?all|reduce_?scatter|all_?reduce|cross_device|custom_?ar|quick_?ar|oneshot|twoshot"),
    ("moe", r"moe|fmoe|fused_experts|expert"),
    ("gemm", r"cijk|gemm|matmul|hipblaslt|wvsplitk|bmm"),
    ("kv_write", r"set_mla_kv|kv_buffer|store_index|quant_and_cache|index_k|cache"),
    ("norm_rope_act", r"norm|rope|rotary|act_and_mul|silu|gelu"),
    ("copy_misc", r"elementwise|copy|fill|cat|index|scatter|gather|memcpy|memset"),
]
STEP = re.compile(r"^step\[(\w+)")


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]


def categorize(name, cats):
    low = name.lower()
    for cat, rx in cats:
        if rx.search(low):
            return cat
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--cat", action="append", default=[])
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    cats = [(c.split("=", 1)[0], re.compile(c.split("=", 1)[1])) for c in args.cat]
    cats += [(n, re.compile(r)) for n, r in DEFAULT_CATS]

    ev = load(args.trace)
    kernels = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    runtime = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    spans = [e for e in ev if e.get("cat") == "user_annotation" and STEP.match(e["name"])]
    gspans = [
        e for e in ev if e.get("cat") == "gpu_user_annotation" and STEP.match(e["name"])
    ]
    by_corr = collections.defaultdict(list)
    for k in kernels:
        c = k.get("args", {}).get("correlation")
        if c is not None:
            by_corr[c].append(k)

    # Trace-level busy / gaps (union over streams).
    iv = sorted((k["ts"], k["ts"] + k["dur"]) for k in kernels)
    busy, (cs, ce) = 0.0, iv[0]
    for s, e in iv[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = iv[-1][1] - iv[0][0]
    print(f"trace: kernels={len(kernels)} GPU span {span / 1e3:.1f} ms busy {busy / 1e3:.1f} ms "
          f"({100 * busy / span:.1f}%) step spans={len(spans)} gpu step spans={len(gspans)}")

    # CPU-span attribution via launch correlation.
    rt_by_tid = collections.defaultdict(list)
    for r in runtime:
        rt_by_tid[(r.get("pid"), r.get("tid"))].append(r)
    for v in rt_by_tid.values():
        v.sort(key=lambda r: r["ts"])
    per_stage = collections.defaultdict(list)  # stage -> [ {cat: us} per step ]
    seen = set()
    for sp in sorted(spans, key=lambda e: e["ts"]):
        stage = STEP.match(sp["name"]).group(1)
        acc = collections.Counter()
        for r in rt_by_tid.get((sp.get("pid"), sp.get("tid")), []):
            if r["ts"] < sp["ts"]:
                continue
            if r["ts"] > sp["ts"] + sp["dur"]:
                break
            c = r.get("args", {}).get("correlation")
            for k in by_corr.get(c, ()):
                if id(k) in seen:
                    continue
                seen.add(id(k))
                acc[categorize(k["name"], cats)] += k["dur"]
        per_stage[stage].append(acc)

    all_cats = [c for c, _ in cats] + ["other"]
    for stage, steps in sorted(per_stage.items()):
        tot = [sum(s.values()) for s in steps]
        if not tot or max(tot) == 0:
            print(f"\n{stage}: {len(steps)} steps, no kernels attributed")
            continue
        print(f"\n{stage}: steps={len(steps)} kernel-ms/step mean {st.mean(tot) / 1e3:.2f} "
              f"median {st.median(tot) / 1e3:.2f}")
        for c in all_cats:
            vals = [s.get(c, 0) for s in steps]
            m = st.mean(vals)
            if m > 0:
                print(f"  {c:15s} {m / 1e3:8.3f} ms/step  {100 * m / st.mean(tot):5.1f}%")

    if gspans:
        print("\ncross-check (GPU step annotations, time overlap):")
        gk = sorted(kernels, key=lambda k: k["ts"])
        per = collections.defaultdict(list)
        for g in gspans:
            stage = STEP.match(g["name"]).group(1)
            t0, t1 = g["ts"], g["ts"] + g["dur"]
            per[stage].append(sum(k["dur"] for k in gk if t0 <= k["ts"] < t1))
        for stage, v in sorted(per.items()):
            print(f"  {stage}: steps={len(v)} kernel-ms/step mean {st.mean(v) / 1e3:.2f}")
        for g in gspans:
            t0, t1 = g["ts"], g["ts"] + g["dur"]
            acc = collections.Counter()
            for k in gk:
                if t0 <= k["ts"] < t1:
                    acc[categorize(k["name"], cats)] += k["dur"]
            tot = sum(acc.values()) or 1
            print(f"  {g['name'][:40]} at +{(t0 - iv[0][0]) / 1e3:.1f} ms:")
            for c, v in acc.most_common():
                print(f"    {c:20s} {v / 1e3:8.3f} ms  {100 * v / tot:5.1f}%")

    # Graph-replay attribution: kernels replayed from one CUDA graph share the
    # hipGraphLaunch correlation id. Group by launch; a launch whose kernels
    # include >= 40 attention kernels is a target forward (78 layers), else draft.
    launches = {
        r["args"]["correlation"]: r
        for r in runtime
        if "GraphLaunch" in r["name"] and "correlation" in r.get("args", {})
    }
    if launches:
        print("\nper graph launch (kernels sharing the launch correlation):")
        per_kind = collections.defaultdict(list)
        traced = 0
        for corr, r in sorted(launches.items(), key=lambda x: x[1]["ts"]):
            ks = by_corr.get(corr, [])
            if not ks:
                continue
            traced += 1
            acc = collections.Counter()
            for k in ks:
                acc[categorize(k["name"], cats)] += k["dur"]
            n_attn = sum(1 for k in ks if categorize(k["name"], cats) == "attention")
            kind = "target" if n_attn >= 40 else "draft"
            per_kind[kind].append((len(ks), acc))
        print(f"  graph launches={len(launches)} with traced kernels={traced} "
              "(launches without kernels were not captured by the profiler)")
        for kind, items in sorted(per_kind.items()):
            tots = [sum(a.values()) for _, a in items]
            print(f"  {kind}: launches={len(items)} kernels/launch mean "
                  f"{st.mean(n for n, _ in items):.0f} kernel-ms/launch mean {st.mean(tots) / 1e3:.2f}")
            for c in all_cats:
                m = st.mean(a.get(c, 0) for _, a in items)
                if m > 0:
                    print(f"    {c:20s} {m / 1e3:8.3f} ms  {100 * m / st.mean(tots):5.1f}%")

    unc = collections.Counter()
    for k in kernels:
        if categorize(k["name"], cats) == "other":
            unc[k["name"][:100]] += k["dur"]
    print(f"\nuncategorized top kernels (whole trace, refine with --cat):")
    for n, t in unc.most_common(args.top):
        print(f"  {t / 1e3:8.2f} ms  {n}")
    top = collections.Counter()
    cat_of = {}
    for k in kernels:
        top[k["name"][:100]] += k["dur"]
        cat_of[k["name"][:100]] = categorize(k["name"], cats)
    print(f"\ntop kernels (whole trace):")
    for n, t in top.most_common(args.top):
        print(f"  {t / 1e3:8.2f} ms {100 * t / busy:5.1f}% [{cat_of[n]}] {n}")


if __name__ == "__main__":
    main()
