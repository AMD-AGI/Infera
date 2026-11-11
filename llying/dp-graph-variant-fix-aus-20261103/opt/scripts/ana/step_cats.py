#!/usr/bin/env python3
"""Where a decode step's GPU time goes: kernel time per verify / draft / draft_extend graph launch, by category, grouped
by the launch's captured batch, plus the top kernels of each graph kind. Uses kdeep.load (one file per DP rank).

Usage: step_cats.py [--skip-ms 60] [--top 30] TRACE_RANK*.json.gz ...
"""
import argparse
import collections
import multiprocessing as mp
import re
import statistics as st
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
import kdeep  # noqa: E402

CATS = [
    ("mega_moe", r"mega_moe_tp"),
    ("moe_other", r"gemm1_a4w4|gemm2_a4w4|mfma_moe|moe_sort|moe_reduction|topk_reg|append_shared|opus_moe|fused_moe|moe2_|grouped_topk|biased_grouped"),
    ("sparse_mla", r"_sparse_mla_"),
    ("mla_bmm", r"_fused_fp8_bmm_rope_cat_and_cache_mla|_batched_gemm_a8w8"),
    ("indexer", r"mqa_logits|_topk|topk_|hadamard|indexer|kn_entry|_k_cache|fp8_index"),
    ("comm", r"nccl|cross_device_reduce|all_gather|allgather|reduce_scatter|allreduce|all_reduce|custom_ar"),
    ("dense_gemm", r"Cijk|hgemm|gemm_a16|bf16gemm|gemm_a8w8|_gemm|wvSplitK|skinny"),
    ("norm", r"rmsnorm|layernorm|rms_norm"),
    ("quant/act/rope", r"quant|rope|silu|act_and_mul|swiglu|gelu"),
    ("copy/elementwise", r"elementwise|copy|fill|memcpy|index|cat|scatter|gather|reduce_kernel|vectorized"),
    ("other", r"."),
]
CRE = [(n, re.compile(p, re.I)) for n, p in CATS]


def cat(name):
    for n, r in CRE:
        if r.search(name):
            return n
    return "other"


def bucket(bs):
    if bs is None:
        return "?"
    for hi in (8, 16, 24, 32, 48, 64, 96, 128):
        if bs <= hi:
            return f"<={hi}"
    return ">128"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ms", type=float, default=60.0)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    with mp.Pool(min(4, len(a.paths))) as pool:
        loads = pool.map(kdeep.load, [(p, a.skip_ms * 1e3) for p in a.paths])
    per = collections.defaultdict(lambda: collections.defaultdict(list))  # (kind, bucket) -> cat -> [us per launch]
    count = collections.Counter()
    top = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))  # kind -> name -> [us, calls]
    nlaunch = collections.Counter()
    for d in loads:
        for l in d["launches"]:
            key = (l["kind"], bucket(l["bs"]))
            count[key] += 1
            sums = collections.Counter()
            for name, dur in l["k"]:
                sums[cat(name)] += dur
                t = top[l["kind"]][kdeep.short(name, 90)]
                t[0] += dur
                t[1] += 1
            nlaunch[l["kind"]] += 1
            for c, _ in CATS:
                per[key][c].append(sums.get(c, 0.0))
            per[key]["total"].append(sum(sums.values()))
    names = [c for c, _ in CATS] + ["total"]
    print("mean kernel ms per graph launch (all ranks pooled)")
    print(f"{'graph':13s} {'batch':>6s} {'n':>5s} " + " ".join(f"{c[:10]:>10s}" for c in names))
    for key in sorted(per, key=lambda k: (k[0], int(re.sub(r'\D', '', k[1]) or 999))):
        row = per[key]
        print(f"{key[0]:13s} {key[1]:>6s} {count[key]:5d} " + " ".join(f"{st.mean(row[c]) / 1e3:10.3f}" for c in names))
    for kind in sorted(top, key=lambda k: k != "verify"):
        n_l = nlaunch[kind]
        print(f"\ntop {kind} kernels (us per {kind} launch, over {n_l} launches)")
        tot = sum(v[0] for v in top[kind].values())
        for name, (us, n) in sorted(top[kind].items(), key=lambda kv: -kv[1][0])[: a.top if kind == "verify" else 12]:
            print(f"{us / n_l:9.1f} us {100 * us / tot:5.1f}%  {n / n_l:6.1f} calls  {us / n:7.2f} us/call  {name}")


if __name__ == "__main__":
    main()
