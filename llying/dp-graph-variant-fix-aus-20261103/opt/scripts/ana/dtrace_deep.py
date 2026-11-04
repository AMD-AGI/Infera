#!/usr/bin/env python3
"""Deeper decode-trace breakdown across the DP ranks of one profile (one trace per rank, same steps).

Per rank: GPU time of the three graphs (draft / verify / draft_extend, kernels mapped to their hipGraphLaunch by
correlation), the verify graph by category with every kernel name assigned (top kernels per category listed), and
for the in-graph collectives a transfer-vs-wait split: the i-th collective kernel is the same call on every rank
(same graph, same order), so its shortest duration across ranks approximates the transfer, the rest is waiting for
the slowest rank. Optional MoE bytes per GPU give the achieved bandwidth of the MoE GEMMs.
Usage: dtrace_deep.py [--moe-gb GB] TRACE_RANK0.json.gz TRACE_RANK1.json.gz ...
"""
import argparse
import collections
import gzip
import json
import re
import statistics as st

CATS = [
    ("moe_gemm1", r"gemm1_a4w4|mfma_moe1|moe1_"),
    ("moe_gemm2", r"gemm2_a4w4|moe2_"),
    ("moe_reduce", r"moe_reduction"),
    ("moe_sort_quant", r"moe_sort|opus_moe|topk_reg|append_shared_experts|dynamic_per_group_scaled_quant|mxfp4_moe_sort"),
    ("comm_allgather", r"allgather"),
    ("comm_reduce_scatter", r"reduce_scatter"),
    ("comm_allreduce", r"cross_device_reduce|all_reduce|allreduce|nccl"),
    ("attn_mla", r"sparse_mla|bmm_rope|batched_gemm_a8w8|rope_cat|mla_"),
    ("indexer", r"mqa_logits|_topk_|topk_transform|hadamard|kn_entry|indexer"),
    ("dense_gemm", r"Cijk|hgemm|_gemm_a16|bf16gemm|gemm_a8w8|flydsl_hgemm"),
    ("norm", r"rmsnorm|layernorm|Layernorm|norm_kernel"),
    ("draft_moe", r"fmoe_bf16|fmoe_"),
    ("elementwise", r"elementwise|index_|gather_kernel|FillFunctor|copy|cat_|reduce_kernel|softmax|argmax"),
]
COMM = ("comm_allgather", "comm_reduce_scatter", "comm_allreduce")


def cat_of(name):
    for c, rx in CATS:
        if re.search(rx, name):
            return c
    return "other"


def load(path):
    d = json.load(gzip.open(path))
    ev = [e for e in d["traceEvents"] if e.get("ph") == "X"]
    gl = sorted((e for e in ev if e.get("name") == "hipGraphLaunch"), key=lambda e: e["ts"])
    idx = {e["args"]["correlation"]: i for i, e in enumerate(gl)}
    ks = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    graphs = {"draft": [], "verify": [], "draft_extend": []}
    for k in ks:
        i = idx.get(k["args"].get("correlation"))
        if i is not None:
            graphs[("draft", "verify", "draft_extend")[i % 3]].append(k)
    return len(gl) // 3, graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--moe-gb", type=float, default=0.0, help="routed-expert bytes per GPU read per verify step (GB)")
    a = ap.parse_args()
    ranks = [load(p) for p in a.traces]
    n = ranks[0][0]
    print(f"{len(ranks)} ranks, {n} steps per rank")
    for g in ("draft", "verify", "draft_extend"):
        per = [sum(k["dur"] for k in r[1][g]) / 1e3 / r[0] for r in ranks]
        print(f"  {g:13s} GPU ms/step per rank: " + " ".join(f"{x:5.2f}" for x in per))
    cat_ms = collections.defaultdict(list)
    top = collections.defaultdict(collections.Counter)
    for steps, graphs in ranks:
        acc = collections.Counter()
        for k in graphs["verify"]:
            c = cat_of(k["name"])
            acc[c] += k["dur"]
            top[c][k["name"][:70]] += k["dur"] / steps / len(ranks)
        for c in [x for x, _ in CATS] + ["other"]:
            cat_ms[c].append(acc[c] / 1e3 / steps)
    print("  verify by category (ms/step: mean over ranks [min..max])")
    for c in [x for x, _ in CATS] + ["other"]:
        v = cat_ms[c]
        if max(v) < 0.05:
            continue
        names = ", ".join(f"{nm} {us / 1e3:.2f}" for nm, us in top[c].most_common(2))
        print(f"    {c:20s} {st.mean(v):6.2f} [{min(v):5.2f}..{max(v):5.2f}]  {names}")
    # Collectives: transfer (min over ranks of the i-th call) vs wait.
    for c in COMM:
        seqs = [[k["dur"] for k in g["verify"] if cat_of(k["name"]) == c] for _, g in ranks]
        m = min(len(s) for s in seqs)
        if m == 0:
            continue
        floor = sum(min(s[i] for s in seqs) for i in range(m)) / 1e3 / n
        mean = st.mean(sum(s[:m]) for s in seqs) / 1e3 / n
        print(f"  {c:20s} {mean:5.2f} ms/step = transfer ~{floor:5.2f} + waiting for the slowest rank ~{mean - floor:5.2f}  ({m // n} calls/step)")
    if a.moe_gb:
        g12 = st.mean(cat_ms["moe_gemm1"]) + st.mean(cat_ms["moe_gemm2"])
        print(f"  MoE GEMM {g12:.2f} ms/step for {a.moe_gb:.1f} GB -> {a.moe_gb / g12:.2f} TB/s")


if __name__ == "__main__":
    main()
