#!/usr/bin/env python3
"""Decode-step bf16 GEMMs at every verify M of the TP8 decode graphs (graph batches 1-8, 10-32 step 2 -> 4 rows each),
timed with aiter tgemm.mm and whatever tables are active (production tables mounted; AITER_CONFIG_GEMM_BF16 adds more).
Run once per table set and compare the JSON files with --compare.

Usage (1 GPU): python3 gemm_table_bench.py --label prod --out A.json
               python3 gemm_table_bench.py --compare A.json B.json
"""
import argparse
import json
import sys

SHAPES = {
    "q_b_proj": (16384, 2048, 78), "fused_qkv_a": (2624, 6144, 78), "idx_wq_b": (4096, 2048, 21),
    "idx_wk": (128, 6144, 21), "idx_wp": (32, 6144, 21),
    "merged_qkva_wk_wp": (2784, 6144, 21), "merged_qb_wqb": (20480, 2048, 21),
}
MS = [4 * b for b in list(range(1, 9)) + list(range(10, 33, 2))]


def run(label, out):
    import torch

    sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
    from aklr_gemm_bench import graph_us
    from aiter.tuned_gemm import tgemm

    g = torch.Generator(device="cuda").manual_seed(0)
    res = {"label": label, "us": {}}
    for name, (n, k, _) in SHAPES.items():
        ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02 for _ in range(8)]
        for m in MS:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
            res["us"][f"{name}/{m}"] = round(graph_us(lambda: [tgemm.mm(x, w, None, torch.bfloat16) for w in ws], len(ws)), 2)
        print(name, {m: res["us"][f"{name}/{m}"] for m in MS}, flush=True)
    json.dump(res, open(out, "w"))


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print(f"us per call, {a['label']} -> {b['label']} (M = verify rows of one rank)")
    for name, (_, _, layers) in SHAPES.items():
        row = []
        for m in MS:
            ta, tb = a["us"][f"{name}/{m}"], b["us"][f"{name}/{m}"]
            row.append(f"{m}:{ta:.1f}->{tb:.1f}")
        sav = [a["us"][f"{name}/{m}"] - b["us"][f"{name}/{m}"] for m in MS if 48 <= m <= 104]
        print(f"{name:18s} ({layers} layers) mean saving at M 48-104: {sum(sav) / len(sav):5.2f} us/call, "
              f"{layers * sum(sav) / len(sav) / 1e3:5.3f} ms/step")
        print("    " + " ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a.label, a.out)


if __name__ == "__main__":
    main()
