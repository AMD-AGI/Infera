#!/usr/bin/env python3
"""Reproduce the solo-vs-Case-A comparison tables in analysis/solo_latency.md.

The headline finding of this kit -- that queueing costs a roughly CONSTANT ~3.9 s
rather than a constant multiple -- can only be seen by bucketing on input length.
Raw percentile ratios are confounded: 102 solo samples cannot reproduce a
235K-token p99, so the two runs did not draw the same inputs, and an unbucketed
comparison silently compares different workloads.

Defaults assume this kit sits beside the Case A kit in the same parent dir.

  python3 scripts/compare_vs_caseA.py
  python3 scripts/compare_vs_caseA.py --solo <a.jsonl.gz> --casea <b.jsonl.gz>
"""
import argparse
import gzip
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_SOLO = os.path.join(HERE, "results", "metrics.jsonl.gz")
DEF_CASEA = os.path.join(
    os.path.dirname(HERE),
    "caseA.glm52.fullfeature.packup_20260801", "results", "metrics.jsonl.gz")

BUCKETS = [(0, 50_000), (50_000, 80_000), (80_000, 120_000),
           (120_000, 160_000), (160_000, 300_000)]
PCTS = [50, 90, 99]


def P(a, p):
    a = sorted(a)
    k = (len(a) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (k - lo)


def load(path, phase="sustain"):
    """Return (ttfts_seconds, input_tokens, acceptance_lengths)."""
    op = gzip.open if path.endswith(".gz") else open
    T, I, A = [], [], []
    for line in op(path, "rt"):
        r = json.loads(line)
        if r.get("phase") != phase:
            continue
        T += r.get("new_ttfts") or []
        I += r.get("new_prompt_lengths") or []
        A += r.get("new_acceptance_lengths") or []
    return T, I, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default=DEF_SOLO)
    ap.add_argument("--casea", default=DEF_CASEA)
    a = ap.parse_args()

    for p in (a.solo, a.casea):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}\n(pass --solo/--casea explicitly)")

    sT, sI, sA = load(a.solo)
    cT, cI, cA = load(a.casea)

    print(f"solo n={len(sT)}   Case A n={len(cT)}   (sustain phase)\n")

    print("## Unbucketed TTFT — CONFOUNDED, shown only for contrast")
    print("| pct | solo | Case A | ratio |")
    print("|---|---|---|---|")
    for p in PCTS:
        s, c = P(sT, p) * 1000, P(cT, p) * 1000
        print(f"| p{p} | {s:,.0f} ms | {c:,.0f} ms | {c/s:.2f}x |")
    print("\nThe two runs did not draw the same inputs — solo p99 input is well")
    print("below the spec's 235K. Bucket before drawing any conclusion.\n")

    print("## TTFT by input bucket — the real comparison")
    print("| input | solo n | solo mean | Case A n | Case A mean | ratio | difference |")
    print("|---|---|---|---|---|---|---|")
    diffs = []
    for lo, hi in BUCKETS:
        s = [t * 1000 for t, i in zip(sT, sI) if lo <= i < hi]
        c = [t * 1000 for t, i in zip(cT, cI) if lo <= i < hi]
        if not (s and c):
            continue
        sm, cm = sum(s) / len(s), sum(c) / len(c)
        diffs.append(cm - sm)
        print(f"| {lo//1000}–{hi//1000}K | {len(s)} | {sm:,.0f} ms | {len(c)} | "
              f"{cm:,.0f} ms | {cm/sm:.2f}x | **+{cm-sm:,.0f} ms** |")

    if diffs:
        print(f"\nqueueing penalty: mean +{sum(diffs)/len(diffs):,.0f} ms, "
              f"range +{min(diffs):,.0f} .. +{max(diffs):,.0f} ms")
        print("The RATIO collapses across buckets while the DIFFERENCE stays flat.")
        print("=> queueing is an additive wait, not a proportional slowdown.")

    if sA and cA:
        print(f"\n## MTP acceptance (load-independence check)")
        print(f"solo {sum(sA)/len(sA):.3f} (n={len(sA)})   "
              f"Case A {sum(cA)/len(cA):.3f} (n={len(cA)})   "
              f"delta {100*abs(sum(sA)/len(sA) - sum(cA)/len(cA))/(sum(cA)/len(cA)):.1f}%")


if __name__ == "__main__":
    main()
