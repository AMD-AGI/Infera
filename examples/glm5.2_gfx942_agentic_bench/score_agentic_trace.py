#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Recompute cache metrics for a multi-turn agentic-trace run.

``sglang.benchmark.serving`` mis-reports the input side in multi-turn mode: it
keeps the conversation-level ``prompt_len`` for every turn, so the summary can
print ``Total input tokens: 0`` next to a cache hit rate above 100%. The
per-request ``cached_tokens`` it records come from the server and are correct.

True per-turn input lengths come from the dataset instead -- weka_to_agentic_trace.py
wrote each turn's target into ``prompt_tokens``, verified exact against the
tokenizer. With both in hand this also reports the ideal hit rate (in a
growing-prefix session turn i can at best reuse all of turn i-1's prompt) and the
actual/ideal efficiency, which is the number worth comparing across tools.

A run containing any failed request is refused rather than scored: a failure is
recorded with ``cached_tokens=0``, so including it drags the hit rate towards zero
and a dead worker reads as a cache problem.

    python score_agentic_trace.py dataset.json run.jsonl <num_prompts>
"""

import json
import sys


def main(dataset_path: str, details_path: str, n_conv: int) -> int:
    ds = json.load(open(dataset_path))["conversations"][:n_conv]
    run = json.loads(open(details_path).read().strip())
    cached = run["cached_tokens"]
    errors = run.get("errors") or [""] * len(cached)

    expected_turns = sum(len(c) for c in ds)
    print(f"conversations={len(ds)}  turns in dataset={expected_turns}  "
          f"requests recorded={len(cached)}")

    # A failed turn is recorded with cached_tokens=0 and breaks its conversation's
    # prefix chain, so scoring one dilutes the hit rate towards zero and the result
    # reads like a cache problem instead of a dead worker. Refuse outright.
    failed = [i for i, e in enumerate(errors) if e]
    if failed:
        print(f"\n  {len(failed)} of {len(errors)} requests FAILED -- nothing to score.")
        print(f"  first at turn {failed[0]}: {errors[failed[0]][:150]}")
        print("  check both legs are registered (/v1/workers) and rerun.")
        return 1
    if expected_turns != len(cached):
        print(f"\n  turn count mismatch: dataset has {expected_turns}, run recorded "
              f"{len(cached)} -- alignment would be wrong, nothing to score.")
        return 1

    actual_in: list[int] = []
    ideal_cached: list[int] = []
    for conv in ds:
        prev = 0
        for turn in conv:
            actual_in.append(turn["prompt_tokens"])
            ideal_cached.append(prev)
            prev = turn["prompt_tokens"]

    n = len(cached)
    tot_in = sum(actual_in)
    tot_cached = sum(cached)
    tot_ideal = sum(ideal_cached)

    print(f"\n  total input tokens   {tot_in:>14,}")
    print(f"  total cached tokens  {tot_cached:>14,}")
    print(f"  ideal cached tokens  {tot_ideal:>14,}")
    print(f"\n  actual hit rate      {100*tot_cached/tot_in:>13.2f} %")
    print(f"  ideal  hit rate      {100*tot_ideal/tot_in:>13.2f} %")
    print(f"  efficiency (a/i)     {100*tot_cached/max(tot_ideal,1):>13.2f} %")

    # A turn that reused less than its predecessor's prompt lost blocks to eviction.
    short = [i for i in range(n) if cached[i] < ideal_cached[i]]
    lost = sum(ideal_cached[i] - cached[i] for i in short)
    print(f"\n  turns short of ideal {len(short):>14,} / {n}")
    print(f"  tokens lost to evict {lost:>14,} ({100*lost/max(tot_ideal,1):.2f}% of ideal)")

    isl = sorted(actual_in)
    print(f"\n  per-turn input: p50={isl[len(isl)//2]:,}  "
          f"p90={isl[int(len(isl)*.9)]:,}  p99={isl[int(len(isl)*.99)]:,}  "
          f"max={isl[-1]:,}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
