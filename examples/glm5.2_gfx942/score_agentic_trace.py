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
tokenizer. With both in hand this also reports the ideal hit rate and the
actual/ideal efficiency, which is the number worth comparing across tools.

The ideal is what a cache that evicted nothing would deliver on THIS engine, so
that a gap means eviction and nothing else:

* A match is whole pages and the last page is always recomputed, because the
  forward has to produce logits. No turn can reuse more than that.
* Cold (growing prefix), turn i can reuse turn i-1's prompt -- the response
  tokens between them were generated on the decode leg, so they are not in the
  prefill cache -- less one more page. That last page is measured, not derived:
  across a 138-turn run every non-first turn landed exactly 64 tokens (one page)
  under turn i-1's prompt length, with 4 exceptions that fell further. Two things
  could produce a fixed one-page offset -- the prefill leg not committing the
  page its running request still holds, or the chat template adding a few tokens
  where the turns join, which costs the whole page it straddles -- and this does
  not try to tell them apart. Without the offset, 114 of 138 turns look "short of
  ideal" when the cache in fact returned everything available.

A run containing any failed request is refused rather than scored: a failure is
recorded with ``cached_tokens=0``, so including it drags the hit rate towards zero
and a dead worker reads as a cache problem.

``--warm`` scores a replay of a trace the cache has already seen once (see
run_kvd_reuse.sh), where the growing-prefix ideal no longer applies: every turn's
whole prompt was materialized by the first pass, so the ceiling is the prompt
minus its last page, which the engine must recompute to produce logits.

    python score_agentic_trace.py dataset.json run.jsonl <num_prompts> [--warm]
"""

import argparse
import json


def main(dataset_path: str, details_path: str, n_conv: int, warm: bool, page: int) -> int:
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
            length = turn["prompt_tokens"]
            actual_in.append(length)
            # A prefix match is whole pages, and the last page always gets
            # recomputed because the forward has to produce logits. So no turn can
            # reuse more than this, whatever the cache holds:
            ceiling = ((length - 1) // page) * page
            # Cold: turn i's cache holds turn i-1's prompt and no more -- the
            # response tokens in between were generated on the decode leg -- minus
            # one page, measured (see the module docstring). Turn 0 reuses nothing.
            # Warm: pass 1 stored every turn's whole prompt.
            ideal_cached.append(
                ceiling if warm else max(0, min(prev - page, ceiling))
            )
            prev = length

    n = len(cached)
    tot_in = sum(actual_in)
    tot_cached = sum(cached)
    tot_ideal = sum(ideal_cached)

    print(f"\n  ideal model          {'warm replay' if warm else 'growing prefix':>14}")
    print(f"  total input tokens   {tot_in:>14,}")
    print(f"  total cached tokens  {tot_cached:>14,}")
    print(f"  ideal cached tokens  {tot_ideal:>14,}")
    print(f"\n  actual hit rate      {100*tot_cached/tot_in:>13.2f} %")
    print(f"  ideal  hit rate      {100*tot_ideal/tot_in:>13.2f} %")
    print(f"  efficiency (a/i)     {100*tot_cached/max(tot_ideal,1):>13.2f} %")

    # Cold: the ideal already excludes the pages no cache could return, so a turn
    # under it lost blocks that were stored and then dropped. Warm: the ideal is
    # what a perfect L3 would serve, and a gap there is as often a prefetch that
    # was abandoned before it asked as a page L3 never held -- the two are
    # indistinguishable from cached_tokens alone. Hence the neutral label below.
    short = [i for i in range(n) if cached[i] < ideal_cached[i]]
    lost = sum(ideal_cached[i] - cached[i] for i in short)
    label = "tokens not served " if warm else "tokens lost to evict"
    print(f"\n  turns short of ideal {len(short):>14,} / {n}")
    print(f"  {label} {lost:>14,} ({100*lost/max(tot_ideal,1):.2f}% of ideal)")
    # Efficiency can pass 100% because the ideal models one conversation growing on
    # its own, and this corpus also shares prefixes BETWEEN conversations. Say so
    # rather than letting the ratio imply the cache beat its own ceiling.
    over = [i for i in range(n) if cached[i] > ideal_cached[i]]
    if over:
        gained = sum(cached[i] - ideal_cached[i] for i in over)
        print(f"  turns ABOVE ideal    {len(over):>14,} (+{gained:,} tokens) — reuse "
              "the ideal does not model,\n                       almost always a "
              "prefix shared with a different conversation")

    report_tiers(run.get("cached_tokens_details"), ds, ideal_cached)

    isl = sorted(actual_in)
    print(f"\n  per-turn input: p50={isl[len(isl)//2]:,}  "
          f"p90={isl[int(len(isl)*.9)]:,}  p99={isl[int(len(isl)*.99)]:,}  "
          f"max={isl[-1]:,}")
    return 0


def report_tiers(details: list | None, ds: list, ideal_cached: list[int]) -> None:
    """Split the hits across L1/L2/L3, which is the only direct evidence that the
    storage backend did anything.

    A conversation's first turn is called out because it is the one place where a
    hit cannot have come from anywhere else: the tiers above were flushed before
    the run, and no earlier turn of that conversation has populated them yet. Every
    later turn is served by the GPU out of its own conversation's traffic, so the
    aggregate storage share understates the backend even when it works."""
    if not details:
        return
    tiers = ("device", "host", "storage")
    tot = dict.fromkeys(tiers, 0)
    first_store = first_ideal = first_dev = 0
    i = 0
    for conv in ds:
        for t, _turn in enumerate(conv):
            d = details[i] if i < len(details) else None
            for k in tiers:
                tot[k] += (d or {}).get(k, 0) or 0
            if t == 0:
                first_store += (d or {}).get("storage", 0) or 0
                first_dev += (d or {}).get("device", 0) or 0
                first_ideal += ideal_cached[i]
            i += 1
    grand = sum(tot.values()) or 1
    print("\n  cached tokens by tier")
    for k in tiers:
        print(f"    {k:<8} {tot[k]:>14,}  {100*tot[k]/grand:>6.2f} %")
    if not (first_ideal or first_store or first_dev):
        return
    print(f"\n  conversation-opening turns ({len(ds)} of them)")
    if first_ideal:
        print(f"    from storage {first_store:>12,} / {first_ideal:,} ideal "
              f"({100*first_store/first_ideal:.1f} %) — nothing above storage "
              "holds these")
    elif first_store:
        print(f"    from storage {first_store:>12,} — served below the GPU")
    if first_dev:
        print(f"    from device  {first_dev:>12,} — a first turn cannot reuse its "
              "own conversation, so\n                 this is a prefix shared with "
              "another conversation (or a tier\n                 that was not cold)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dataset")
    p.add_argument("details")
    p.add_argument("num_prompts", type=int)
    p.add_argument(
        "--warm",
        action="store_true",
        help="score a replay of a trace the cache has already served once",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=64,
        help="KV page size. Both ideals are page-aligned, so this must match the "
        "server: SGLang forces 64 for GLM-5.2's DSA attention, and the router's "
        "/v1/workers reports it as kv_block_size.",
    )
    a = p.parse_args()
    raise SystemExit(main(a.dataset, a.details, a.num_prompts, a.warm, a.page_size))
