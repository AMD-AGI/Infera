#!/usr/bin/env python3
"""Measure output degeneracy (repetition looping) on captured Case A generations.

Reads the JSONL written by the CASEA_GEN capture patch in driver/agent/
agent_throughput.py (enabled by setting CASEA_GENERATIONS=<path>).

Metric matches the one used on the fixlen arms so the numbers are comparable:
a request is "looping" if some 10-gram repeats >= 5 times. Also reports the
unique-word ratio, and breaks results down by output length -- the split
observed on fixlen was clean at osl 320 (0% looping) versus osl 3300 (40-54%).

Usage:
  python3 loopcheck.py generations.jsonl [--n 10] [--times 5]
"""
import argparse
import json
import sys
from collections import Counter


def loop_stats(text: str, n: int = 10):
    """Max n-gram repeat count and unique-word ratio for one generation."""
    words = text.split()
    if len(words) < n:
        return 0, (len(set(words)) / len(words) if words else 1.0), len(words)
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(grams.values()), len(set(words)) / len(words), len(words)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=10, help="n-gram size (default 10)")
    ap.add_argument("--times", type=int, default=5,
                    help="repeats at which a request counts as looping (default 5)")
    ap.add_argument("--show-worst", type=int, default=3)
    a = ap.parse_args()

    rows = []
    with open(a.path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            mx, uniq, nwords = loop_stats(r.get("text", ""), a.n)
            rows.append({
                "request_id": r.get("request_id"),
                "completion_tokens": r.get("completion_tokens", 0),
                "max_repeat": mx,
                "unique_ratio": uniq,
                "words": nwords,
            })

    if not rows:
        print("no records — was CASEA_GENERATIONS set for the run?")
        return 1

    def report(label, sel):
        if not sel:
            print(f"  {label:<24} (none)")
            return
        loop = [r for r in sel if r["max_repeat"] >= a.times]
        toks = sum(r["completion_tokens"] for r in sel) or 1
        looptoks = sum(r["completion_tokens"] for r in loop)
        worst = max(r["max_repeat"] for r in sel)
        print(f"  {label:<24} n={len(sel):>5}  looping={len(loop):>5} "
              f"({100*len(loop)/len(sel):5.1f}%)  "
              f"tokens-in-looping={100*looptoks/toks:5.1f}%  worst=x{worst}")

    print(f"\n{len(rows)} generations, looping = some {a.n}-gram repeats >= {a.times}x\n")
    report("ALL", rows)
    print()
    # The osl split is the thing to look at: degeneracy is length-driven.
    bands = [(0, 320), (320, 1000), (1000, 3300), (3300, 8000), (8000, 10**9)]
    for lo, hi in bands:
        label = f"osl {lo}-{hi}" if hi < 10**9 else f"osl >{lo}"
        report(label, [r for r in rows if lo < r["completion_tokens"] <= hi])

    print("\nworst offenders:")
    for r in sorted(rows, key=lambda r: -r["max_repeat"])[:a.show_worst]:
        print(f"  req {r['request_id']}: osl={r['completion_tokens']} "
              f"max_repeat=x{r['max_repeat']} unique_word_ratio={r['unique_ratio']:.3f}")

    total = sum(r["completion_tokens"] for r in rows) or 1
    looped = sum(r["completion_tokens"] for r in rows if r["max_repeat"] >= a.times)
    print(f"\nTOKEN-WEIGHTED: {100*looped/total:.1f}% of generated tokens came from "
          f"looping requests.\nThroughput is an UPPER BOUND to the extent this is "
          f"non-zero: a repetition loop is trivially predictable, so EAGLE accepts\n"
          f"more drafts, so fewer target forwards per token. Direction is knowable, "
          f"magnitude is not — do not estimate it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
