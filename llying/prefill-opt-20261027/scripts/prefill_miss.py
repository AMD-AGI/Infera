#!/usr/bin/env python3
"""Where the prefill's uncached input tokens come from, per AgentX run (aiperf profile_export.jsonl).

For each profiled request, the reachable cache is the previous turn's input of the same conversation, when that
turn was sent earlier in this run (warm-up included): AgentX resends the whole history, so turn k's prompt starts
with turn k-1's prompt. Uncached tokens split into
  growth   -- ISL_k - ISL_{k-1}: new text, plus the previous answer, which the prefill never computed;
  lost     -- (ISL_{k-1}) - cached_k when positive: a prefix the run already sent but the cache did not return
              (eviction, or another prefill instance);
  first    -- the conversation's first request in this run: whatever was not cached (its prefix may be shared
              with other conversations of the same trace, e.g. a subagent's root context).
Usage: prefill_miss.py PROFILE_EXPORT_JSONL [LABEL]
"""

import json
import sys
from collections import defaultdict


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    recs = []
    for line in open(path):
        r = json.loads(line)
        md, m = r["metadata"], r["metrics"]
        if "input_sequence_length" not in m or md.get("was_cancelled"):
            continue
        recs.append((md["request_start_ns"], md["conversation_id"], md.get("turn_index", 0),
                     md.get("benchmark_phase"), int(m["input_sequence_length"]["value"]),
                     int(m.get("usage_prompt_cache_read_tokens", {}).get("value", 0))))
    recs.sort()
    prev_isl = {}
    tot = defaultdict(int)
    n = defaultdict(int)
    for _, conv, turn, phase, isl, cached in recs:
        p = prev_isl.get(conv)
        prev_isl[conv] = isl
        if phase != "profiling":
            continue
        tot["isl"] += isl
        tot["uncached"] += isl - cached
        n["req"] += 1
        if p is None:
            n["first"] += 1
            tot["first"] += isl - cached
            tot["first_isl"] += isl
            continue
        reach = min(p, isl)
        tot["growth"] += isl - reach
        lost = max(0, reach - cached)
        tot["lost"] += lost
        n["lost>0"] += lost > 0
        n["lost>4k"] += lost > 4096
        tot["lost_big"] += lost if lost > 4096 else 0
    isl = tot["isl"]
    pct = lambda k: 100.0 * tot[k] / isl
    print(f"{label}: {n['req']} profiled requests, hit {100 - pct('uncached'):.2f}%, uncached {pct('uncached'):.2f}% of input"
          f" = growth {pct('growth'):.2f} + lost {pct('lost'):.2f} + first {pct('first'):.2f}")
    print(f"  first-in-run requests {n['first']} ({100 * n['first'] / n['req']:.1f}%), their input {pct('first_isl'):.1f}% of all,"
          f" {100 * tot['first'] / max(1, tot['first_isl']):.1f}% of it uncached")
    print(f"  lost: {n['lost>0']} requests lose some prefix, {n['lost>4k']} lose > 4k tokens"
          f" ({pct('lost_big'):.2f}% of input in those)")
    print(f"  uncached tokens per request: growth {tot['growth'] / n['req']:.0f}, lost {tot['lost'] / n['req']:.0f},"
          f" first {tot['first'] / n['req']:.0f}")


if __name__ == "__main__":
    main()
