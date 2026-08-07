#!/usr/bin/env python3
"""Per-request latency for a concurrency-1 agentic run, SUSTAIN phase only.

The driver's summary.json reports TTFT/TPOT percentiles but no end-to-end latency.
The staged driver carries the SOLO_M1 patch, which appends `new_e2es` and
`new_tpots` to each metrics tick with the SAME slice bounds as `new_ttfts` — so the
three concatenate row-wise into per-request records. `new_tpots` uses 0.0 to mark a
sample the driver filtered out (gen_len<=1 or gen_time<50ms); those are DROPPED here
rather than treated as zero-latency tokens.

Ramp is a warm-up EXCLUSION window (it is what makes the shared prefix resident), so
only `phase == "sustain"` rows are counted.

Usage:  python3 analyze_solo.py <results/agentic_<tag>/<tag>/<timestamp>/> [label]
"""
import json
import sys
from pathlib import Path


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


def main() -> int:
    d = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else d.parent.name
    mfile = d / "metrics.jsonl"
    rows = [json.loads(l) for l in mfile.read_text().splitlines() if l.strip()]

    rec = []
    for r in rows:
        if r.get("phase") != "sustain":
            continue
        ttfts = r.get("new_ttfts") or []
        e2es = r.get("new_e2es") or []
        tpots = r.get("new_tpots") or []
        gens = r.get("new_generation_lengths") or []
        prompts = r.get("new_prompt_lengths") or []
        hits = r.get("new_cache_hit_rates") or []
        n = len(ttfts)
        for i in range(n):
            rec.append({
                "ttft": ttfts[i],
                "e2e": e2es[i] if i < len(e2es) else None,
                "tpot": tpots[i] if i < len(tpots) else None,
                "gen": gens[i] if i < len(gens) else None,
                "prompt": prompts[i] if i < len(prompts) else None,
                "hit": hits[i] if i < len(hits) else None,
            })

    if not rec:
        print(f"{label}: no sustain-phase requests", file=sys.stderr)
        return 1

    ttft = [x["ttft"] * 1000 for x in rec if x["ttft"]]
    e2e = [x["e2e"] * 1000 for x in rec if x["e2e"]]
    # 0.0 means "filtered by the driver", not "instant" — drop, do not average in.
    tpot = [x["tpot"] * 1000 for x in rec if x["tpot"]]
    gen = [x["gen"] for x in rec if x["gen"]]
    prompt = [x["prompt"] for x in rec if x["prompt"]]
    hit = [x["hit"] for x in rec if x["hit"] is not None]

    print(f"===== {label} — sustain phase, n={len(rec)} requests =====")
    print(f"  prompt tokens : p50 {pct(prompt,.5):,.0f}   (n={len(prompt)})")
    print(f"  gen tokens    : p50 {pct(gen,.5):,.0f}   mean {sum(gen)/len(gen):,.1f}")
    print(f"  cache hit     : mean {sum(hit)/len(hit):.4f}" if hit else "  cache hit     : n/a")
    for name, xs in (("TTFT", ttft), ("E2E", e2e), ("TPOT", tpot)):
        if not xs:
            print(f"  {name:<5} ms      : (no samples)")
            continue
        print(f"  {name:<5} ms      : p50 {pct(xs,.5):9.1f}  p90 {pct(xs,.9):9.1f}  "
              f"p99 {pct(xs,.99):9.1f}  mean {sum(xs)/len(xs):9.1f}  n={len(xs)}")
    if len(tpot) < len(rec):
        print(f"  NOTE: {len(rec)-len(tpot)} of {len(rec)} TPOT samples were filtered by the "
              f"driver (gen_len<=1 or gen_time<50ms) and are excluded, not zeroed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
