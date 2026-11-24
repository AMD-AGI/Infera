#!/usr/bin/env python3
"""What did each gate generate? Per gate result dir ($R/results/gate-p4d4-<tag>): completion tokens per file (probe /
longctx / gsm8k), the requests that hit the length cap, and the requests whose completion length differs from the
first gate given (same prompts; a different length means a different trajectory).
  gate_mix.py GATE_DIR [GATE_DIR ...]
The gate's real acceptance is one aggregate over all of it, so a single long, repetitive (easy to draft) completion
more or less moves it by about 1%.
"""
import json
import os
import sys

FILES = (("probe", "doc_id"), ("longctx", "idx"), ("gsm8k", "doc_id"))


def load(d):
    out = {}
    for name, key in FILES:
        p = os.path.join(d, f"{name}.jsonl")
        if not os.path.exists(p):
            continue
        rows = {}
        for line in open(p):
            r = json.loads(line)
            rows[(r.get("kind", ""), r.get(key))] = r
        out[name] = rows
    return out


def main():
    dirs = sys.argv[1:]
    runs = [(os.path.basename(d.rstrip("/")).replace("gate-p4d4-", ""), load(d)) for d in dirs]
    base_tag, base = runs[0]
    for tag, run in runs:
        tot = {n: sum(r.get("completion_tokens") or 0 for r in rows.values()) for n, rows in run.items()}
        capped = [(n, k[1], r.get("completion_tokens")) for n, rows in run.items() for k, r in rows.items()
                  if r.get("finish_reason") == "length"]
        print(f"{tag}: completion tokens " + ", ".join(f"{n} {t:,}" for n, t in tot.items())
              + f", total {sum(tot.values()):,}; capped {len(capped)}: "
              + " ".join(f"{n}#{i}({t})" for n, i, t in sorted(capped)))
        if run is base:
            continue
        diff = []
        for n, rows in run.items():
            for k, r in rows.items():
                b = base.get(n, {}).get(k)
                if b and b.get("completion_tokens") != r.get("completion_tokens"):
                    diff.append((abs((r["completion_tokens"] or 0) - (b["completion_tokens"] or 0)), n, k[1],
                                 b["completion_tokens"], r["completion_tokens"]))
        diff.sort(reverse=True)
        print(f"  vs {base_tag}: {len(diff)} requests with a different length; largest: "
              + " ".join(f"{n}#{i} {a}->{b}" for _, n, i, a, b in diff[:6]))


if __name__ == "__main__":
    main()
