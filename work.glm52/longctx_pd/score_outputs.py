#!/usr/bin/env python3
"""Re-score canary outputs with a finer verdict than the original binary gibberish flag.

The original detector fires on ANY repeated '</think>', which conflates two very different
failure modes seen before vs after the mooncake wait_event patch:

  CORRUPT_REASONING - the reasoning text itself is token salad ("The8292</think>The829.5")
                      => the KV the model read was wrong
  TAIL_REPEAT       - reasoning is fully coherent and the needle is correctly retrieved,
                      only the trailing post-</think> segment loops
                      => KV was right; this is a stop/EOS artifact

usage: score_outputs.py FILE.json [LABEL]
"""
import json, re, sys

EXPECT = "82931"
path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path


def split_reasoning(t):
    """Text before the first </think> is the reasoning; the rest is the answer tail."""
    i = t.find("</think>")
    return (t, "") if i < 0 else (t[:i], t[i:])


def salad(t):
    """Token-salad markers, evaluated on REASONING ONLY."""
    return (len(re.findall(r"\b[Tt]he\d", t)) > 1
            or len(re.findall(r"\d\s+the\b", t)) > 2
            or len(re.findall(r"\b\d+\.\s*\d+\s*the\b", t)) > 1
            or len(re.findall(r"[一-鿿]", t)) > 5      # spurious CJK
            or len(re.findall(r"(\b\w+\b)(?:\W+\1\b){4,}", t)) > 0)


rows = json.load(open(path))
cats = {}
for x in rows:
    if x.get("kind") not in (None, "canary"):
        continue
    out = x.get("output", "")
    reasoning, tail = split_reasoning(out)
    found = EXPECT in out
    if salad(reasoning):
        v = "CORRUPT_REASONING"
    elif found and tail.count("</think>") > 1:
        v = "TAIL_REPEAT"
    elif found:
        v = "CLEAN"
    else:
        v = "NO_NEEDLE"
    cats[v] = cats.get(v, 0) + 1
    print(f"  r{x.get('round','?')} {v:18s} lat={x.get('latency_s','?')}s")

n = sum(cats.values())
print(f"\n[{label}] n={n} " + " ".join(f"{k}={v}" for k, v in sorted(cats.items())))
good = cats.get("CLEAN", 0) + cats.get("TAIL_REPEAT", 0)
print(f"[{label}] needle retrieved with coherent reasoning: {good}/{n}")
