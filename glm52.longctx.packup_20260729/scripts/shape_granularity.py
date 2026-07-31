#!/usr/bin/env python3
"""How coarse is the 'shape' whose first touch corrupts?

This decides whether a startup warmup sweep is even feasible:
  - if every distinct token count is its own shape -> unbounded, warmup impossible
  - if shapes bucket by chunk index (len // chunk) -> ~64 buckets for 131072 ctx -> feasible

Method: pick an anchor length, warm it (send twice; 2nd is warm/correct), then probe
neighbours at increasing deltas. A neighbour that is CLEAN means it shares the anchor's
bucket; a neighbour that CORRUPTS means it fell into a new bucket.

usage: shape_granularity.py BASE MODEL ANCHOR "d1,d2,d3,..." [OUT_JSON]
"""
import json, re, sys, time, urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
ANCHOR = int(sys.argv[3])
DELTAS = [int(x) for x in sys.argv[4].split(",")]
OUT = sys.argv[5] if len(sys.argv) > 5 else "/tmp/granularity.json"

FILLER = ("Record {i}: the maintenance crew inspected corridor {a} and logged "
          "routine status code {b} with no anomalies reported that shift.")
NEEDLE = ("Record SECRET-B: the calibration constant for the orbital gyroscope is "
          "exactly 82931.")
EXPECT = "82931"
TOK_PER_LINE = 28.1
_salt = [0]


def gibberish(t):
    return (t.count("</think>") > 2
            or len(re.findall(r"\b(the|The)\d", t)) > 2
            or len(re.findall(r"\d\s+the\b", t)) > 3)


def ask(tokens, salt):
    n = max(4, int(tokens / TOK_PER_LINE))
    lines = [FILLER.format(i=i + salt, a=(i * 7 + salt) % 400,
                           b=1000 + (i * 13 + salt * 31) % 8000) for i in range(n)]
    lines.insert(n // 2, NEEDLE)
    p = ("Below is a long maintenance log. Read it carefully, then answer the question at "
         "the end using only information from the log.\n\n<log>\n" + "\n".join(lines)
         + "\n</log>\n\nQuestion: What is the calibration constant for the orbital "
           "gyroscope? Answer with the number only.")
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": p}],
                       "max_tokens": 96, "temperature": 0}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    txt = d["choices"][0]["message"]["content"]
    return (("GIBBERISH" if gibberish(txt) else ("OK" if EXPECT in txt else "MISS")),
            d["usage"]["prompt_tokens"], time.time() - t0, txt)


res = []
# 1. warm the anchor: send twice, the 2nd must be clean (that is the known-good state)
for rep in range(2):
    _salt[0] += 1
    v, pt, dt, _ = ask(ANCHOR, _salt[0])
    print(f"[anchor rep{rep}] {v:9s} pt={pt} {dt:.1f}s", flush=True)
    res.append({"kind": "anchor", "rep": rep, "verdict": v, "prompt_tokens": pt})

# 2. probe neighbours, each with a fresh salt so content differs but length is controlled
for d_ in DELTAS:
    _salt[0] += 1
    v, pt, dt, txt = ask(ANCHOR + d_, _salt[0])
    print(f"[delta +{d_:6d}] {v:9s} pt={pt} {dt:.1f}s -> {txt.strip()[:80]!r}", flush=True)
    res.append({"kind": "delta", "delta": d_, "verdict": v, "prompt_tokens": pt,
                "latency_s": round(dt, 2), "output": txt})

json.dump(res, open(OUT, "w"), indent=2)
bad = sum(r["verdict"] == "GIBBERISH" for r in res if r["kind"] == "delta")
print(f"\nneighbours corrupted: {bad}/{len(DELTAS)} -> {OUT}")
