#!/usr/bin/env python3
"""Isolate whether PD long-context corruption is caused by a PARTIAL prefix-cache hit.

Two arms, same lengths, same needle:
  arm "shared" : every prompt starts with the identical filler prefix, so request k+1
                 partially hits the radix cache left by request k, then chunk-prefills
                 the remainder and PD-transfers a MIXED (cached + fresh) KV range.
  arm "unique" : every prompt gets a distinct random-ish prefix salt, so there is no
                 cross-request prefix reuse; every request is a full cold prefill.

If "shared" corrupts and "unique" does not, the bug is in the cached+fresh KV path under
PD disaggregation, not in chunked prefill or context length per se.

usage: prefix_test.py BASE MODEL ARM "len1,len2,..." [OUT_JSON]
"""
import json, re, sys, time, urllib.request

BASE, MODEL, ARM = sys.argv[1], sys.argv[2], sys.argv[3]
LENS = [int(x) for x in sys.argv[4].split(",")]
OUT = sys.argv[5] if len(sys.argv) > 5 else f"/tmp/prefix_{ARM}.json"

FILLER = ("Record {i}: the maintenance crew inspected corridor {a} and logged "
          "routine status code {b} with no anomalies reported that shift.")
NEEDLE = ("Record SECRET-B: the calibration constant for the orbital gyroscope is "
          "exactly 82931.")
EXPECT = "82931"
TOK_PER_LINE = 28.1


def gibberish(t):
    return (t.count("</think>") > 2
            or len(re.findall(r"\b(the|The)\d", t)) > 2
            or len(re.findall(r"\d\s+the\b", t)) > 3)


def build(tokens, salt):
    n = max(4, int(tokens / TOK_PER_LINE))
    # salt shifts every line's content, so "unique" shares no tokenizable prefix at all
    lines = [FILLER.format(i=i + salt, a=(i * 7 + salt) % 400,
                           b=1000 + (i * 13 + salt * 31) % 8000) for i in range(n)]
    lines.insert(n // 2, NEEDLE)
    return ("Below is a long maintenance log. Read it carefully, then answer the question "
            "at the end using only information from the log.\n\n<log>\n" + "\n".join(lines)
            + "\n</log>\n\nQuestion: What is the calibration constant for the orbital "
              "gyroscope? Answer with the number only.")


res, bad = [], 0
for k, L in enumerate(LENS):
    salt = 0 if ARM == "shared" else (k + 1) * 7919
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": build(L, salt)}],
                       "max_tokens": 96, "temperature": 0}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
    except Exception as e:
        print(f"[{ARM}] req~{L} ERROR {e}", flush=True)
        res.append({"len": L, "verdict": "ERROR", "error": str(e)}); bad += 1; continue
    dt = time.time() - t0
    txt = d["choices"][0]["message"]["content"]
    pt = d["usage"]["prompt_tokens"]
    v = "GIBBERISH" if gibberish(txt) else ("OK" if EXPECT in txt else "MISS")
    bad += v == "GIBBERISH"
    print(f"[{ARM}] {v:9s} req~{L} pt={pt} {dt:.1f}s -> {txt.strip()[:110]!r}", flush=True)
    res.append({"len": L, "prompt_tokens": pt, "verdict": v,
                "latency_s": round(dt, 2), "output": txt})

json.dump(res, open(OUT, "w"), indent=2)
print(f"\n[{ARM}] gibberish {bad}/{len(LENS)} -> {OUT}")
