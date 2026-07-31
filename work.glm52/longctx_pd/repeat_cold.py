#!/usr/bin/env python3
"""Repeat the SAME long prompt N times, flushing the radix cache before each call, so every
iteration takes the cold path (fresh chunked prefill + fresh PD KV transfer).

The warm repeats in the earlier sweep were prefix-cache hits and always looked correct; this
isolates whether the corruption is a non-deterministic cold-path race.

usage: repeat_cold.py BASE MODEL TOKENS N [FLUSH_URL] [OUT_JSON]
"""
import json, re, sys, time, urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
TOKENS, N = int(sys.argv[3]), int(sys.argv[4])
FLUSH = sys.argv[5] if len(sys.argv) > 5 else ""
OUT = sys.argv[6] if len(sys.argv) > 6 else "/tmp/repeat_cold.json"

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


n = max(4, int(TOKENS / TOK_PER_LINE))
lines = [FILLER.format(i=i, a=(i * 7) % 400, b=1000 + (i * 13) % 8000) for i in range(n)]
lines.insert(n // 2, NEEDLE)
PROMPT = ("Below is a long maintenance log. Read it carefully, then answer the question at "
          "the end using only information from the log.\n\n<log>\n" + "\n".join(lines)
          + "\n</log>\n\nQuestion: What is the calibration constant for the orbital "
            "gyroscope? Answer with the number only.")

res, bad = [], 0
for i in range(N):
    if FLUSH:
        for u in FLUSH.split(","):
            try:
                urllib.request.urlopen(urllib.request.Request(u + "/flush_cache",
                                                              data=b""), timeout=60).read()
            except Exception as e:
                print(f"  (flush {u} failed: {e})", flush=True)
        time.sleep(2)
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 96, "temperature": 0}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
    except Exception as e:
        print(f"[{i}] ERROR {e}", flush=True)
        res.append({"i": i, "verdict": "ERROR", "error": str(e)}); bad += 1; continue
    dt = time.time() - t0
    txt = d["choices"][0]["message"]["content"]
    pt = d["usage"]["prompt_tokens"]
    cached = (d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", "?")
    v = "GIBBERISH" if gibberish(txt) else ("OK" if EXPECT in txt else "MISS")
    bad += v != "OK"
    print(f"[{i}] {v:9s} pt={pt} cached={cached} {dt:.1f}s -> {txt.strip()[:110]!r}", flush=True)
    res.append({"i": i, "verdict": v, "prompt_tokens": pt, "cached": cached,
                "latency_s": round(dt, 2), "output": txt})

json.dump(res, open(OUT, "w"), indent=2)
print(f"\n{N - bad}/{N} OK  ({bad} bad) -> {OUT}")
