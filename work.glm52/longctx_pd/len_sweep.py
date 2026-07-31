#!/usr/bin/env python3
"""Binary-search the prompt length at which PD output goes from coherent to garbage.

Reuses the needle haystack but only ONE needle at 50% depth, and scores by
(a) needle hit and (b) a crude gibberish detector (repeated '</think>' / digit-splatter),
because the failure mode observed is token-salad, not a wrong answer.

usage: len_sweep.py BASE MODEL "len1,len2,..." [OUT_JSON]
"""
import json, re, sys, time, urllib.request

BASE = sys.argv[1]
MODEL = sys.argv[2]
LENS = [int(x) for x in sys.argv[3].split(",")]
OUT = sys.argv[4] if len(sys.argv) > 4 else "/tmp/len_sweep.json"

FILLER = ("Record {i}: the maintenance crew inspected corridor {a} and logged "
          "routine status code {b} with no anomalies reported that shift.")
NEEDLE = ("Record SECRET-B: the calibration constant for the orbital gyroscope is "
          "exactly 82931.")
QUESTION = ("What is the calibration constant for the orbital gyroscope? "
            "Answer with the number only.")
EXPECT = "82931"
TOK_PER_LINE = 28.1          # measured for this filler on the GLM-5.2 tokenizer


def post(content, max_tokens=96, timeout=900):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t0


def build(target_tokens):
    n = max(4, int(target_tokens / TOK_PER_LINE))
    lines = [FILLER.format(i=i, a=(i * 7) % 400, b=1000 + (i * 13) % 8000) for i in range(n)]
    lines.insert(n // 2, NEEDLE)
    return ("Below is a long maintenance log. Read it carefully, then answer the question "
            "at the end using only information from the log.\n\n<log>\n" + "\n".join(lines)
            + "\n</log>\n\nQuestion: " + QUESTION)


def gibberish(t):
    """Heuristic: the observed corruption emits many </think> tags and bare digit runs."""
    return (t.count("</think>") > 2
            or len(re.findall(r"\b(the|The)\d", t)) > 2
            or len(re.findall(r"\d\s+the\b", t)) > 3)


res = []
for L in LENS:
    try:
        d, dt = post(build(L))
        txt = d["choices"][0]["message"]["content"]
        pt = d["usage"]["prompt_tokens"]
        hit = EXPECT in txt
        gib = gibberish(txt)
        verdict = "OK" if (hit and not gib) else ("GIBBERISH" if gib else "MISS")
        print(f"[{verdict:9s}] req~{L} prompt_tokens={pt} {dt:.1f}s -> {txt.strip()[:150]!r}",
              flush=True)
        res.append({"requested": L, "prompt_tokens": pt, "verdict": verdict,
                    "hit": hit, "gibberish": gib, "latency_s": round(dt, 2), "output": txt})
    except Exception as e:
        print(f"[ERROR    ] req~{L}: {e}", flush=True)
        res.append({"requested": L, "verdict": "ERROR", "error": str(e)})

json.dump(res, open(OUT, "w"), indent=2)
print(f"\n-> {OUT}")
