#!/usr/bin/env python3
"""Is the PD corruption FIRST-TOUCH (per-shape, self-healing) or PROGRESSIVE DEGRADATION
(server gets worse the more long requests it serves, and never recovers)?

Earlier evidence pointed both ways:
  - first-touch: novel shapes corrupt, immediate repeat is clean
  - degradation: later, one anchor corrupted on BOTH reps and then 6/6 with no recovery

Method: fix ONE canary shape. Interleave (canary, then N long novel requests) x rounds.
The canary is identical every round, so it is warm from round 1 onward. If the canary stays
clean, the bug is per-shape first-touch. If the canary starts failing as rounds accumulate,
the instance is degrading and the failure is a function of served long-request volume.

usage: degrade_test.py BASE MODEL ROUNDS [CANARY_LEN] [NOVEL_PER_ROUND] [OUT_JSON]
"""
import json, re, sys, time, urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
ROUNDS = int(sys.argv[3])
CANARY = int(sys.argv[4]) if len(sys.argv) > 4 else 30303
NOVEL = int(sys.argv[5]) if len(sys.argv) > 5 else 3
OUT = sys.argv[6] if len(sys.argv) > 6 else "/tmp/degrade.json"

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
    v = "GIBBERISH" if gibberish(txt) else ("OK" if EXPECT in txt else "MISS")
    return v, d["usage"]["prompt_tokens"], time.time() - t0, txt


res, served = [], 0
for r in range(ROUNDS):
    # canary: SAME length + SAME salt every round -> always the identical shape & content
    v, pt, dt, txt = ask(CANARY, 12345)
    print(f"[r{r}] CANARY  {v:9s} pt={pt} {dt:.1f}s served_long={served} "
          f"-> {txt.strip()[:70]!r}", flush=True)
    res.append({"round": r, "kind": "canary", "verdict": v, "served_before": served,
                "latency_s": round(dt, 2), "output": txt})
    if v == "GIBBERISH":
        print(f"  >>> canary corrupted at round {r} after {served} long requests", flush=True)

    for j in range(NOVEL):
        L = 31000 + r * 4099 + j * 1373          # never repeats across rounds
        try:
            v2, pt2, dt2, _ = ask(L, 900 + r * 17 + j)
        except Exception as e:
            print(f"[r{r}] novel~{L} ERROR {e}", flush=True); continue
        served += 1
        res.append({"round": r, "kind": "novel", "len": L, "verdict": v2,
                    "prompt_tokens": pt2})
    print(f"  round {r} novel verdicts: "
          f"{[x['verdict'] for x in res if x.get('round')==r and x['kind']=='novel']}",
          flush=True)

json.dump(res, open(OUT, "w"), indent=2)
can = [x["verdict"] for x in res if x["kind"] == "canary"]
print(f"\ncanary by round: {can}\n-> {OUT}")
