#!/usr/bin/env python3
"""Track B smoke: fire N concurrent long-prompt completions at the router to push decode in-flight KV over a cap.

Usage: longprompt_burst.yihou.py URL N PROMPT_TOKENS MAX_TOKENS [MODEL]
Prompts are distinct (per-request random words) so prefix caching cannot shrink them. Prints per-request status,
latency and output token count, then a summary; exit 1 if any request failed.
"""
import json, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

url, n, ptok, mtok = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
model = sys.argv[5] if len(sys.argv) > 5 else "glm5.2-mxfp4"
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()


def one(i):
    rnd = random.Random(1000 + i)
    text = f"[req {i}] " + " ".join(rnd.choice(WORDS) for _ in range(ptok)) + "\nSummarize the list above."
    body = json.dumps({"model": model, "prompt": text, "max_tokens": mtok, "temperature": 0}).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(url.rstrip("/") + "/v1/completions", body, {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        u = r.get("usage", {})
        return i, "ok", time.time() - t0, u.get("prompt_tokens"), u.get("completion_tokens")
    except Exception as e:
        return i, f"ERR {type(e).__name__}: {e}"[:200], time.time() - t0, None, None


with ThreadPoolExecutor(n) as ex:
    res = list(ex.map(one, range(n)))
bad = 0
for i, st, dt, pt, ct in res:
    print(f"req {i:3d} {st} {dt:8.1f}s prompt={pt} completion={ct}", flush=True)
    bad += st != "ok"
print(f"SUMMARY n={n} ok={n-bad} failed={bad} max_latency={max(r[2] for r in res):.1f}s")
sys.exit(1 if bad else 0)
