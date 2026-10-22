#!/usr/bin/env python3
"""decodebench.py BASE_URL OUT_JSON [PROMPT_TOKENS] [OUT_TOKENS] [WAVES]: fixed-batch long-context decode load.

Primes the prefix cache with 32 distinct ~PROMPT_TOKENS prompts (max_tokens 1), then for each wave size N runs the
first N prompts concurrently with OUT_TOKENS ignore_eos tokens, so decode runs at batch N with cached prefill.
Records UTC wall-clock start/end of each wave for server-log binning."""
import datetime, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

base, out = sys.argv[1], sys.argv[2]
ptok = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
otok = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
waves = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "8,16,24,32").split(",")]
filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "

def prompt(i):
    return (f"Document {i}. " + filler * (ptok // 21) + f"\n\nWrite a very long essay about document {i}.")

def req(i, mt):
    body = {"model": "amd/GLM-5.2-MXFP4", "messages": [{"role": "user", "content": prompt(i)}], "temperature": 0,
            "max_tokens": mt, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(r, timeout=7200) as f:
        j = json.load(f)
    return {"i": i, "usage": j.get("usage"), "secs": round(time.time() - t, 2)}

now = lambda: datetime.datetime.utcnow().strftime("%H:%M:%S")
res = {"ptok": ptok, "otok": otok, "waves": []}
t0 = now()
with ThreadPoolExecutor(8) as ex:
    prime = list(ex.map(lambda i: req(i, 1), range(max(waves))))
res["prime"] = {"start": t0, "end": now(), "prompt_tokens": [p["usage"]["prompt_tokens"] for p in prime]}
print("primed", res["prime"]["start"], res["prime"]["end"], flush=True)
for n in waves:
    s, ts = now(), time.time()
    with ThreadPoolExecutor(n) as ex:
        r = list(ex.map(lambda i: req(i, otok), range(n)))
    w = {"n": n, "start": s, "end": now(), "secs": round(time.time() - ts, 1),
         "cached": [x["usage"].get("prompt_tokens_details", {}).get("cached_tokens") for x in r],
         "completion": [x["usage"]["completion_tokens"] for x in r]}
    res["waves"].append(w)
    print("wave", n, w["start"], w["end"], w["secs"], "min_cached", min(c or 0 for c in w["cached"]), flush=True)
json.dump(res, open(out, "w"), indent=1)
