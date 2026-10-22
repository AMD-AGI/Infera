#!/usr/bin/env python3
"""conc_check.py BASE_URL N OUT_JSON [N_MATH]: N concurrent passkey requests (8K-34K tokens, distinct keys) followed by a
256-token ignore_eos tail so decode runs at batch size ~N. Prints pass count; temperature 0."""
import json, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

base, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "

def one(i):
    rng = random.Random(1000 + i)
    key = str(rng.randint(10000, 99999))
    reps = rng.randint(8000, 34000) // 21
    parts = [filler] * reps
    parts.insert(rng.randint(0, reps), f"The pass key is {key}. Remember it. {key} is the pass key. ")
    prompt = ("There is an important piece of information hidden inside a lot of irrelevant text. Find it and memorize it.\n\n"
              + "".join(parts) + "\n\nWhat is the pass key? Answer with the number first, then write a short poem about it.")
    body = {"model": "amd/GLM-5.2-MXFP4", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": 256, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        j = json.load(r)
    c = j["choices"][0]["message"]["content"] or ""
    return {"i": i, "key": key, "pass": key in c[:64], "head": c[:80], "usage": j.get("usage"), "secs": round(time.time() - t, 1)}

MATH = ("A farmer has 17 sheep. He buys 3 more sheep, then 5 sheep run away, and finally he buys 14 more. "
        "How many sheep does he have now? Answer with only the number.")

def math(i):
    body = {"model": "amd/GLM-5.2-MXFP4", "messages": [{"role": "user", "content": MATH}], "temperature": 0,
            "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        j = json.load(r)
    c = j["choices"][0]["message"]["content"] or ""
    return {"i": "math%d" % i, "key": "29", "pass": "29" in c, "head": c[:80], "usage": j.get("usage"),
            "secs": round(time.time() - t, 1)}

n_math = int(sys.argv[4]) if len(sys.argv) > 4 else 0
with ThreadPoolExecutor(n + n_math) as ex:
    futs = [ex.submit(math, i) for i in range(n_math)] + [ex.submit(one, i) for i in range(n)]
    res = [f.result() for f in futs]
n = n + n_math
json.dump(res, open(out, "w"), indent=1)
print("pass", sum(r["pass"] for r in res), "/", n)
for r in res:
    if not r["pass"]:
        print("FAIL", r)
