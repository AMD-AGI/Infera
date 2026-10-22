#!/usr/bin/env python3
"""Correctness probes against an OpenAI-compatible sglang endpoint: math prompt + passkey at ~8K/34K/96K tokens.
Usage: check.py BASE_URL OUT_JSON"""
import json, sys, time, urllib.request, random

base, out = sys.argv[1], sys.argv[2]
MODEL = __import__("os").environ.get("CHECK_MODEL", "amd/GLM-5.2-MXFP4")

def chat(content, max_tokens, thinking=False):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    m = j["choices"][0]["message"]
    return {"content": m.get("content"), "reasoning": (m.get("reasoning_content") or "")[:400],
            "usage": j.get("usage"), "secs": round(time.time() - t, 2)}

res = {}
math_q = ("A farmer has 17 sheep. He buys 3 more sheep, then 5 sheep run away, and finally he buys 14 more. "
          "How many sheep does he have now? Answer with only the number.")
r = chat(math_q, 64); r["expect"] = "29"; r["pass"] = (r["content"] or "").strip().rstrip(".") == "29"
res["math_nothink"] = r
r = chat(math_q, 2048, thinking=True); r["expect"] = "29"; r["pass"] = "29" in (r["content"] or "")
res["math_think"] = r
filler = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")
rng = random.Random(1234)
for name, approx_tokens in (("8k", 8000), ("34k", 34000), ("96k", 96000)):
    key = str(rng.randint(10000, 99999))
    n = approx_tokens // 21
    parts = [filler] * n
    parts.insert(n // 2, f"The pass key is {key}. Remember it. {key} is the pass key. ")
    prompt = ("There is an important piece of information hidden inside a lot of irrelevant text. "
              "Find it and memorize it.\n\n" + "".join(parts) + "\n\nWhat is the pass key? Answer with only the number.")
    r = chat(prompt, 32); r["expect"] = key; r["pass"] = key in (r["content"] or "")
    res["passkey_" + name] = r
for k, v in res.items():
    print(k, "PASS" if v["pass"] else "FAIL", repr(v["content"])[:120], v.get("usage", {}).get("prompt_tokens"), v["secs"])
json.dump(res, open(out, "w"), indent=1)
