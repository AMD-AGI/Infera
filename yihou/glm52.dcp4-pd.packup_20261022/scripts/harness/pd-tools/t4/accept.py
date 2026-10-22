#!/usr/bin/env python3
"""T4c: MTP accept length with the real draft. 4 prompts x 400 tokens, greedy, sequential, through the router;
decode /metrics snapshot before/after. Usage: accept.py ROUTER_URL DECODE_URL OUT_JSON"""
import json, os, sys, time, urllib.request
router, decode, out = sys.argv[1:4]
MODEL = os.environ.get("T4_MODEL", "amd/GLM-5.2-MXFP4")
PROMPTS = ["Write a detailed explanation of how a hash map handles collisions, with examples.",
           "Describe the water cycle step by step for a high-school student.",
           "Write a Python function that parses ISO-8601 dates, then explain each line.",
           "Summarize the causes and consequences of the French Revolution."]
def metrics():
    txt = urllib.request.urlopen(decode + "/metrics", timeout=30).read().decode()
    return [l for l in txt.splitlines() if l.startswith("sglang:spec") or l.startswith("sglang:num_spec")
            or "accept" in l and not l.startswith("#")]
res = {"model": MODEL, "before": metrics(), "runs": []}
for p in PROMPTS:
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 400,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(router + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time(); j = json.load(urllib.request.urlopen(req, timeout=1800))
    res["runs"].append({"prompt": p[:40], "secs": round(time.time() - t, 2), "usage": j.get("usage"),
                        "meta": j.get("choices", [{}])[0].get("meta_info"),
                        "text": (j["choices"][0]["message"].get("content") or "")[:200]})
res["after"] = metrics()
json.dump(res, open(out, "w"), indent=1)
for r in res["runs"]: print(r["prompt"], r["secs"], r["usage"])
print("metrics after:"); print("\n".join(res["after"]))
