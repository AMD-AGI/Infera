#!/usr/bin/env python3
"""temp=0 factual correctness probe for the GLM-5.1-FP8 SGLang server.

THE actual test. A 200 / tokens-per-second result means NOTHING about correctness
(a broken KV path returns fluent garbage). This asserts the model gets the FACTS
right at temp=0. Run inside (or with network access to) the container after /health
is 200 and the model is registered.

Uses urllib (NOT nested `curl -d '{...}'` through docker exec, which mangles JSON).

    python3 glm_probe.py [http://127.0.0.1:30000] [GLM-5.1-FP8]
"""
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "GLM-5.1-FP8"

# (prompt, substring that MUST appear in the reply, case-insensitive)
PROBES = [
    ("The capital of France is", "paris"),
    ("The capital of China is", "beijing"),
    ("2+2=", "4"),
]


def ask(prompt: str) -> str:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 32,
        }
    ).encode()
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return (data["choices"][0]["message"].get("content") or "").strip()


ok_all = True
for prompt, need in PROBES:
    reply = ask(prompt)
    ok = need in reply.lower()
    ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {prompt!r} -> {reply!r}")
print("VERDICT:", "ALL PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
