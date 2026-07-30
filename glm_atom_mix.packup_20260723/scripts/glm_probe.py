#!/usr/bin/env python3
"""temp=0 factual correctness probe for the GLM-5.1-FP8 ATOM server.

THE actual test. A 200 / ready server means NOTHING about correctness (a broken
decode returns empty / first-token-only, or fluent garbage). This asserts the
model gets the FACTS right at temp=0.

CRITICAL for GLM: it is a THINKING model. With thinking ON (default) and a small
max_tokens, the whole budget is spent inside the reasoning preamble and content
looks empty/truncated (NOT wrong). So we send chat_template_kwargs=
{"enable_thinking": false} and a generous max_tokens to get the direct answer.

Uses urllib (NOT nested `curl -d '{...}'` through docker exec, which mangles JSON).

    python3 glm_probe.py [http://127.0.0.1:8000] [GLM-5.1-FP8]
"""
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "GLM-5.1-FP8"

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
            "max_tokens": 200,
            # GLM thinking model: disable thinking so the direct answer isn't
            # crowded out by the reasoning preamble.
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
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
