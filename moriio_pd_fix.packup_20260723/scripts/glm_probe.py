#!/usr/bin/env python3
"""PD correctness probe via infera router. Runs INSIDE the worker container.
Usage: pd_probe.py <router_url> <model>
"""
import json
import sys
import urllib.request

ROUTER = sys.argv[1] if len(sys.argv) > 1 else "http://10.2.122.10:8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "GLM-5.1-FP8"

PROMPTS = [
    "The capital of France is",
    "Question: What is 2+2? Answer:",
    "The capital of China is",
    "Explain in one sentence why the sky is blue:",
]


def ask(prompt, n=32):
    body = json.dumps(
        {"model": MODEL, "prompt": prompt, "max_tokens": n, "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        f"{ROUTER}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        return repr(d["choices"][0]["text"])
    except Exception as e:
        return f"ERR: {type(e).__name__}: {e}"


print(f"router={ROUTER} model={MODEL}")
for p in PROMPTS:
    print(f"  [{p!r}] -> {ask(p)}")
