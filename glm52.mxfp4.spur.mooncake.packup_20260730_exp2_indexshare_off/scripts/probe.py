#!/usr/bin/env python3
"""Sequential probe against a PD decode leg (or a mix server).

Usage: probe.py URL [n] [max_tokens] [timeout]

Prints per-request dp_rank + spec_accept_length so a pass can be distinguished
from "spec-dec silently bypassed", and so we can see which ranks served.
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30001"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 24
TMO = float(sys.argv[4]) if len(sys.argv) > 4 else 120

PROMPTS = [
    "The capital of France is",
    "Quantum computing is a type of computation that harnesses",
    "The three primary colors are",
    "In 1969, humans first landed on",
]

ok = 0
for i in range(N):
    body = json.dumps(
        {
            "text": PROMPTS[i % len(PROMPTS)],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": MAXTOK},
            "rid": f"probe-{i}",
        }
    ).encode()
    req = urllib.request.Request(
        URL + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TMO) as r:
            d = json.loads(r.read())
        mi = d.get("meta_info", {})
        dt = time.time() - t0
        ok += 1
        print(
            f"[{i}] OK {dt:5.2f}s dp={mi.get('dp_rank')} "
            f"acc_len={mi.get('spec_accept_length')} "
            f"tok={mi.get('completion_tokens')} "
            f"text={d.get('text','')[:60]!r}"
        )
    except Exception as e:
        print(f"[{i}] FAIL after {time.time()-t0:5.1f}s: {type(e).__name__} {e}")

print(f"\n{ok}/{N} ok")
sys.exit(0 if ok == N else 1)
