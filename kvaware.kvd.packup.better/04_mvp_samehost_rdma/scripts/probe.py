#!/usr/bin/env python3
"""temp=0 factual probe through the infera router.

A 200 on /health tells you nothing about correctness, and in this experiment it
tells you nothing about the transport either — both legs report ready and
register with the router even when the KV transfer is completely broken. The
failure only appears when a request actually needs KV moved from prefill to
decode.

This version distinguishes three outcomes, because in experiment 04 the
interesting one is the middle case:

    OK        the answer contains the expected token
    WRONG     a completion came back, but the content is wrong
    ERROR     no completion at all — HTTP 500, timeout, connection refused

An HTTP 500 carrying "Failed to get kvcache from prefill instance" is the
same-host RDMA failure, and is reported as such.

Usage:
    python3 probe.py http://10.2.122.10:8100 qwen3
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8100"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3"

CASES = [
    ("The capital of France is", "paris"),
    ("The capital of China is", "beijing"),
    ("2+2=", "4"),
    ("The largest planet in our solar system is", "jupiter"),
]

ok = wrong = err = 0
kvcache_failures = 0

for prompt, want in CASES:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        txt = d["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        err += 1
        if "kvcache" in detail.lower():
            kvcache_failures += 1
            print(f"[ERROR] {prompt!r} -> HTTP {e.code} KV-TRANSFER FAILURE: {detail!r}")
        else:
            print(f"[ERROR] {prompt!r} -> HTTP {e.code}: {detail!r}")
        continue
    except Exception as e:
        err += 1
        print(f"[ERROR] {prompt!r} -> {type(e).__name__}: {e}")
        continue

    if want.lower() in txt.lower():
        ok += 1
        print(f"[OK   ] {prompt!r} -> {txt!r}")
    else:
        wrong += 1
        print(f"[WRONG] {prompt!r} -> {txt!r}")

n = len(CASES)
print(f"\n{ok}/{n} correct, {wrong} wrong-content, {err} errored")

if kvcache_failures:
    print(
        f"\n>>> {kvcache_failures}/{n} failed with a KV-transfer error.\n"
        ">>> On a SAME-HOST PD pair this is the mooncake cross-rail loopback\n"
        ">>> limitation, not a model or wiring problem. Check the prefill leg\n"
        ">>> log for 'transport retry counter exceeded' / 'received packet\n"
        ">>> mismatch'. Workaround: MC_FORCE_TCP=1."
    )

if wrong and not ok:
    print(
        "\n>>> Completions succeeded but every answer is wrong. That is a\n"
        ">>> DIFFERENT failure from a transport error — the KV moved, the\n"
        ">>> content is bad. Isolate it with a differential run (flip one\n"
        ">>> switch, hold everything else), not with a direct leg probe."
    )

# exit 0 only on a real pass; the two failure modes get distinct codes so a
# driver script can tell them apart.
if ok >= 3:
    sys.exit(0)
sys.exit(2 if kvcache_failures else 1)
