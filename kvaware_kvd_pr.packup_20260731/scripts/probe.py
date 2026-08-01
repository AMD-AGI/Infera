#!/usr/bin/env python3
# temp=0 factual probe — proves coherent output (a 200/health tells nothing about correctness).
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm5.2-mxfp4"
CASES = [
    ("The capital of France is", "paris"),
    ("The capital of China is", "beijing"),
    ("2+2=", "4"),
    ("The largest planet in our solar system is", "jupiter"),
]
ok = 0
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
        f"{BASE}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        txt = d["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"FAIL {prompt!r}: {e}")
        continue
    hit = want.lower() in txt.lower()
    ok += hit
    print(f"[{'OK' if hit else 'XX'}] {prompt!r} -> {txt!r}")
print(f"\n{ok}/{len(CASES)} correct")
sys.exit(0 if ok >= 3 else 1)
