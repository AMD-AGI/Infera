#!/usr/bin/env python3
"""Sequential correctness probe against the INFERA ROUTER (OpenAI API).

The sanctioned kits' probe.py posts to sglang's native /generate. The infera
router does not expose that route -- it serves /v1/completions and
/v1/chat/completions -- so that probe returns 404 for every prompt, which reads
as a total failure of the deployment rather than a wrong URL. This is the same
probe against the API the router actually has.

Usage: probe_oai.py URL [max_tokens] [timeout]
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1].rstrip("/")
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 32
TMO = float(sys.argv[3]) if len(sys.argv) > 3 else 300
MODEL = "glm5.2-mxfp4"

# (prompt, one substring that MUST appear in a correct answer). Graded on the
# expected token, not on a classifier: GLM-5.2 is bilingual and code-switches,
# and a CJK-based "corruption" rule has produced false failures here before.
CASES = [
    ("The capital of France is", ["Paris"]),
    ("The capital of China is", ["Beijing", "北京"]),
    ("2 + 2 =", ["4", "four"]),
    ("The largest planet in our solar system is", ["Jupiter", "木星"]),
]

ok = 0
for i, (prompt, wants) in enumerate(CASES):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAXTOK,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{URL}/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TMO) as r:
            d = json.loads(r.read())
        dt = time.time() - t0
        text = d["choices"][0].get("text", "")
        usage = d.get("usage") or {}
        hit = any(w in text for w in wants)
        ok += hit
        print(f"[{i}] {'OK  ' if hit else 'FAIL'} {dt:6.2f}s "
              f"prompt_tok={usage.get('prompt_tokens')} "
              f"cached={(usage.get('prompt_tokens_details') or {}).get('cached_tokens')} "
              f"finish={d['choices'][0].get('finish_reason')} :: {text[:110]!r}")
    except Exception as e:
        print(f"[{i}] FAIL after {time.time()-t0:6.2f}s: {type(e).__name__} {e}")

print(f"\n{ok}/{len(CASES)} ok")
sys.exit(0 if ok == len(CASES) else 1)
