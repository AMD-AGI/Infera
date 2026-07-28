#!/usr/bin/env python3
# kv-aware + kvd functional test: send a long shared-prefix request twice; the 2nd
# should hit the prefix cache (radix/hicache), visible as cached prompt tokens and lower TTFT.
import json, urllib.request, time, sys
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8100"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "glm5.2-mxfp4"
prefix = "You are a helpful assistant. " + ("The quick brown fox jumps over the lazy dog. " * 200)

def call(tag):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prefix + " Now answer: what is 2+2?"}],
                       "max_tokens": 16, "temperature": 0}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.load(resp)
    dt = time.time() - t
    u = d.get("usage", {})
    cached = u.get("prompt_tokens_details", {})
    print(f"[{tag}] {dt:.2f}s prompt_tok={u.get('prompt_tokens')} cached={cached} "
          f"txt={d['choices'][0]['message']['content'][:40]!r}")

call("req1-cold")
time.sleep(2)
call("req2-warm")
