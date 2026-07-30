"""Escalating long-context smoke test through the router.

The production profile reaches p90 ~300K and caps at 380K, far past anything the
correctness suite covered (needle topped out at 58K). Before spending 10+ minutes
on a benchmark that would just collect errors, find out where the stack stops
working: send one request per target length and report what came back.
"""

import json
import sys
import time
import urllib.error
import urllib.request

ROUTER = "http://127.0.0.1:8000"
MODEL = "/wekafs/models/GLM-5.2-FP8"
TARGETS = [int(x) for x in (sys.argv[1:] or [60_000, 150_000, 300_000, 380_000])]

# ~11 tokens per line with this tokenizer; salt each prompt so nothing is served
# from the radix tree.
def build(target_tokens: int) -> str:
    lines = max(1, target_tokens // 11)
    salt = f"Session {time.time():.6f}"
    return salt + "\n" + "\n".join(
        f"Line {i}: budget {i * 41 % 89}, region {i % 11}, flag {i % 3}, id {i}."
        for i in range(lines)
    )


for target in TARGETS:
    body = build(target)
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": body + "\n\nReply with OK only."}],
            "max_tokens": 24,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{ROUTER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            got = json.loads(resp.read())
        wall = time.time() - t0
        usage = got.get("usage", {})
        text = (got["choices"][0]["message"].get("content") or "").strip()
        reason = got["choices"][0].get("finish_reason")
        print(
            f"target={target:>7}  prompt_tokens={usage.get('prompt_tokens'):>7}  "
            f"wall={wall:7.1f}s  finish={reason}  reply={text[:40]!r}",
            flush=True,
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        print(
            f"target={target:>7}  HTTP {exc.code} after {time.time() - t0:.1f}s: {detail}",
            flush=True,
        )
    except Exception as exc:
        print(f"target={target:>7}  FAILED after {time.time() - t0:.1f}s: {exc!r}", flush=True)
