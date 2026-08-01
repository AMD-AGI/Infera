#!/usr/bin/env python3
"""Correctness for the agentic-bench deployment: short factual + needle-at-depth.

WHY BOTH, AND WHY CHAT:

  * agent-bench drives /v1/chat/completions, so that is the path graded here.
    The raw /v1/completions endpoint has no chat template and GLM-5.2 simply
    continues the text ("2 + 2 =" -> " 5\nI was reading a book about..."), which
    is base-LM behaviour, not a KV fault. Grading a chat/reasoning model on raw
    completion manufactures failures.
  * The 4 short prompts are ~5 tokens: ONE prefill chunk. They say nothing about
    the multi-chunk path. Case A prompts are 74K-235K tokens against an 8192
    per-rank chunk, i.e. 9-29 chunks each -- the regime where mooncake PD can
    RDMA-read KV pages the writing forward has not finished. The needle test is
    the one that actually covers the deployment being benchmarked.

Grading is on the expected string, never on a text classifier: GLM-5.2 is
bilingual and code-switches, and a CJK-based "corruption" rule has produced
false failures on this stack before.

Usage: correctness.py URL [needle_tokens] [tokenizer]
"""
import json
import random
import sys
import time
import urllib.request

URL = sys.argv[1].rstrip("/")
NEEDLE_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 120000
TOKENIZER = sys.argv[3] if len(sys.argv) > 3 else "/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4"
MODEL = "glm5.2-mxfp4"
TMO = 900


def chat(messages, max_tokens=256, temperature=0.0):
    body = json.dumps({
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature, "stream": False,
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TMO) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    msg = ch["message"]
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    return text, ch.get("finish_reason"), (d.get("usage") or {}), time.time() - t0


print("=" * 78)
print("PART 1 - short factual, chat template, temp=0")
print("=" * 78)
CASES = [
    ("What is the capital of France? Answer briefly.", ["Paris", "巴黎"]),
    ("What is the capital of China? Answer briefly.", ["Beijing", "北京"]),
    ("What is 2 + 2? Answer with just the number.", ["4", "four", "四"]),
    ("What is the largest planet in our solar system? Answer briefly.",
     ["Jupiter", "木星"]),
]
ok1 = 0
for i, (q, wants) in enumerate(CASES):
    try:
        text, fin, usage, dt = chat([{"role": "user", "content": q}])
        hit = any(w in text for w in wants)
        ok1 += hit
        print(f"[{i}] {'OK  ' if hit else 'FAIL'} {dt:6.2f}s finish={fin} "
              f"prompt_tok={usage.get('prompt_tokens')} :: {text[:100]!r}")
    except Exception as e:
        print(f"[{i}] FAIL {type(e).__name__}: {e}")
print(f"\nshort factual: {ok1}/{len(CASES)}\n")

print("=" * 78)
print(f"PART 2 - needle at depth in a ~{NEEDLE_TOK:,}-token prompt "
      f"(multi-chunk prefill)")
print("=" * 78)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

# Filler that is cheap to build and carries no digits, so the only 7-digit run
# in the whole prompt is the needle -- a partial retrieval cannot be confused
# with the model copying something else.
rnd = random.Random(20260731)
WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "flint", "grove",
         "harbor", "ivory", "jasper", "kelp", "lumen", "maple", "nimbus"]
chunk = " ".join(rnd.choice(WORDS) for _ in range(4000))
n_per = len(tok.encode(chunk, add_special_tokens=False))
reps = max(1, NEEDLE_TOK // n_per + 1)
filler = " ".join([chunk] * reps)
ftoks = tok.encode(filler, add_special_tokens=False)[:NEEDLE_TOK]
print(f"built filler: {len(ftoks):,} tokens")

ok2 = 0
depths = [0.05, 0.25, 0.50, 0.75, 0.95]
for d in depths:
    needle = str(rnd.randint(1000000, 9999999))
    cut = int(len(ftoks) * d)
    body = (tok.decode(ftoks[:cut])
            + f"\n\nThe secret access code is {needle}. Remember it.\n\n"
            + tok.decode(ftoks[cut:]))
    msgs = [{"role": "user", "content":
             body + "\n\nWhat is the secret access code mentioned above? "
                    "Reply with the 7-digit number only."}]
    try:
        text, fin, usage, dt = chat(msgs, max_tokens=256)
        hit = needle in text
        ok2 += hit
        print(f"depth={d:4.0%} {'OK  ' if hit else 'FAIL'} {dt:7.2f}s "
              f"prompt_tok={usage.get('prompt_tokens')} "
              f"cached={(usage.get('prompt_tokens_details') or {}).get('cached_tokens')} "
              f"want={needle} :: {text[-90:]!r}")
    except Exception as e:
        print(f"depth={d:4.0%} FAIL {type(e).__name__}: {str(e)[:120]}")

print(f"\nneedle: {ok2}/{len(depths)}")
print(f"\nTOTAL short={ok1}/{len(CASES)} needle={ok2}/{len(depths)}")
sys.exit(0 if (ok1 == len(CASES) and ok2 == len(depths)) else 1)
