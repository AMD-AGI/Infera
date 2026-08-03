#!/usr/bin/env python3
"""Correctness for the agentic-bench deployment: short factual + needle-at-depth.

Derived from agenticbench.glm52.spur.packup_20260731/scripts/correctness.py with
TWO deliberate changes, both forced by MTP being ON in this deployment:

  1. SAMPLING IS THE MODEL'S OWN, NOT GREEDY. GLM-5.2's generation_config.json
     says temperature 1.0 / top_p 0.95. The predecessor probe sent
     `temperature: 0`, which under EAGLE/MTP is INDISTINGUISHABLE FROM KV
     CORRUPTION: greedy decoding sends a reasoning model into repetition on a
     long prompt, and the draft model predicts the loop perfectly, so `accept
     len` pins at its maximum and the response runs to `max_tokens`. That
     manufactured a 3/5 needle result on a healthy engine once and cost a full
     debugging cycle. `accept len: 4.00` is a SYMPTOM OF THE LOOP, not evidence
     MTP is healthy.
  2. NEEDLE max_tokens 256 -> 2048. At 256 the reasoning is cut off mid-thought
     and the run-on tail mimics corruption exactly.

Sampling is non-greedy, so grading must not depend on an exact continuation --
it does not: the needle is a 7-digit number and the filler is DIGIT-FREE, so the
only 7-digit run in the whole prompt is the needle itself. A partial retrieval
cannot be confused with the model copying something else. `seed` is sent so a
failure can be replayed byte-identically.

WHY BOTH TESTS:

  * agent-bench drives /v1/chat/completions, so that is the path graded here.
    The raw /v1/completions endpoint has no chat template and GLM-5.2 simply
    continues the text, which is base-LM behaviour, not a KV fault.
  * The 4 short prompts are ~5 tokens: ONE prefill chunk. They say nothing about
    the multi-chunk path. Case A prompts are 74K-235K tokens against an 8192
    per-rank chunk, i.e. 9-29 chunks each -- the regime where mooncake PD can
    RDMA-read KV pages the writing forward has not finished. The needle test is
    the one that actually covers the deployment being benchmarked.

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

# GLM-5.2 generation_config.json. Do NOT set these to 0 -- see the header.
TEMP = 1.0
TOP_P = 0.95


def chat(messages, max_tokens=256, seed=None):
    body = {
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMP, "top_p": TOP_P,
        "stream": False,
    }
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TMO) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    msg = ch["message"]
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    return text, ch.get("finish_reason"), (d.get("usage") or {}), time.time() - t0


print("=" * 78)
print(f"PART 1 - short factual, chat template, temp={TEMP} top_p={TOP_P}")
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
        text, fin, usage, dt = chat([{"role": "user", "content": q}], max_tokens=512, seed=1000 + i)
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

# Filler that is cheap to build and carries NO DIGITS, so the only 7-digit run
# in the whole prompt is the needle -- a partial retrieval cannot be confused
# with the model copying something else. This is what makes grading robust under
# non-greedy sampling.
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
for di, d in enumerate(depths):
    needle = str(rnd.randint(1000000, 9999999))
    cut = int(len(ftoks) * d)
    body = (tok.decode(ftoks[:cut])
            + f"\n\nThe secret access code is {needle}. Remember it.\n\n"
            + tok.decode(ftoks[cut:]))
    msgs = [{"role": "user", "content":
             body + "\n\nWhat is the secret access code mentioned above? "
                    "Reply with the 7-digit number only."}]
    try:
        # 2048, not 256: at 256 the reasoning is cut off mid-thought and the
        # run-on tail mimics corruption exactly.
        text, fin, usage, dt = chat(msgs, max_tokens=2048, seed=2000 + di)
        hit = needle in text
        ok2 += hit
        think = text.count("</think>")
        print(f"depth={d:4.0%} {'OK  ' if hit else 'FAIL'} {dt:7.2f}s finish={fin} "
              f"prompt_tok={usage.get('prompt_tokens')} "
              f"cached={(usage.get('prompt_tokens_details') or {}).get('cached_tokens')} "
              f"compl={usage.get('completion_tokens')} </think>x{think} "
              f"want={needle} :: {text[-90:]!r}")
    except Exception as e:
        print(f"depth={d:4.0%} FAIL {type(e).__name__}: {str(e)[:120]}")

print(f"\nneedle: {ok2}/{len(depths)}")
print(f"\nTOTAL short={ok1}/{len(CASES)} needle={ok2}/{len(depths)}")
sys.exit(0 if (ok1 == len(CASES) and ok2 == len(depths)) else 1)
