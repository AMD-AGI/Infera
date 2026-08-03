#!/usr/bin/env python3
"""Populate, then replay, a fixed long-prompt corpus so a kvd L3 read can be
ATTRIBUTED.

THE ATTRIBUTION PROBLEM. A latency win is not evidence that kvd did anything:
SGLang's in-GPU radix cache serves a repeated prefix without ever touching L3.
That is exactly why the counters read thousands of `sets` and ZERO `gets` in
steady state -- the GPU tier answers first. The only clean proof is
restart-and-replay:

    engine restarted   -> in-GPU radix cache is EMPTY
    kvd daemon alive   -> L3 still holds the pages
    replay same prompt -> if the pages come back, they came from L3

    WANT: gets_total climbs, hits_total climbs, sets_total FLAT.
    sets climbing instead means it re-stored rather than read, i.e. a miss.

WHY NOT REUSE THE PREDECESSOR PROBE. That one rebuilt its prompts by replaying a
shared RNG in the same call order as correctness.py -- so a single extra draw
anywhere silently changes every prompt, the radix prefix differs, and the replay
tests nothing while still printing plausible numbers. Here the corpus is a pure
function of (index, length): no shared RNG state, no ordering coupling. Both
phases call the same builder, so byte-identity is structural rather than
maintained by discipline.

Answers are irrelevant to this test -- only the PROMPT needs to be identical --
so max_tokens is tiny and the output is not graded. GLM-5.2's own sampling is
used anyway, to stay clear of the temperature-0 + MTP loop trap.

Usage: replay_probe.py URL [phase] [n_prompts] [tokens]
"""
import json
import random
import sys
import time
import urllib.request

from transformers import AutoTokenizer

URL = sys.argv[1].rstrip("/")
PHASE = sys.argv[2] if len(sys.argv) > 2 else "warm"
NPROMPT = int(sys.argv[3]) if len(sys.argv) > 3 else 5
NTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 120000
TOKENIZER = "/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4"
MODEL = "glm5.2-mxfp4"

tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "flint", "grove",
         "harbor", "ivory", "jasper", "kelp", "lumen", "maple", "nimbus"]


def prompt_for(idx: int) -> str:
    """Pure function of idx -- the same text in both phases, by construction."""
    rnd = random.Random(770000 + idx)          # own RNG, no shared state
    chunk = " ".join(rnd.choice(WORDS) for _ in range(4000))
    n = len(tok.encode(chunk, add_special_tokens=False))
    ids = tok.encode(" ".join([chunk] * (NTOK // n + 1)),
                     add_special_tokens=False)[:NTOK]
    return tok.decode(ids) + f"\n\nDocument {idx}. Reply with the word OK."


def send(text):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 8, "temperature": 1.0, "top_p": 0.95, "stream": False,
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    u = d.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    return u.get("prompt_tokens"), det.get("cached_tokens"), time.time() - t0


print(f"[{PHASE}] {NPROMPT} prompts x ~{NTOK:,} tokens")
for i in range(NPROMPT):
    try:
        ptok, cached, dt = send(prompt_for(i))
        pct = (100.0 * cached / ptok) if (ptok and cached) else 0.0
        print(f"  [{PHASE}] doc={i}  prompt={ptok}  cached={cached} ({pct:.1f}%)  {dt:6.1f}s")
    except Exception as e:
        print(f"  [{PHASE}] doc={i}  ERROR {type(e).__name__}: {str(e)[:110]}")
