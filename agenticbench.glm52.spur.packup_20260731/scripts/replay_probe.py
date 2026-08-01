#!/usr/bin/env python3
"""Replay the exact long prompts that populated the kvd store, after an engine
restart, so that a read from L3 can be attributed.

THE ATTRIBUTION PROBLEM (CLAUDE.md): a latency win is NOT evidence that kvd did
anything. sglang's in-GPU radix cache serves a repeated prefix without ever
touching L3, so re-sending a prompt to a warm server proves nothing at all. The
only clean proof is restart-and-replay:

    engine restarted   -> in-GPU radix cache is EMPTY
    kvd daemon alive   -> L3 store still holds the pages (12,942 entries)
    replay same prompt -> if the pages come back, they came from L3

    WANT: gets_total climbs, hits_total climbs, sets_total FLAT.
    sets climbing instead means it re-stored rather than read, i.e. a miss.

The prompts are rebuilt bit-for-bit from correctness.py's generator: same seed
(20260731), same word list, same call order, so the filler and every needle are
identical to the run that populated the store. Any drift here and the radix
prefix differs and the replay silently tests nothing.
"""
import json
import random
import sys
import time
import urllib.request

from transformers import AutoTokenizer

URL = sys.argv[1].rstrip("/")
NEEDLE_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 120000
TOKENIZER = "/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4"
MODEL = "glm5.2-mxfp4"


def chat(messages, max_tokens=64):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    usage = d.get("usage") or {}
    det = usage.get("prompt_tokens_details") or {}
    return (usage.get("prompt_tokens"), det.get("cached_tokens"),
            d["choices"][0].get("finish_reason"), time.time() - t0)


tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

rnd = random.Random(20260731)
WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "flint", "grove",
         "harbor", "ivory", "jasper", "kelp", "lumen", "maple", "nimbus"]
chunk = " ".join(rnd.choice(WORDS) for _ in range(4000))
n_per = len(tok.encode(chunk, add_special_tokens=False))
reps = max(1, NEEDLE_TOK // n_per + 1)
ftoks = tok.encode(" ".join([chunk] * reps), add_special_tokens=False)[:NEEDLE_TOK]

depths = [0.05, 0.25, 0.50, 0.75, 0.95]
needles = [str(rnd.randint(1000000, 9999999)) for _ in depths]
print(f"filler {len(ftoks):,} tokens -- replaying {len(depths)} prompts")

for d, needle in zip(depths, needles):
    cut = int(len(ftoks) * d)
    body = (tok.decode(ftoks[:cut])
            + f"\n\nThe secret access code is {needle}. Remember it.\n\n"
            + tok.decode(ftoks[cut:]))
    msgs = [{"role": "user", "content":
             body + "\n\nWhat is the secret access code mentioned above? "
                    "Reply with the 7-digit number only."}]
    try:
        ptok, cached, fin, dt = chat(msgs)
        pct = (100.0 * cached / ptok) if (ptok and cached is not None) else 0.0
        print(f"  depth={d:.0%}  prompt={ptok}  cached={cached} ({pct:.1f}%)  "
              f"{dt:6.1f}s  finish={fin}")
    except Exception as e:
        print(f"  depth={d:.0%}  ERROR {type(e).__name__}: {e}")
