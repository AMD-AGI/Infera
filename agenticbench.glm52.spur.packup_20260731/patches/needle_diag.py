#!/usr/bin/env python3
"""Diagnose the needle FAILs: is it retrieval (KV/transport) or output budget?

The failing outputs contain the CORRECT needle digits and then repeat, e.g.
want=6159362 -> 'the secret access code is 6159\nThe secret access code is 6159...'
That points at degenerate repetition truncated by max_tokens, NOT at partial KV
(which would yield WRONG digits).

This distinguishes them directly, on the two depths that failed (5%, 95%):

  A) max_tokens 256 -> 1024      : if a bigger budget alone fixes it, the model
                                   was retrieving correctly all along.
  B) finish_reason                : 'length' means we cut it off; 'stop' means
                                   the model really ended there.
  C) digits present anywhere      : does the correct 7-digit run appear at all?

Nothing here changes the server; it is a client-side probe only.
"""
import json
import re
import sys
import time
import urllib.request

import random
from transformers import AutoTokenizer

URL = sys.argv[1].rstrip("/")
NEEDLE_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 120000
TOKENIZER = "/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4"
MODEL = "glm5.2-mxfp4"


def chat(messages, max_tokens):
    body = json.dumps({
        "model": MODEL, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    m = ch["message"]
    return ((m.get("content") or "") + (m.get("reasoning_content") or ""),
            ch.get("finish_reason"), d.get("usage") or {}, time.time() - t0)


tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

# Rebuild the EXACT filler and needles correctness.py uses: same seed, same
# call order, so depth i gets the same 7-digit needle as in that run.
rnd = random.Random(20260731)
WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "flint", "grove",
         "harbor", "ivory", "jasper", "kelp", "lumen", "maple", "nimbus"]
chunk = " ".join(rnd.choice(WORDS) for _ in range(4000))
n_per = len(tok.encode(chunk, add_special_tokens=False))
reps = max(1, NEEDLE_TOK // n_per + 1)
ftoks = tok.encode(" ".join([chunk] * reps), add_special_tokens=False)[:NEEDLE_TOK]
print(f"filler {len(ftoks):,} tokens")

depths = [0.05, 0.25, 0.50, 0.75, 0.95]
needles = [str(rnd.randint(1000000, 9999999)) for _ in depths]

# Only re-probe the depths that failed at max_tokens=256.
for d, needle in zip(depths, needles):
    if d not in (0.05, 0.95):
        continue
    cut = int(len(ftoks) * d)
    body = (tok.decode(ftoks[:cut])
            + f"\n\nThe secret access code is {needle}. Remember it.\n\n"
            + tok.decode(ftoks[cut:]))
    msgs = [{"role": "user", "content":
             body + "\n\nWhat is the secret access code mentioned above? "
                    "Reply with the 7-digit number only."}]
    for mt in (256, 1024):
        text, fin, usage, dt = chat(msgs, mt)
        exact = needle in text
        digits = re.findall(r"\d{7}", text)
        print(f"\ndepth={d:.0%} max_tokens={mt} want={needle}")
        print(f"  exact={exact} finish={fin} {dt:.1f}s "
              f"completion_tok={usage.get('completion_tokens')}")
        print(f"  7-digit runs found: {sorted(set(digits))[:6]}")
        print(f"  tail: {text[-120:]!r}")
