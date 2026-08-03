#!/usr/bin/env python3
"""One discriminating round for the depth-5% needle failure.

THE OBSERVATION. In the first correctness run, depth 5% failed and depths
25/50/75/95 passed. The failure emitted `6159` -- the first four digits of the
wanted `6159362` -- then looped `</think>` 1118 times to the 2048 cap. That
partial-digits-then-loop shape is the documented mooncake early-send signature.

THE CONFOUND, which is why that run cannot settle it. `cached_tokens` climbed
monotonically across the five depths: None, 5952, 29952, 59968, 89984. Depth 5%
was the ONLY fully-cold prefill; every later depth re-used a growing cached
prefix and therefore exercised progressively LESS of the fresh chunk-transfer
path. So "early depth" and "cold cache" are perfectly confounded in that run,
and either could explain the result.

THE 2x2. Two binary variables, four cells, one round:

    filler       depth        what it isolates
    ---------------------------------------------------------------
    SAME (warm)  0.05         cold-vs-warm at the failing depth
    SAME (warm)  0.50         warm control (expected OK, sanity)
    FRESH (cold) 0.05         does a cold prefill fail REPRODUCIBLY at 5%?
    FRESH (cold) 0.50         does a cold prefill fail at MID depth too?

Reading it:
  * fresh-0.05 FAIL + fresh-0.50 OK      -> depth-specific on cold prefill
  * fresh-0.05 FAIL + fresh-0.50 FAIL    -> cold multi-chunk prefill generally
  * both fresh OK                        -> the original failure was a one-off
                                            sampling excursion, not a defect
  * same-0.05 OK                         -> warm cache masks it (as the
                                            predecessor kit measured but could
                                            not explain)

The fresh filler uses a different RNG seed, so its prompt shares no prefix with
the first run's -- that is what makes it genuinely cold. Sampling is GLM-5.2's
own (temp 1.0 / top_p 0.95); each cell sends a fixed `seed` so any failure can
be replayed byte-identically.

Usage: needle_2x2.py URL [tokens] [tokenizer]
"""
import json
import random
import sys
import time
import urllib.request

URL = sys.argv[1].rstrip("/")
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 120000
TOKENIZER = sys.argv[3] if len(sys.argv) > 3 else "/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4"
MODEL = "glm5.2-mxfp4"
TEMP, TOP_P = 1.0, 0.95

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)

WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "flint", "grove",
         "harbor", "ivory", "jasper", "kelp", "lumen", "maple", "nimbus"]
# A second, disjoint word list for the FRESH filler: different tokens from the
# very first block, so not one prefix block can hit the radix cache.
WORDS2 = ["quartz", "raven", "sable", "tundra", "umber", "vellum", "willow",
          "xenon", "yarrow", "zephyr", "onyx", "pewter", "cobalt", "amber"]


def build(seed, words):
    rnd = random.Random(seed)
    chunk = " ".join(rnd.choice(words) for _ in range(4000))
    n = len(tok.encode(chunk, add_special_tokens=False))
    filler = " ".join([chunk] * (NTOK // n + 1))
    return tok.encode(filler, add_special_tokens=False)[:NTOK], rnd


def ask(ftoks, depth, needle, seed):
    cut = int(len(ftoks) * depth)
    body = (tok.decode(ftoks[:cut])
            + f"\n\nThe secret access code is {needle}. Remember it.\n\n"
            + tok.decode(ftoks[cut:]))
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      body + "\n\nWhat is the secret access code mentioned above? "
                             "Reply with the 7-digit number only."}],
        "max_tokens": 2048, "temperature": TEMP, "top_p": TOP_P,
        "seed": seed, "stream": False,
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    m = ch["message"]
    text = (m.get("content") or "") + (m.get("reasoning_content") or "")
    u = d.get("usage") or {}
    return text, ch.get("finish_reason"), u, time.time() - t0


# SAME filler = the exact one the failing run used (seed 20260731, WORDS), so
# its blocks are already resident. Its needles must also be drawn in the same
# sequence to reproduce the same prompts.
same, rnd_same = build(20260731, WORDS)
same_needles = []
_r = random.Random(20260731)
# replay the generator: build() consumed 4000 draws for the chunk, then the
# original script drew one needle per depth from the SAME rnd -- so re-derive.
for _ in range(4000):
    _r.choice(WORDS)
for _ in range(5):
    same_needles.append(str(_r.randint(1000000, 9999999)))

fresh, _ = build(20260801, WORDS2)

print(f"filler tokens: same={len(same):,} fresh={len(fresh):,}")
print(f"{'cell':<22} {'res':<5} {'t':>7} {'finish':<8} {'prompt':>7} {'cached':>7} "
      f"{'compl':>6} {'</think>':>8}  tail")
rows = [
    ("SAME(warm) d=0.05", same,  0.05, same_needles[0], 3001),
    ("SAME(warm) d=0.50", same,  0.50, same_needles[2], 3002),
    ("FRESH(cold) d=0.05", fresh, 0.05, "4471903", 3003),
    ("FRESH(cold) d=0.50", fresh, 0.50, "8820516", 3004),
]
res = {}
for label, ft, depth, needle, seed in rows:
    try:
        text, fin, u, dt = ask(ft, depth, needle, seed)
        ok = needle in text
        res[label] = ok
        print(f"{label:<22} {'OK' if ok else 'FAIL':<5} {dt:7.2f} {str(fin):<8} "
              f"{u.get('prompt_tokens',0):>7} "
              f"{(u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0:>7} "
              f"{u.get('completion_tokens',0):>6} {text.count('</think>'):>8}  "
              f"want={needle} :: {text[-70:]!r}")
    except Exception as e:
        res[label] = None
        print(f"{label:<22} ERR   {type(e).__name__}: {str(e)[:100]}")

print()
print("verdict inputs:", {k: v for k, v in res.items()})
