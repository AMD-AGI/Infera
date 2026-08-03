#!/usr/bin/env python3
"""Is the needle failure DETERMINISTIC (KV corruption) or STOCHASTIC (a sampling
excursion into a repetition loop)? And is the repetitive filler causing it?

WHY THIS ROUND. The 2x2 refuted both simple explanations:

  cell                 run1              run2 (2x2)
  -------------------------------------------------------------
  SAME d=0.05          FAIL cached=None  OK   cached=120000
  SAME d=0.50          OK   cached=29952 FAIL cached=120000

The SAME prompt flipped in BOTH directions, and cache warmth does not predict
the outcome -- a warmer 120000-cached request failed while a colder 5952-cached
one passed. So it is neither "depth-specific" nor "cold cache".

What every failure DOES share: `finish=length`, i.e. it ran to the 2048 cap,
with either a `</think>` storm (873, 1118) or verbatim filler regurgitation.
Those are degenerate-GENERATION shapes, not failed RETRIEVAL -- FRESH d=0.05
even emitted `447190`, six of the seven digits of `4471903`, before looping.

TWO ARMS, one round:

  A. REPEAT the identical prompt 6x at a fully warm cache.
       all pass / all fail  -> deterministic -> KV state is the cause
       mixed                -> stochastic    -> sampling excursion
     KV corruption cannot be intermittent for a fixed prompt against a fixed
     cache; sampling at temperature 1.0 can.

  B. Same length and depth, but filler drawn from a 14-WORD list (as before)
     vs a ~2,900-WORD English vocabulary. The current filler repeats 4,000
     draws from 14 words about seven times over to reach 120K tokens, which is
     a pathological repetition-inducing input for a reasoning model. If the
     loop disappears on varied filler, the degeneration is the PROMPT's, not
     the KV's.

Both arms are graded on a 7-digit needle in digit-free filler, so a partial
retrieval cannot be confused with the model copying something else.

Usage: needle_determinism.py URL [tokens] [tokenizer]
"""
import json
import random
import string
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

NARROW = ["alpha", "bravo", "cedar", "delta", "ember", "flint", "grove",
          "harbor", "ivory", "jasper", "kelp", "lumen", "maple", "nimbus"]


def wide_vocab():
    """A digit-free, reasonably varied English-ish vocabulary. Built from
    /usr/share/dict/words when present, else synthesised -- either way it is
    two orders of magnitude larger than NARROW, which is the variable."""
    try:
        with open("/usr/share/dict/words") as f:
            ws = [w.strip().lower() for w in f
                  if w.strip().isalpha() and 4 <= len(w.strip()) <= 9]
        if len(ws) > 2000:
            return ws[:20000]
    except OSError:
        pass
    rnd = random.Random(7)
    cons, vow = "bcdfghjklmnprstvwz", "aeiou"
    out = set()
    while len(out) < 2900:
        out.add("".join(rnd.choice(cons) + rnd.choice(vow) for _ in range(3)))
    return sorted(out)


def build(seed, words):
    rnd = random.Random(seed)
    chunk = " ".join(rnd.choice(words) for _ in range(4000))
    n = len(tok.encode(chunk, add_special_tokens=False))
    filler = " ".join([chunk] * (NTOK // n + 1))
    return tok.encode(filler, add_special_tokens=False)[:NTOK]


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
    return text, ch.get("finish_reason"), (d.get("usage") or {}), time.time() - t0


HDR = (f"{'cell':<24} {'res':<5} {'t':>6} {'finish':<7} {'cached':>7} "
       f"{'compl':>6} {'</think>':>8}  tail")

narrow = build(20260731, NARROW)
WIDE = wide_vocab()
wide = build(20260731, WIDE)
print(f"filler: narrow_vocab={len(NARROW)} wide_vocab={len(WIDE)} "
      f"tokens={len(narrow):,}/{len(wide):,}")

print()
print("=== ARM A: identical prompt x6, warm cache (mixed => stochastic) ===")
print(HDR)
NEEDLE_A = "5271814"        # the SAME d=0.50 prompt from the 2x2
a_ok = 0
for i in range(6):
    try:
        text, fin, u, dt = ask(narrow, 0.50, NEEDLE_A, 4100 + i)
        ok = NEEDLE_A in text
        a_ok += ok
        print(f"{'A repeat #%d' % i:<24} {'OK' if ok else 'FAIL':<5} {dt:6.1f} "
              f"{str(fin):<7} {(u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0:>7} "
              f"{u.get('completion_tokens',0):>6} {text.count('</think>'):>8}  {text[-60:]!r}")
    except Exception as e:
        print(f"{'A repeat #%d' % i:<24} ERR   {type(e).__name__}: {str(e)[:80]}")
print(f"ARM A: {a_ok}/6 passed")

print()
print("=== ARM B: wide-vocabulary filler, same length/depths ===")
print(HDR)
b_ok = 0
B = [(0.05, "3958174", 4200), (0.50, "6127350", 4201), (0.95, "8043692", 4202)]
for depth, needle, seed in B:
    try:
        text, fin, u, dt = ask(wide, depth, needle, seed)
        ok = needle in text
        b_ok += ok
        print(f"{'B wide d=%.2f' % depth:<24} {'OK' if ok else 'FAIL':<5} {dt:6.1f} "
              f"{str(fin):<7} {(u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0:>7} "
              f"{u.get('completion_tokens',0):>6} {text.count('</think>'):>8}  {text[-60:]!r}")
    except Exception as e:
        print(f"{'B wide d=%.2f' % depth:<24} ERR   {type(e).__name__}: {str(e)[:80]}")
print(f"ARM B: {b_ok}/3 passed")
