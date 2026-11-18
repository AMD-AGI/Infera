#!/usr/bin/env python3
"""Read-back check for the KV offload tiers: N distinct long prompts with a needle each, then the first ones again.
With a GPU cache and CPU tier smaller than the working set, the repeats are served from the lower tiers, so a wrong
chunk layout shows up as a wrong needle.
  tier_probe.py URL MODEL OUT [N_PROMPTS] [TOKENS_PER_PROMPT] [N_REPEAT]
Writes one JSON line per request and prints a summary.
"""
import json
import random
import sys
import time
import urllib.request

url, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 6
tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 24000
n_repeat = int(sys.argv[6]) if len(sys.argv) > 6 else 3
words = ("amber basil cobalt delta ember fjord granite harbor indigo juniper kestrel lantern meadow nectar orchid "
         "pebble quartz raven saffron timber umber velvet willow xenon yarrow zephyr").split()


def prompt(i):
    rng = random.Random(1000 + i)
    code = f"{rng.choice(words).upper()}-{rng.randint(1000, 9999)}"
    filler = " ".join(rng.choice(words) for _ in range(int(tokens * 0.75)))
    text = (f"Remember this access code: {code}.\n\n{filler}\n\n"
            "Question: what is the access code given at the very beginning? Answer with the code only.\nAnswer:")
    return code, text


def ask(text):
    body = json.dumps({"model": model, "prompt": text, "max_tokens": 12, "temperature": 0}).encode()
    req = urllib.request.Request(f"{url}/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        res = json.load(r)
    return res["choices"][0]["text"], res["usage"]["prompt_tokens"], time.time() - t0


order = list(range(n)) + list(range(n_repeat))
ok = 0
with open(out, "w") as f:
    for k, i in enumerate(order):
        code, text = prompt(i)
        try:
            ans, ptok, dt = ask(text)
            err = None
        except Exception as e:  # noqa: BLE001
            ans, ptok, dt, err = "", 0, 0.0, repr(e)
        passed = code in ans
        ok += passed
        rec = {"k": k, "prompt": i, "repeat": k >= n, "code": code, "answer": ans.strip(), "passed": passed,
               "prompt_tokens": ptok, "elapsed_s": round(dt, 2), "error": err}
        f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print(f"{ok}/{len(order)} passed", flush=True)
