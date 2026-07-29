#!/usr/bin/env python3
"""Is the degenerate output actually a BUG, or just what this prompt does?

The prompts are sent to /generate with a raw `text` field -- no chat template.
So the model is doing base-LM completion on:

    "Explain quantum computing in detail, part 31."

A numbered-list continuation ("1. ... 2. ... 3. ...") is a *perfectly
plausible* completion of that string, and greedy decoding is notorious for
falling into repetition loops.  So "the output is a loop" is NOT by itself
evidence of an engine bug.  I never checked this baseline before calling it a
bug.

The discriminator is not the shape of the output, it is REPRODUCIBILITY:
with temperature=0 the same prompt must give the same answer.  So:

  solo   -- the exact prompts that failed under load, run one at a time
  repeat -- the same prompt several times solo, to check solo is itself stable

If a prompt loops solo too, the loop is the model's honest greedy output and
the "bug" is my benchmark.  If it is clean solo but loops under load, the
concurrency path is corrupting it.
"""
import argparse
import json
import re
import sys
import time

import requests


def is_degenerate(s):
    s = s.strip()
    if not s:
        return True
    return (len(set(s)) < 12
            or re.search(r"(.)\1{30,}", s) is not None
            or s.count("!") > len(s) * 0.3)


def gen(url, text, ntok, temp=0.0, rid=None, chat=False):
    body = {"text": text,
            "sampling_params": {"max_new_tokens": ntok, "temperature": temp}}
    if rid:
        body["rid"] = rid
    r = requests.post(f"{url}/generate", json=body, timeout=400)
    r.raise_for_status()
    j = r.json()
    return j.get("text", ""), (j.get("meta_info") or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--idx", required=True,
                    help="comma-separated prompt indices that failed under load")
    ap.add_argument("--ntok", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    idxs = [int(x) for x in args.idx.split(",")]
    print(f"=== SOLO replay of {len(idxs)} prompts that degenerated under "
          f"conc=128, temperature=0, {args.repeat}x each ===\n")

    solo_bad = solo_tot = 0
    unstable = []
    for i in idxs:
        prompt = f"Explain quantum computing in detail, part {i}."
        outs = []
        for k in range(args.repeat):
            t = time.time()
            txt, mi = gen(args.url, prompt, args.ntok,
                          rid=f"SOLO-{i}-{k}")
            outs.append(txt)
            d = is_degenerate(txt)
            solo_tot += 1
            solo_bad += int(d)
            print(f"  part {i:3d} run {k}: {'DEGENERATE' if d else 'coherent  '} "
                  f"uniq={len(set(txt.strip())):3d} dp={mi.get('dp_rank')} "
                  f"{time.time()-t:.1f}s  {txt[:60]!r}")
        # temperature=0 solo should be byte-identical across repeats
        if len(set(outs)) != 1:
            unstable.append(i)
            print(f"      !! {len(set(outs))} DIFFERENT outputs for the same "
                  f"prompt at temperature=0 -- solo is not deterministic either")
    print(f"\nsolo: {solo_bad}/{solo_tot} degenerate")
    print(f"solo non-deterministic prompts: {unstable}")


if __name__ == "__main__":
    main()
