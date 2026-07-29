#!/usr/bin/env python3
"""Three questions I asserted answers to without measuring.

Q1  Is the looping caused by CONCURRENCY, or would 512 sequential requests
    show it too?  I only ever ran conc=128 and compared it to nothing.

Q2  Do temperature / top_p / top_k actually take effect?  I claimed
    `eagle_utils.py:620`'s `or _is_hip` discards them -- but that line is in
    the SPEC-DECODE verify path, and the no-MTP servers never execute it.  For
    the plain sampler I never checked at all.  Also: the model's own
    generation_config.json says temperature=1.0 / top_p=0.95, and sglang's
    `--sampling-defaults model` honours it -- so temperature=0 is a setting the
    model was never intended to run at, and *I* forced it.

Q3  Does the chat template matter?  GLM-5.2 ships chat_template.jinja starting
    with `[gMASK]<sop>` and injecting a `<|system|>Reasoning Effort: ...` turn.
    Every test so far sent RAW TEXT to /generate, i.e. base-LM completion with
    none of that.  A base LM rambling into a numbered list is not a bug.

Each section is a controlled comparison, not a code reading.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

CYCLE = re.compile(r"(.{1,12}?)\1{5,}")


def loops(s):
    if len(s) < 60:
        return False
    tail = s[-200:]
    return bool(CYCLE.search(tail)) and len(set(tail)) < 15


def raw(url, i, ntok, sp, rid):
    body = {"text": f"Explain quantum computing in detail, part {i}.",
            "sampling_params": dict(sp, max_new_tokens=ntok), "rid": rid}
    r = requests.post(f"{url}/generate", json=body, timeout=600)
    r.raise_for_status()
    j = r.json()
    return j.get("text", "")


def chat(url, i, ntok, sp, model):
    body = {"model": model,
            "messages": [{"role": "user",
                          "content": f"Explain quantum computing in detail, part {i}."}],
            "max_tokens": ntok}
    body.update(sp)
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"] or ""


def batch(fn, idxs, conc):
    with ThreadPoolExecutor(max_workers=conc) as ex:
        return list(ex.map(fn, idxs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="glm5.2-fp8")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ntok", type=int, default=512)
    ap.add_argument("--only", default="1,2,3")
    a = ap.parse_args()
    idxs = list(range(1, a.n + 1))
    only = set(a.only.split(","))

    # ---------------- Q2 first: do the knobs even work? ----------------
    if "2" in only:
        print("=" * 74)
        print("Q2  Do temperature / top_p / top_k take effect?")
        print("=" * 74)
        print("  Method: same prompt, same rid-free request, 4 samples per setting.")
        print("  If a knob works, temperature=1.0 must produce VARIED text while")
        print("  temperature=0 stays (mostly) fixed; top_k=1 must equal greedy.\n")
        settings = [
            ("temperature=0",            {"temperature": 0.0}),
            ("temperature=1.0",          {"temperature": 1.0}),
            ("temperature=1.0,top_p=.95", {"temperature": 1.0, "top_p": 0.95}),
            ("temperature=2.0",          {"temperature": 2.0}),
            ("top_k=1 (=greedy)",        {"temperature": 1.0, "top_k": 1}),
        ]
        for name, sp in settings:
            outs = [raw(a.url, 7, 64, sp, f"q2-{name}-{k}") for k in range(4)]
            uniq = len(set(outs))
            print(f"  {name:28s} distinct/4 = {uniq}   {outs[0][:52]!r}")
        print("\n  distinct=1 at temperature>0 would mean the knob is IGNORED.")
        print("  distinct>1 at temperature=0 means batching/nondeterminism, not sampling.\n")

    # ---------------- Q1: concurrency, or just volume? ----------------
    if "1" in only:
        print("=" * 74)
        print("Q1  Concurrency-caused, or would sequential requests loop too?")
        print("=" * 74)
        sp = {"temperature": 0.0}
        for label, conc in (("SEQUENTIAL (conc=1)", 1), ("conc=8", 8), ("conc=128", 128)):
            t0 = time.time()
            outs = batch(lambda i: raw(a.url, i, a.ntok, sp, f"q1-c{conc}-{i}"),
                         idxs, conc)
            nl = sum(loops(o) for o in outs)
            print(f"  {label:22s} n={len(outs)} looping={nl} "
                  f"({100*nl/len(outs):.1f}%)  {time.time()-t0:.0f}s")
        print("  Flat across rows => volume/greedy, not concurrency.\n")

    # ---------------- Q3: chat template ----------------
    if "3" in only:
        print("=" * 74)
        print("Q3  Does the chat template change anything?")
        print("=" * 74)
        for label, sp in (("temperature=0", {"temperature": 0.0}),
                          ("model default (t=1,p=.95)", {"temperature": 1.0, "top_p": 0.95})):
            rawo = batch(lambda i: raw(a.url, i, a.ntok, sp, f"q3r-{i}"), idxs, 8)
            cho = batch(lambda i: chat(a.url, i, a.ntok, sp, a.model), idxs, 8)
            nr, nc = sum(loops(o) for o in rawo), sum(loops(o) for o in cho)
            print(f"  [{label}]")
            print(f"     /generate  raw text  : looping {nr}/{len(rawo)} "
                  f"({100*nr/len(rawo):.1f}%)")
            print(f"     /v1/chat  templated  : looping {nc}/{len(cho)} "
                  f"({100*nc/len(cho):.1f}%)")
            print(f"     sample raw : {rawo[0][:64]!r}")
            print(f"     sample chat: {cho[0][:64]!r}")
        print()


if __name__ == "__main__":
    main()
