#!/usr/bin/env python3
"""Is the looping a BUG, or just what greedy decoding does?

I have been treating "output collapsed into a repeating loop" as evidence of an
engine defect.  That is not sound on its own: greedy decoding is famously prone
to degenerate repetition (Holtzman et al., "The Curious Case of Neural Text
Degeneration"), and these prompts are raw base-LM completions with no chat
template.  Some baseline rate of looping is EXPECTED.

Nor is "same prompt gives different answers at temperature=0" automatically a
bug: sglang ships `--enable-deterministic-inference` ("batch invariant ops"),
whose existence says that WITHOUT it, batching legitimately perturbs reduction
order and can flip an argmax.  Our servers run with it False.

So neither "it loops" nor "it varies" is by itself a defect.  What would be a
defect is a LOAD-DEPENDENT difference: the model's own greedy behaviour cannot
depend on how many other requests happen to share the batch.

This measures the loop rate as a function of concurrency ONLY, everything else
held fixed -- same prompts, same server, same sampling params:

    conc=1   -> the model's intrinsic greedy loop rate (the baseline I lacked)
    conc=8   -> mild batching
    conc=128 -> the regime where I saw failures

If the rate is flat, looping is the model's nature and there is no bug here.
If it climbs with concurrency, batching is injecting corruption.
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

CYCLE = re.compile(r"(.{1,12}?)\1{5,}")


def loop_onset(s):
    n = len(s)
    if n < 60:
        return None

    def looping(i):
        tail = s[i:]
        return len(tail) >= 50 and bool(CYCLE.search(tail)) and len(set(tail)) < 15

    if not looping(max(0, n - 200)):
        return None
    if looping(0):
        return 0
    lo, hi = 0, n - 200
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if looping(mid):
            hi = mid
        else:
            lo = mid
    return hi


def gen(url, i, ntok, temp, tag):
    body = {"text": f"Explain quantum computing in detail, part {i}.",
            "sampling_params": {"max_new_tokens": ntok, "temperature": temp},
            "rid": f"{tag}-{i:04d}"}
    try:
        r = requests.post(f"{url}/generate", json=body, timeout=600)
        if r.status_code != 200:
            return {"i": i, "http": r.status_code}
        j = r.json()
        t = j.get("text", "")
        o = loop_onset(t)
        return {"i": i, "http": 200, "text": t, "onset": o,
                "n_chars": len(t), "uniq": len(set(t.strip())),
                "dp_rank": (j.get("meta_info") or {}).get("dp_rank")}
    except Exception as e:
        return {"i": i, "http": 0, "error": str(e)[:120]}


def run(url, idxs, conc, ntok, temp, tag):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        recs = list(ex.map(lambda i: gen(url, i, ntok, temp, tag), idxs))
    ok = [r for r in recs if r.get("http") == 200]
    loop = [r for r in ok if r.get("onset") is not None]
    return recs, ok, loop, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=64, help="how many prompts")
    ap.add_argument("--ntok", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--conc", default="1,8,128")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", default="/tmp/baseline")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    idxs = list(range(1, args.n + 1))
    print(f"prompts={args.n} ntok={args.ntok} temp={args.temp}  "
          f"(identical prompt set at every concurrency)\n")

    table = []
    per_conc_text = {}
    for conc in [int(c) for c in args.conc.split(",")]:
        tag = f"{args.tag}-c{conc}"
        recs, ok, loop, dt = run(args.url, idxs, conc, args.ntok, args.temp, tag)
        with open(os.path.join(args.out, f"{tag}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        rate = 100 * len(loop) / len(ok) if ok else float("nan")
        at0 = sum(1 for r in loop if r["onset"] == 0)
        table.append((conc, len(ok), len(loop), rate, at0, dt))
        per_conc_text[conc] = {r["i"]: r.get("text", "") for r in ok}
        print(f"  conc={conc:4d}: ok={len(ok):3d} looping={len(loop):3d} "
              f"({rate:5.2f}%)  onset0={at0}  {dt:.0f}s")

    print(f"\n{'conc':>6s} {'ok':>4s} {'loop':>5s} {'rate':>7s} {'onset=0':>8s}")
    for conc, nok, nl, rate, at0, dt in table:
        print(f"{conc:6d} {nok:4d} {nl:5d} {rate:6.2f}% {at0:8d}")

    # Cross-concurrency agreement: at temperature=0 the SAME prompt should give
    # the SAME text regardless of batch size, if execution were batch-invariant.
    concs = sorted(per_conc_text)
    if len(concs) >= 2:
        base = per_conc_text[concs[0]]
        print(f"\ntext agreement vs conc={concs[0]} (temperature={args.temp}):")
        for c in concs[1:]:
            other = per_conc_text[c]
            shared = [i for i in base if i in other]
            same = sum(1 for i in shared if base[i] == other[i])
            print(f"  conc={c:4d}: identical for {same}/{len(shared)} prompts")


if __name__ == "__main__":
    main()
