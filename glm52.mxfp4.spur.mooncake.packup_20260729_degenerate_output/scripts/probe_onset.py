#!/usr/bin/env python3
"""Where does a 'coherent' output actually break?

My degeneracy predicate looked at the WHOLE string (unique-char count, a long
single-char run).  A 512-token output that is 200 tokens of good prose and then
300 tokens of "1.1.1.1." has plenty of unique characters, so it passed as
coherent.  Spot-checking stored tails showed ~2% of the "coherent" bucket is
looping at the end -- so the real failure rate is higher than reported and the
"degenerate" cases are just the ones that broke at token 1.

This tool keeps the FULL text and reports the character offset where the tail
becomes periodic, so we can ask: is this one failure mode with a variable onset,
or two different things?
"""
import argparse
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

CYCLE = re.compile(r"(.{1,12}?)\1{5,}")


def loop_onset(s):
    """Smallest i such that s[i:] is (almost) entirely a repeating cycle.

    Binary-search-free: walk a coarse grid then refine, so 512-token strings
    stay cheap.
    """
    n = len(s)
    if n < 60:
        return None

    def looping(i):
        tail = s[i:]
        if len(tail) < 50:
            return False
        m = CYCLE.search(tail)
        # require the cycle to cover most of the tail, not just appear in it
        return bool(m) and len(set(tail)) < 15

    if not looping(max(0, n - 200)):
        return None                      # the end is fine -> no loop
    lo, hi = 0, n - 200
    if looping(0):
        return 0
    while lo < hi - 1:                   # invariant: looping(hi), not looping(lo)
        mid = (lo + hi) // 2
        if looping(mid):
            hi = mid
        else:
            lo = mid
    return hi


def one(args, i, run_id):
    rid = f"{run_id}-{i:04d}"
    body = {"text": f"Explain quantum computing in detail, part {i}.",
            "sampling_params": {"max_new_tokens": args.ntok,
                                "temperature": args.temp},
            "rid": rid}
    rec = {"rid": rid, "i": i}
    try:
        r = requests.post(f"{args.url}/generate", json=body, timeout=args.timeout)
        rec["http"] = r.status_code
        if r.status_code != 200:
            return rec
        j = r.json()
        txt = j.get("text", "")
        mi = j.get("meta_info") or {}
        rec["dp_rank"] = mi.get("dp_rank")
        rec["completion_tokens"] = mi.get("completion_tokens")
        rec["full_text"] = txt
        rec["n_chars"] = len(txt)
        rec["n_unique"] = len(set(txt.strip()))
        onset = loop_onset(txt)
        rec["loop_onset_chars"] = onset
        rec["loop_frac"] = (1 - onset / len(txt)) if (onset is not None and txt) else 0.0
    except Exception as e:
        rec["http"] = 0
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ntok", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=400)
    ap.add_argument("--tag", default="on")
    ap.add_argument("--out", default="/tmp/onset")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_id = f"{args.tag}-{uuid.uuid4().hex[:6]}"
    with ThreadPoolExecutor(max_workers=args.n) as ex:
        recs = list(ex.map(lambda i: one(args, i, run_id), range(1, args.n + 1)))

    path = os.path.join(args.out, f"{run_id}.jsonl")
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in recs if r.get("http") == 200]
    looped = [r for r in ok if r.get("loop_onset_chars") is not None]
    at0 = [r for r in looped if r["loop_onset_chars"] == 0]
    print(f"run_id={run_id} n={args.n} ok={len(ok)}")
    print(f"  looping at all : {len(looped)}/{len(ok)} = {100*len(looped)/max(1,len(ok)):.1f}%")
    print(f"    onset == 0   : {len(at0)}  (broken from the first token)")
    print(f"    onset  > 0   : {len(looped)-len(at0)}  (good prose, then collapses)")
    if looped:
        on = sorted(r["loop_onset_chars"] for r in looped)
        print(f"  onset chars: min={on[0]} p25={on[len(on)//4]} med={on[len(on)//2]} max={on[-1]}")
    for r in sorted(looped, key=lambda x: x["loop_onset_chars"])[:6]:
        o = r["loop_onset_chars"]
        t = r["full_text"]
        print(f"  --- {r['rid']} dp={r.get('dp_rank')} onset={o} "
              f"({100*r['loop_frac']:.0f}% of output is loop)")
        print(f"      before: {t[max(0,o-90):o]!r}")
        print(f"      after : {t[o:o+70]!r}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
