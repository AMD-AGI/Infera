#!/usr/bin/env python3
"""Per-request lifecycle tracker for the GLM-5.2 PD degeneration bug.

Every request carries a client-supplied `rid`, so the same string appears in
the client record, the prefill log, the decode log and the router log.  That
turns "2% of outputs are garbage" into "these specific rids are garbage, and
here is where each of them went".

`meta_info` comes back with `id` (== our rid) and, when dp-attention is on,
`dp_rank` -- so the rank that served a request is recorded without having to
infer it from timing.

Usage:
    track_probe.py --url http://IP:PORT --n 128 --ntok 512 [--temp 0.0]
                   [--tag rN] [--out /tmp/track]
"""
import argparse
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests


def is_degenerate(s: str) -> bool:
    """Same predicate used by qcheck2.py, kept identical so counts compare."""
    s = s.strip()
    if not s:
        return True
    return (
        len(set(s)) < 12
        or re.search(r"(.)\1{30,}", s) is not None
        or s.count("!") > len(s) * 0.3
    )


def one(args, i, run_id):
    rid = f"{run_id}-{i:04d}"
    body = {
        "text": f"Explain quantum computing in detail, part {i}.",
        "sampling_params": {
            "max_new_tokens": args.ntok,
            "temperature": args.temp,
        },
        "rid": rid,
    }
    rec = {"rid": rid, "i": i, "t_send": time.time()}
    try:
        r = requests.post(f"{args.url}/generate", json=body, timeout=args.timeout)
        rec["http"] = r.status_code
        rec["t_recv"] = time.time()
        if r.status_code == 200:
            j = r.json()
            text = j.get("text", "")
            mi = j.get("meta_info", {}) or {}
            rec["meta_id"] = mi.get("id")
            rec["dp_rank"] = mi.get("dp_rank")
            rec["completion_tokens"] = mi.get("completion_tokens")
            rec["prompt_tokens"] = mi.get("prompt_tokens")
            rec["cached_tokens"] = mi.get("cached_tokens")
            rec["finish_reason"] = (mi.get("finish_reason") or {}).get("type")
            rec["e2e_latency"] = mi.get("e2e_latency")
            # Spec-decode telemetry is per-request.  If degenerate outputs come
            # with an anomalous accept rate the draft/verify path is implicated;
            # if their spec stats look normal, the corruption is elsewhere.
            for k in ("spec_accept_length", "spec_accept_rate", "spec_verify_ct",
                      "spec_num_correct_drafts", "spec_num_proposed_drafts"):
                rec[k] = mi.get(k)
            # Retraction = the scheduler evicted and re-ran the request under
            # memory pressure.  A prime suspect for concurrency-only corruption.
            rec["num_retractions"] = mi.get("num_retractions")
            rec["degenerate"] = is_degenerate(text)
            rec["text_head"] = text[:120]
            rec["text_tail"] = text[-120:]
            rec["n_unique_chars"] = len(set(text.strip()))
            # rid echo check: proves the server honoured our rid rather than
            # minting its own -- without this the correlation is worthless.
            rec["rid_echoed"] = (mi.get("id") == rid)
        else:
            rec["body_head"] = r.text[:200]
    except Exception as e:
        rec["http"] = 0
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["t_recv"] = time.time()
    rec["wall"] = rec["t_recv"] - rec["t_send"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ntok", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=400)
    ap.add_argument("--tag", default="t")
    ap.add_argument("--out", default="/tmp/track")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_id = f"{args.tag}-{uuid.uuid4().hex[:6]}"
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.n) as ex:
        recs = list(ex.map(lambda i: one(args, i, run_id), range(1, args.n + 1)))

    elapsed = time.time() - t0
    path = os.path.join(args.out, f"{run_id}.jsonl")
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    http = {}
    for r in recs:
        http[r.get("http")] = http.get(r.get("http"), 0) + 1
    ok = [r for r in recs if r.get("http") == 200]
    bad = [r for r in ok if r.get("degenerate")]
    not_echoed = [r for r in ok if not r.get("rid_echoed")]

    print(f"run_id={run_id}  n={args.n} ntok={args.ntok} temp={args.temp} "
          f"elapsed={elapsed:.1f}s")
    print(f"  http={http}")
    print(f"  coherent={len(ok) - len(bad)} degenerate={len(bad)}")
    if not_echoed:
        # If this fires the whole tracking premise is broken -- say so loudly
        # rather than reporting correlations built on mismatched ids.
        print(f"  !! rid NOT echoed for {len(not_echoed)} reqs -- tracking unreliable")

    ranks = {}
    for r in ok:
        k = r.get("dp_rank")
        ranks.setdefault(k, [0, 0])
        ranks[k][0] += 1
        if r.get("degenerate"):
            ranks[k][1] += 1
    print(f"  per dp_rank (total, degenerate): "
          f"{ {k: tuple(v) for k, v in sorted(ranks.items(), key=lambda x: (x[0] is None, x[0]))} }")

    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return sum(xs) / len(xs) if xs else float("nan")

    good = [r for r in ok if not r.get("degenerate")]
    print(f"  spec_accept_length: coherent={mean([r.get('spec_accept_length') for r in good]):.3f} "
          f"degenerate={mean([r.get('spec_accept_length') for r in bad]):.3f}")
    print(f"  retractions: coherent={sum(r.get('num_retractions') or 0 for r in good)} "
          f"degenerate={sum(r.get('num_retractions') or 0 for r in bad)}")

    for r in bad[:8]:
        print(f"     BAD rid={r['rid']} dp={r.get('dp_rank')} "
              f"tok={r.get('completion_tokens')} fin={r.get('finish_reason')} "
              f"uniq={r.get('n_unique_chars')} accept={r.get('spec_accept_length')} "
              f"retr={r.get('num_retractions')} wall={r['wall']:.1f}s")
        print(f"          head={r.get('text_head')!r}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
