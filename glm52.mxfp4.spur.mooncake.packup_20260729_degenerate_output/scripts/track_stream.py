#!/usr/bin/env python3
"""Streaming tracker: WHERE (which prefill/decode rank) and WHEN (which token
index) a request degenerates.

Two things the non-streaming tracker could not answer:

(a) which ranks.  `meta_info["dp_rank"]` is set in output_streamer.py:525 from
    `self.ps.dp_rank` -- the scheduler emitting the output -- so on a PD
    deployment it is the DECODE rank only.  The prefill rank is not reported
    back at all.  It IS however client-controllable: `disagg_prefill_dp_rank`
    pins the request to a chosen prefill rank (decode.py:527 short-circuits on
    it before the bootstrap_room/round-robin logic).  So we pin it and record
    what we pinned -- causal, not observational.

(b) when.  Streaming yields tokens as they are produced, so we can record the
    index of the first token that enters the degenerate loop.  In PD the
    prefill leg produces the FIRST token and decode produces the rest, so
    "degenerate from index 0" and "degenerate from index k>0" point at
    different legs.

Usage:
  track_stream.py --url http://IP:PORT --n 128 --ntok 512 [--pin-prefill N]
"""
import argparse
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

# A run of the same short cycle repeated -- "1.1.1.1", "9.9.1.2.3.9.9.1.2.3".
# Detects the loop rather than just low character diversity, so we can locate
# the token index at which the loop starts.
LOOP = re.compile(r"(.{1,8}?)\1{6,}")


def first_bad_index(tokens):
    """Index of the first token after which the tail is a repeating loop.

    Walks forward and asks "is everything from here on a loop?".  Returns None
    if the output never enters one.
    """
    for i in range(len(tokens)):
        tail = "".join(tokens[i:])
        if len(tail) < 40:
            break
        if LOOP.search(tail) and len(set(tail)) < 12:
            # found the loop; now make sure i is the EARLIEST such index by
            # construction (we scan forward), so return it
            return i
    return None


def is_degenerate(s):
    s = s.strip()
    if not s:
        return True
    return (len(set(s)) < 12
            or re.search(r"(.)\1{30,}", s) is not None
            or s.count("!") > len(s) * 0.3)


def one(args, i, run_id):
    rid = f"{run_id}-{i:04d}"
    body = {
        "text": f"Explain quantum computing in detail, part {i}.",
        "sampling_params": {"max_new_tokens": args.ntok, "temperature": args.temp},
        "rid": rid,
        "stream": True,
    }
    if args.pin_prefill is not None:
        body["disagg_prefill_dp_rank"] = args.pin_prefill

    rec = {"rid": rid, "i": i, "pinned_prefill": args.pin_prefill,
           "t_send": time.time()}
    toks, tstamps = [], []
    prev = ""
    try:
        r = requests.post(f"{args.url}/generate", json=body,
                          timeout=args.timeout, stream=True)
        rec["http"] = r.status_code
        if r.status_code != 200:
            rec["body_head"] = r.text[:200]
            return rec
        mi = {}
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "replace")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                j = json.loads(payload)
            except Exception:
                continue
            mi = j.get("meta_info", mi) or mi
            text = j.get("text", "")
            # sglang streams cumulative text by default; take the delta
            delta = text[len(prev):] if text.startswith(prev) else text
            prev = text if text.startswith(prev) else prev + text
            if delta:
                toks.append(delta)
                tstamps.append(time.time())
        full = prev
        rec["t_recv"] = time.time()
        rec["dp_rank_decode"] = mi.get("dp_rank")
        rec["completion_tokens"] = mi.get("completion_tokens")
        rec["prompt_tokens"] = mi.get("prompt_tokens")
        rec["finish_reason"] = (mi.get("finish_reason") or {}).get("type")
        for k in ("spec_accept_length", "spec_accept_rate", "spec_verify_ct"):
            rec[k] = mi.get(k)
        rec["rid_echoed"] = (mi.get("id") == rid)
        rec["n_chunks"] = len(toks)
        rec["degenerate"] = is_degenerate(full)
        rec["n_unique_chars"] = len(set(full.strip()))
        rec["text_head"] = full[:150]
        # The FIRST token is produced by the prefill leg; every later token
        # comes from decode.  So chunk 0 tells us which leg to blame, and it is
        # the single most important field here.
        rec["chunk0"] = toks[0] if toks else None
        rec["chunk1_5"] = toks[1:6]
        if tstamps:
            rec["t_first_tok"] = tstamps[0] - rec["t_send"]
        if rec["degenerate"]:
            bi = first_bad_index(toks)
            rec["first_bad_chunk"] = bi
            rec["chars_before_bad"] = len("".join(toks[:bi])) if bi is not None else None
            if bi is not None and tstamps:
                rec["t_to_bad"] = tstamps[min(bi, len(tstamps) - 1)] - tstamps[0]
            rec["prefix_before_bad"] = "".join(toks[:bi])[:200] if bi else ""
            # Keep the whole stream for degenerate requests -- the loop-detector
            # is a heuristic and missed a counting-sequence failure ("1.2.345...")
            # that was plainly degenerate, so store the raw evidence and judge
            # offline rather than trusting the predicate.
            rec["all_chunks"] = toks
            rec["chunk_dt"] = [round(tstamps[i] - tstamps[i - 1], 4)
                               for i in range(1, len(tstamps))]
    except Exception as e:
        rec["http"] = 0
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["t_recv"] = time.time()
    rec["wall"] = rec.get("t_recv", time.time()) - rec["t_send"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ntok", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=400)
    ap.add_argument("--tag", default="s")
    ap.add_argument("--out", default="/tmp/track")
    ap.add_argument("--pin-prefill", type=int, default=None,
                    help="pin every request to this prefill dp rank")
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
    bad = [r for r in ok if r.get("degenerate")]
    http = {}
    for r in recs:
        http[r.get("http")] = http.get(r.get("http"), 0) + 1
    print(f"run_id={run_id} n={args.n} pin_prefill={args.pin_prefill} http={http}")
    print(f"  coherent={len(ok) - len(bad)} degenerate={len(bad)}")
    if any(not r.get("rid_echoed") for r in ok):
        print("  !! rid echo failed -- correlation unreliable")

    from collections import Counter
    print(f"  decode dp_rank all : {dict(Counter(r.get('dp_rank_decode') for r in ok))}")
    print(f"  decode dp_rank BAD : {dict(Counter(r.get('dp_rank_decode') for r in bad))}")

    idx = [r.get("first_bad_chunk") for r in bad if r.get("first_bad_chunk") is not None]
    if idx:
        print(f"  first_bad_chunk: {sorted(idx)}")
        print(f"    at index 0 (i.e. broken from the very first token): "
              f"{sum(1 for x in idx if x == 0)}/{len(idx)}")
    for r in bad[:8]:
        print(f"     BAD {r['rid']} dec_dp={r.get('dp_rank_decode')} "
              f"chunks={r.get('n_chunks')} first_bad={r.get('first_bad_chunk')} "
              f"chars_before={r.get('chars_before_bad')} acc={r.get('spec_accept_length')}")
        if r.get("prefix_before_bad"):
            print(f"        clean prefix: {r['prefix_before_bad'][:110]!r}")
        print(f"        head: {r.get('text_head', '')[:110]!r}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
