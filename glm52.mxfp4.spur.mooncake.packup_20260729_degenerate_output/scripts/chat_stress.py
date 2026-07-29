#!/usr/bin/env python3
"""Concurrency stress via the CHAT endpoint with the model's own sampling.

Everything before this used POST /generate with a raw `text` field -- base-LM
completion, no chat template -- and forced temperature=0.  Both were wrong for
this model:

  * GLM-5.2 ships chat_template.jinja beginning `[gMASK]<sop>` and injecting a
    `<|system|>Reasoning Effort: ...` turn.  Sending raw text skips all of it.
  * generation_config.json recommends temperature=1.0, top_p=0.95.  Greedy
    decoding is famously prone to degenerate repetition, so temperature=0 was
    manufacturing the very symptom under investigation.

This runs /v1/chat/completions (server applies the template) at the model's
recommended sampling, and sweeps concurrency so "is it concurrency?" is
answered by comparison rather than assumed.

`--compare-raw` additionally runs the old raw+greedy path against the same
prompts, so the two are measured on one server in one sitting.
"""
import argparse
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

CYCLE = re.compile(r"(.{1,12}?)\1{5,}")


def loop_onset(s):
    n = len(s)
    if n < 60:
        return None

    def looping(i):
        t = s[i:]
        return len(t) >= 50 and bool(CYCLE.search(t)) and len(set(t)) < 15

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


PROMPT = "Explain quantum computing in detail, part {}."


def call_chat(url, model, i, ntok, temp, top_p):
    body = {"model": model,
            "messages": [{"role": "user", "content": PROMPT.format(i)}],
            "max_tokens": ntok, "temperature": temp, "top_p": top_p}
    try:
        r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=900)
        if r.status_code != 200:
            return {"i": i, "http": r.status_code, "body": r.text[:200]}
        j = r.json()
        ch = j["choices"][0]
        msg = ch.get("message") or {}
        txt = msg.get("content") or ""
        rsn = msg.get("reasoning_content") or ""
        return {"i": i, "http": 200, "text": txt, "reasoning_len": len(rsn),
                "finish": ch.get("finish_reason"),
                "onset": loop_onset(txt), "uniq": len(set(txt.strip())),
                "n_chars": len(txt),
                "completion_tokens": (j.get("usage") or {}).get("completion_tokens")}
    except Exception as e:
        return {"i": i, "http": 0, "error": f"{type(e).__name__}: {e}"[:150]}


def call_raw(url, i, ntok, temp, top_p):
    sp = {"max_new_tokens": ntok, "temperature": temp}
    if top_p is not None:
        sp["top_p"] = top_p
    body = {"text": PROMPT.format(i), "sampling_params": sp,
            "rid": f"raw-{uuid.uuid4().hex[:6]}-{i}"}
    try:
        r = requests.post(f"{url}/generate", json=body, timeout=900)
        if r.status_code != 200:
            return {"i": i, "http": r.status_code}
        j = r.json()
        txt = j.get("text", "")
        return {"i": i, "http": 200, "text": txt, "onset": loop_onset(txt),
                "uniq": len(set(txt.strip())), "n_chars": len(txt),
                "dp_rank": (j.get("meta_info") or {}).get("dp_rank")}
    except Exception as e:
        return {"i": i, "http": 0, "error": f"{type(e).__name__}: {e}"[:150]}


def sweep(fn, idxs, conc):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        recs = list(ex.map(fn, idxs))
    ok = [r for r in recs if r.get("http") == 200]
    loop = [r for r in ok if r.get("onset") is not None]
    return recs, ok, loop, time.time() - t0


def report(label, ok, loop, dt, recs):
    nerr = len(recs) - len(ok)
    rate = 100 * len(loop) / len(ok) if ok else float("nan")
    at0 = sum(1 for r in loop if r["onset"] == 0)
    print(f"  {label:34s} ok={len(ok):3d} err={nerr:2d} looping={len(loop):3d} "
          f"({rate:5.1f}%) onset0={at0} {dt:5.0f}s")
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="glm5.2-mxfp4")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ntok", type=int, default=512)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--conc", default="1,8,128")
    ap.add_argument("--compare-raw", action="store_true")
    ap.add_argument("--out", default="/tmp/chatstress")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    idxs = list(range(1, a.n + 1))
    run = uuid.uuid4().hex[:6]
    concs = [int(c) for c in a.conc.split(",")]

    print(f"prompts={a.n} ntok={a.ntok}  model={a.model}")
    print(f"CHAT endpoint (server applies chat_template), "
          f"temperature={a.temp} top_p={a.top_p}\n")

    texts = {}
    for c in concs:
        recs, ok, loop, dt = sweep(
            lambda i: call_chat(a.url, a.model, i, a.ntok, a.temp, a.top_p), idxs, c)
        report(f"chat t={a.temp} p={a.top_p} conc={c}", ok, loop, dt, recs)
        texts[c] = {r["i"]: r["text"] for r in ok}
        with open(os.path.join(a.out, f"chat-c{c}-{run}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        for r in loop[:2]:
            print(f"      onset={r['onset']} uniq={r['uniq']} "
                  f"{r['text'][max(0,(r['onset'] or 0)-50):(r['onset'] or 0)+60]!r}")

    if a.compare_raw:
        print()
        for label, temp, tp in (("raw  t=0     (the old test)", 0.0, None),
                                (f"raw  t={a.temp} p={a.top_p}", a.temp, a.top_p)):
            for c in concs:
                recs, ok, loop, dt = sweep(
                    lambda i: call_raw(a.url, i, a.ntok, temp, tp), idxs, c)
                report(f"{label} conc={c}", ok, loop, dt, recs)
                with open(os.path.join(a.out,
                          f"raw-t{temp}-c{c}-{run}.jsonl"), "w") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")

    if len(concs) >= 2:
        base = texts.get(concs[0], {})
        print(f"\n  text identical vs conc={concs[0]} "
              f"(expect NOT identical at temperature>0 -- sampling is random):")
        for c in concs[1:]:
            oth = texts.get(c, {})
            sh = [i for i in base if i in oth]
            same = sum(1 for i in sh if base[i] == oth[i])
            print(f"    conc={c:4d}: {same}/{len(sh)}")
    print(f"\n  -> {a.out}/*-{run}.jsonl")


if __name__ == "__main__":
    main()
