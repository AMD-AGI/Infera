#!/usr/bin/env python3
"""Functional check of the prefill HiCache path (write_back): a prefix pushed out of the GPU pool to
host memory and loaded back must give the same first token as when it was computed fresh.

  A. probes: PROBES distinct prompts of ~PROBE_TOKENS tokens, each with its own session id,
     thinking off, max_tokens 1 (the token comes from the prefill forward, so MTP simulation does
     not enter), top-5 logprobs recorded;
  B. fillers: FILLERS distinct prompts of ~FILLER_TOKENS tokens, more than the prefill GPU pools hold,
     so the probes' pages are evicted (under write_back that is when they are copied to host);
  C. probes again, same text and session ids (the router must run with prefill session affinity so
     they reach the same DP rank): first token must match A, logprobs within LOGPROB_TOL.

Pass needs: no failed request; every probe's first token equal in A and C; the prefill counters show
evictions and host backups during B and host load-back during C of at least half the probe tokens
(otherwise the probes never left the GPU and the host path was not exercised: exit 2, inconclusive).

Usage: hicache_check.py ROUTER_URL MODEL OUT_JSON PREFILL_METRICS_URL [PREFILL_METRICS_URL ...]
Env: PROBES=16 PROBE_TOKENS=40000 FILLERS=240 FILLER_TOKENS=120000 CONCURRENCY=48 LOGPROB_TOL=0.1
Exit 0 pass, 1 fail, 2 inconclusive.
"""

import concurrent.futures
import json
import os
import random
import re
import sys
import time
import urllib.request

ROUTER, MODEL, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
METRICS = sys.argv[4:]
PROBES = int(os.environ.get("PROBES", 16))
PROBE_TOKENS = int(os.environ.get("PROBE_TOKENS", 40000))
FILLERS = int(os.environ.get("FILLERS", 240))
FILLER_TOKENS = int(os.environ.get("FILLER_TOKENS", 120000))
CONCURRENCY = int(os.environ.get("CONCURRENCY", 48))
LOGPROB_TOL = float(os.environ.get("LOGPROB_TOL", 0.1))
COUNTERS = ("sglang:evicted_tokens_total", "sglang:hicache_backup_tokens_total",
            "sglang:load_back_tokens_total", "sglang:cached_tokens_total", "sglang:prompt_tokens_total")
SAMPLE = re.compile(r"^(sglang:[a-z_]+)\{[^}]*\} ([0-9.eE+-]+)$")
WORDS = ("river stone paper window garden silver orange planet market winter engine signal camera "
         "forest pocket ladder candle bridge harbor rocket violin desert castle mirror button tunnel "
         "anchor feather jungle marble needle saddle temple velvet walnut yellow zipper").split()


def text(seed: int, tokens: int) -> str:
    rng = random.Random(seed)
    # Roughly one token per word plus separators for this vocabulary and the GLM tokenizer.
    return " ".join(rng.choice(WORDS) for _ in range(int(tokens / 1.15)))


def counters() -> dict:
    total = dict.fromkeys(COUNTERS, 0.0)
    for url in METRICS:
        for line in urllib.request.urlopen(url, timeout=15).read().decode().splitlines():
            m = SAMPLE.match(line)
            if m and m.group(1) in total:
                total[m.group(1)] += float(m.group(2))
    return total


def ask(prompt: str, session: str, logprobs: bool) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if logprobs:
        body.update(logprobs=True, top_logprobs=5)
    request = urllib.request.Request(
        ROUTER + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Dynamo-Session-ID": session})
    start = time.time()
    reply = json.load(urllib.request.urlopen(request, timeout=1800))
    choice = reply["choices"][0]
    top = None
    content = (choice.get("logprobs") or {}).get("content") or []
    if content:
        top = {t["token"]: t["logprob"] for t in content[0].get("top_logprobs") or []}
        top.setdefault(content[0]["token"], content[0]["logprob"])
    return {"text": choice["message"].get("content") or "", "top": top, "seconds": round(time.time() - start, 2),
            "usage": reply.get("usage")}


def run(jobs, logprobs):
    results, errors = {}, []
    with concurrent.futures.ThreadPoolExecutor(CONCURRENCY) as pool:
        futures = {pool.submit(ask, prompt, session, logprobs): key for key, prompt, session in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                results[futures[future]] = future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{futures[future]}: {exc}")
    return results, errors


def delta(a: dict, b: dict) -> dict:
    return {k.removeprefix("sglang:"): b[k] - a[k] for k in COUNTERS}


def main() -> int:
    probes = [(i, text(1000 + i, PROBE_TOKENS) + "\n\nReply with the single word that appears most often above.",
               f"hicache-probe-{i}") for i in range(PROBES)]
    c0 = counters()
    first, err_a = run(probes, True)
    c1 = counters()
    fillers = [(i, text(5000 + i, FILLER_TOKENS) + "\n\nReply with one word.", f"hicache-filler-{i}")
               for i in range(FILLERS)]
    _, err_b = run(fillers, False)
    c2 = counters()
    second, err_c = run(probes, True)
    c3 = counters()

    mismatches, max_dlp = [], 0.0
    for i, _, _ in probes:
        a, c = first.get(i), second.get(i)
        if a is None or c is None:
            continue
        if a["text"] != c["text"]:
            mismatches.append({"probe": i, "fresh": a["text"], "reloaded": c["text"]})
        if a["top"] and c["top"]:
            for token, lp in a["top"].items():
                if token in c["top"]:
                    max_dlp = max(max_dlp, abs(lp - c["top"][token]))
    phase_b, phase_c = delta(c1, c2), delta(c2, c3)
    probe_tokens = PROBES * PROBE_TOKENS
    exercised = (phase_b["evicted_tokens_total"] > 0 and phase_b["hicache_backup_tokens_total"] > 0
                 and phase_c["load_back_tokens_total"] >= 0.5 * probe_tokens)
    errors = err_a + err_b + err_c
    failed = bool(errors or mismatches or max_dlp > LOGPROB_TOL)
    verdict = "fail" if failed else ("pass" if exercised else "inconclusive")
    summary = {
        "verdict": verdict, "errors": errors[:20], "mismatches": mismatches, "max_logprob_delta": round(max_dlp, 4),
        "logprobs_returned": any(r["top"] for r in first.values()),
        "phase_a": delta(c0, c1), "phase_b": phase_b, "phase_c": phase_c, "probe_tokens": probe_tokens,
        "probe_seconds": {"fresh": sorted(r["seconds"] for r in first.values()),
                          "reloaded": sorted(r["seconds"] for r in second.values())},
        "params": {"probes": PROBES, "probe_tokens": PROBE_TOKENS, "fillers": FILLERS,
                   "filler_tokens": FILLER_TOKENS, "concurrency": CONCURRENCY},
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("verdict", "max_logprob_delta", "logprobs_returned",
                                             "phase_b", "phase_c")}, indent=1))
    print(f"errors {len(errors)}, mismatches {len(mismatches)}; details in {OUT}")
    return {"pass": 0, "fail": 1, "inconclusive": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
