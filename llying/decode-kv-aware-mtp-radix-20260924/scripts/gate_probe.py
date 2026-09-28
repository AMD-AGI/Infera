#!/usr/bin/env python3
"""Correctness gate for decode radix cache under MTP on a live 1P1D deployment.

Checks:
  1. discovery: what the router holds for each worker (KV endpoint, block size);
  2. coherence: yihou's three temperature-0 probes do not degenerate;
  3. decode prefix reuse: the same P+Q2 request, once with the decode cache
     holding P ("hit") and once with only the decode cache flushed ("miss"),
     must give identical greedy output. Before each, both legs are flushed and
     P+Q1 is sent first, so the prefill leg does identical work for P+Q2.
     flush_cache also clears the prefill HiCache host pool.
  4. decode metrics after each step (kv_evictable_tokens, spec_accept_*).

Usage:
    gate_probe.py --router URL --prefill URL --decode URL --model NAME --out FILE
Artifacts: one JSON file with every raw reply, metric snapshot, and the verdict.
Exit: 0 pass, 1 a check failed, 2 the deployment could not be probed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

COHERENCE_PROMPTS = (
    "What is 2+2? Answer with a single digit.",
    "List the first 10 prime numbers in order, comma separated.",
    "Reply with exactly: PROBE_OK_7391",
)
METRICS = (
    "sglang:kv_evictable_tokens",
    "sglang:token_usage",
    "sglang:spec_accept_length",
    "sglang:spec_accept_rate",
    "sglang:cache_hit_rate",
)
CITIES = ("Lisbon", "Osaka", "Quito", "Tallinn", "Nairobi", "Hobart", "Bergen",
          "Cusco", "Tbilisi", "Windhoek", "Valparaiso", "Ushuaia", "Kyoto")


def http(url: str, payload=None, timeout: float = 300.0, raw: bool = False):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST" if data or url.endswith("flush_cache") else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
        return body if raw else json.loads(body)


def document(records: int) -> str:
    lines = [f"Record {i}: the city is {CITIES[i % len(CITIES)]}, the population "
             f"is {(i * 7919) % 100000}, and the code is Z{(i * 31337) % 9973:04d}."
             for i in range(records)]
    return "You are given a registry. Answer questions about it.\n" + "\n".join(lines)


def chat(router: str, model: str, messages: list, max_tokens: int) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": max_tokens}
    start = time.monotonic()
    d = http(router + "/v1/chat/completions", body)
    msg = d["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "finish": d["choices"][0].get("finish_reason"),
        "usage": d.get("usage"),
        "elapsed_s": round(time.monotonic() - start, 3),
    }


def degenerate(text: str) -> bool:
    return text.strip().startswith("1!") or bool(re.search(r"(.{1,8}?)\1{4,}", text))


def flush(url: str) -> str:
    for _ in range(30):
        try:
            return http(url + "/flush_cache", raw=True, timeout=60).strip()[:120]
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            time.sleep(2)  # engine not idle yet
    raise RuntimeError(f"flush_cache kept failing on {url}")


def metrics(url: str) -> dict:
    out: dict = {}
    for line in http(url + "/metrics", raw=True, timeout=30).splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in METRICS:
            series, value = line.rsplit(" ", 1)
            out[series] = float(value)
    return out


def first_divergence(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--router", required=True)
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--trials", type=int, default=2)
    args = ap.parse_args()
    router, prefill, decode = (u.rstrip("/") for u in (args.router, args.prefill, args.decode))
    result: dict = {"args": vars(args), "checks": {}}
    ok = True

    try:
        result["workers"] = http(router + "/v1/workers")
    except Exception as e:  # noqa: BLE001
        print(f"cannot reach router: {e}", file=sys.stderr)
        return 2

    coherence = []
    for prompt in COHERENCE_PROMPTS:
        r = chat(router, args.model, [{"role": "user", "content": prompt}], 512)
        r["prompt"] = prompt
        r["degenerate"] = degenerate(r["content"]) or degenerate(r["reasoning"])
        r["empty"] = not (r["content"].strip() or r["reasoning"].strip())
        coherence.append(r)
    result["coherence"] = coherence
    coherent = not any(r["degenerate"] or r["empty"] for r in coherence)
    echo_exact = "PROBE_OK_7391" in coherence[2]["content"]
    result["checks"]["coherence"] = coherent
    result["checks"]["echo_exact"] = echo_exact
    ok &= coherent and echo_exact

    doc = document(args.records)
    q1 = [{"role": "system", "content": doc},
          {"role": "user", "content": "Which city appears in record 17? Answer briefly."}]
    q2 = [{"role": "system", "content": doc},
          {"role": "user", "content": "What is the code in record 311? Answer briefly."}]
    trials = []
    for t in range(args.trials):
        trial: dict = {}
        trial["flush_hit"] = [flush(prefill), flush(decode)]
        trial["r1"] = chat(router, args.model, q1, args.max_tokens)
        trial["metrics_after_r1"] = metrics(decode)
        trial["r2_hit"] = chat(router, args.model, q2, args.max_tokens)
        trial["metrics_after_r2_hit"] = metrics(decode)
        trial["flush_miss"] = [flush(prefill), flush(decode)]
        trial["r1b"] = chat(router, args.model, q1, args.max_tokens)
        trial["flush_decode_only"] = flush(decode)
        trial["metrics_after_decode_flush"] = metrics(decode)
        trial["r2_miss"] = chat(router, args.model, q2, args.max_tokens)
        full = {k: trial[k]["reasoning"] + "\n----\n" + trial[k]["content"]
                for k in ("r1", "r1b", "r2_hit", "r2_miss")}
        trial["r1_equal_r1b"] = full["r1"] == full["r1b"]
        trial["r2_hit_equal_miss"] = full["r2_hit"] == full["r2_miss"]
        trial["r2_first_divergence"] = first_divergence(full["r2_hit"], full["r2_miss"])
        trial["degenerate"] = any(degenerate(v) for v in full.values())
        trials.append(trial)
    result["prefix_trials"] = trials
    reuse_equal = all(t["r2_hit_equal_miss"] for t in trials)
    result["checks"]["prefix_reuse_equal"] = reuse_equal
    result["checks"]["prefix_no_degenerate"] = not any(t["degenerate"] for t in trials)
    ok &= reuse_equal and not any(t["degenerate"] for t in trials)

    result["metrics_final"] = metrics(decode)
    result["pass"] = bool(ok)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({"pass": result["pass"], **result["checks"],
                      "trials": [{k: t[k] for k in ("r1_equal_r1b", "r2_hit_equal_miss",
                                                     "r2_first_divergence")} for t in trials]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
