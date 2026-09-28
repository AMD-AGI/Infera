#!/usr/bin/env python3
"""Force decode HiCache write-back and host load-back, and check the answers.

Phases, against a live 1P1D deployment whose decode device pool is capped:
  seed    one question per seed document; decode inserts and writes it to host;
  flood   one question per flood document, sized well past the decode device
          cap per rank, so the seed documents are evicted from decode device
          memory but stay on the host (and on the much larger prefill device);
  reuse   a new question per seed document; decode should hit the host copy and
          load it back instead of receiving it from prefill.
Every question has a known answer. Decode /metrics is snapshotted around each
phase: hicache_backup_tokens_total (device->host), evicted_tokens_total,
load_back_tokens_total (host->device), hicache_host_used_tokens.

Usage:
    l2_probe.py --router URL --decode URL --model NAME --out FILE
Artifacts: one JSON file with every reply, metric snapshot, delta, and verdict.
Exit: 0 pass, 1 a check failed, 2 the deployment could not be probed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request

COUNTERS = (
    "sglang:hicache_backup_tokens_total",
    "sglang:evicted_tokens_total",
    "sglang:load_back_tokens_total",
    "sglang:hicache_host_used_tokens",
)
CITIES = ("Lisbon", "Osaka", "Quito", "Tallinn", "Nairobi", "Hobart", "Bergen",
          "Cusco", "Tbilisi", "Windhoek", "Valparaiso", "Ushuaia", "Kyoto")


def http(url: str, payload=None, timeout: float = 900.0):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def code(doc: int, record: int) -> str:
    return f"Z{(doc * 7919 + record * 31337) % 9973:04d}"


def document(doc: int, records: int) -> str:
    lines = [f"Record {i}: the city is {CITIES[(doc + i) % len(CITIES)]}, the population "
             f"is {(doc * 131 + i * 7919) % 100000}, and the code is {code(doc, i)}."
             for i in range(records)]
    return f"Registry {doc}. Answer questions about it.\n" + "\n".join(lines)


def ask(router: str, model: str, doc: int, records: int, record: int, max_tokens: int) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": [
        {"role": "system", "content": document(doc, records)},
        {"role": "user", "content": f"What is the code in record {record}? Answer briefly."}]}
    start = time.monotonic()
    try:
        d = json.loads(http(router + "/v1/chat/completions", body))
    except Exception as e:  # noqa: BLE001 - recorded as a failed request
        return {"doc": doc, "record": record, "error": f"{type(e).__name__}: {e}"}
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    return {
        "doc": doc, "record": record, "expected": code(doc, record),
        "correct": code(doc, record) in content,
        "content": content, "reasoning_head": (msg.get("reasoning_content") or "")[:200],
        "finish": d["choices"][0].get("finish_reason"), "usage": d.get("usage"),
        "elapsed_s": round(time.monotonic() - start, 3),
    }


def metrics(decode: str) -> dict:
    out: dict = {}
    for line in http(decode + "/metrics", timeout=30).splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in COUNTERS:
            out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def delta(after: dict, before: dict) -> dict:
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in COUNTERS}


def run(router, model, docs, records, record, max_tokens, concurrency) -> list:
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        return list(pool.map(lambda d: ask(router, model, d, records, record, max_tokens), docs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--router", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", type=int, default=740, help="~20k tokens per document")
    ap.add_argument("--seed-docs", type=int, default=8)
    ap.add_argument("--flood-docs", type=int, default=80)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    router, decode = args.router.rstrip("/"), args.decode.rstrip("/")
    seeds = list(range(1000, 1000 + args.seed_docs))
    floods = list(range(2000, 2000 + args.flood_docs))
    result: dict = {"args": vars(args), "metrics": {}, "delta": {}, "phases": {}}
    try:
        result["metrics"]["start"] = metrics(decode)
    except Exception as e:  # noqa: BLE001
        print(f"cannot read decode metrics: {e}", file=sys.stderr)
        return 2

    plan = (("seed", seeds, 100), ("flood", floods, 200), ("reuse", seeds, 600))
    previous = "start"
    for phase, docs, record in plan:
        t0 = time.monotonic()
        replies = run(router, args.model, docs, args.records, record, args.max_tokens,
                      1 if phase == "reuse" else args.concurrency)
        time.sleep(5)  # let write-through acks and metric updates land
        result["phases"][phase] = {"seconds": round(time.monotonic() - t0, 1), "replies": replies}
        result["metrics"][phase] = metrics(decode)
        result["delta"][phase] = delta(result["metrics"][phase], result["metrics"][previous])
        previous = phase
        ok = sum(1 for r in replies if r.get("correct"))
        print(f"{phase}: {ok}/{len(replies)} correct, delta={result['delta'][phase]}", flush=True)

    reuse = result["phases"]["reuse"]["replies"]
    checks = {
        "all_correct": all(r.get("correct") for p in result["phases"].values() for r in p["replies"]),
        "no_request_errors": not any("error" in r for p in result["phases"].values() for r in p["replies"]),
        "wrote_to_host": result["delta"]["seed"]["sglang:hicache_backup_tokens_total"] > 0,
        "evicted_during_flood": result["delta"]["flood"]["sglang:evicted_tokens_total"] > 0,
        "loaded_back_on_reuse": result["delta"]["reuse"]["sglang:load_back_tokens_total"] > 0,
    }
    prompt_tokens = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in reuse)
    result["reuse_prompt_tokens"] = prompt_tokens
    result["checks"] = checks
    result["pass"] = all(checks.values())
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({"pass": result["pass"], **checks, "reuse_prompt_tokens": prompt_tokens,
                      "reuse_load_back_tokens": result["delta"]["reuse"]["sglang:load_back_tokens_total"]}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
