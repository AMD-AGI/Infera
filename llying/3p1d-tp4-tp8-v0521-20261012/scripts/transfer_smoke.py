#!/usr/bin/env python3
"""KV-transfer smoke test after an engine restart: N concurrent chat requests with distinct
prompts (~TOKENS tokens each) through the router, then checks that every response has content
and that the decode log shows no transfer failure since the start, and which decode DP ranks ran.

Usage: transfer_smoke.py ROUTER_URL MODEL DECODE_LOG [N=96] [TOKENS=8000]
Exit 0 only if all responses are non-empty and there are 0 "Decode transfer failed" lines.
"""

import concurrent.futures
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone


def one(url, model, i, tokens):
    words = " ".join(f"item{i}-{k} value {(k * 7919 + i * 104729) % 100003}." for k in range(tokens // 6))
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": f"smoke {i} {time.time()}: {words}"}],
                       "max_tokens": 8, "temperature": 0}).encode()
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=body,
                                   headers={"Content-Type": "application/json"}), timeout=600)
        j = json.load(r)
        text = (j["choices"][0]["message"].get("content") or "") + (j["choices"][0]["message"].get("reasoning_content") or "")
        return bool(text.strip()), time.time() - t, None
    except Exception as exc:  # noqa: BLE001
        return False, time.time() - t, str(exc)[:120]


def main():
    url, model, decode_log = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 96
    tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 8000
    start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    with concurrent.futures.ThreadPoolExecutor(n) as pool:
        results = list(pool.map(lambda i: one(url, model, i, tokens), range(n)))
    time.sleep(5)
    ok = sum(r[0] for r in results)
    errors = [r[2] for r in results if r[2]]
    fails, ranks = {}, set()
    dec = re.compile(r"DP(\d+) TP\d+\] Decode (transfer failed|batch)")
    for line in open(decode_log, errors="replace"):
        if line[:19] < start:
            continue
        m = dec.search(line)
        if m:
            if m.group(2) == "batch":
                ranks.add(int(m.group(1)))
            else:
                fails[int(m.group(1))] = fails.get(int(m.group(1)), 0) + 1
    lat = sorted(r[1] for r in results)
    print(f"since {start}: {ok}/{n} non-empty, {len(errors)} errors {errors[:2]}, latency p50 {lat[len(lat) // 2]:.1f}s"
          f" max {lat[-1]:.1f}s; decode transfer failures by DP {fails or 0}; DP ranks with decode batches {sorted(ranks)}")
    return 0 if ok == n and not fails else 1


if __name__ == "__main__":
    sys.exit(main())
