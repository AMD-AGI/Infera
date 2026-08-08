#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Why is kv-aware routing behaving like round-robin?

Answers that without reading router logs, by checking the three things that
make the cache term drop out of the pick decision. When it drops out, every
--kv-overlap-weight gives an identical result, because the cost function

    cost(w) = overlap_weight * (request_blocks - hits(w)) + load(w)

only lets the weight matter when workers DIFFER in hits. If request_blocks is 0,
or every worker reports the same hits, `overlap_weight * <same number>` is a
constant across candidates and cancels inside min() -- so 1.0 and 0.01 land on
exactly the same worker, and the policy reduces to least-loaded.

    python3 bench/_kv_aware_diagnose.py \
        --router http://host:8000 --workers http://a:30000 http://b:30000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

_QUERIES = "vllm:prefix_cache_queries_total"
_HITS = "vllm:prefix_cache_hits_total"


def _get(url: str, timeout: float = 20.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
        return r.read().decode("utf-8", "replace")


def _post(url: str, payload: dict, timeout: float = 300.0) -> tuple[int, str]:
    req = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _scrape(text: str, metric: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(metric):
            continue
        m = re.match(rf"{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", line.strip())
        if m:
            total += float(m.group(1))
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True)
    ap.add_argument("--workers", nargs="+", required=True)
    ap.add_argument("--model", default="kimi-k3")
    args = ap.parse_args()

    problems: list[str] = []

    # 1. Do the engines publish kv events at all? Without them the router's
    #    cache view stays empty and every worker reports hits=0 forever.
    print("[1] engine kv-event publishing")
    for w in args.workers:
        try:
            q = _scrape(_get(f"{w}/metrics"), _QUERIES)
            print(f"    {w}: prefix_cache_queries_total={q:.0f}")
        except Exception as exc:  # noqa: BLE001
            print(f"    {w}: UNREACHABLE ({exc})")
            problems.append(f"{w} /metrics unreachable -- wrong URL, or engine down")

    # 2. Probe with a deliberately LONG prompt and read the engines' own
    #    counters. The router's index block size defaults to 768 tokens; a
    #    prompt shorter than that hashes to zero blocks and the cache term
    #    never participates, which looks exactly like "kv-aware is not working".
    print("\n[2] block-size probe (long prompt, ~4k tokens)")
    before = []
    for w in args.workers:
        try:
            t = _get(f"{w}/metrics")
            before.append((_scrape(t, _QUERIES), _scrape(t, _HITS)))
        except Exception:  # noqa: BLE001
            before.append((0.0, 0.0))

    preamble = "Review guideline: check correctness, then performance.\n" * 300
    ok = 0
    for i in range(6):
        code, body = _post(
            f"{args.router}/v1/chat/completions",
            {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": preamble},
                    {"role": "user", "content": f"Explain topic {i % 2} in detail. " * 80},
                ],
                "max_tokens": 8,
                "temperature": 0.0,
            },
        )
        if code == 200:
            ok += 1
        elif i == 0:
            print(f"    request failed: HTTP {code}: {body[:200]}")
            problems.append(f"router returned HTTP {code} -- fix that before reading anything else")
    print(f"    {ok}/6 long requests succeeded")

    after = []
    for w in args.workers:
        try:
            t = _get(f"{w}/metrics")
            after.append((_scrape(t, _QUERIES), _scrape(t, _HITS)))
        except Exception:  # noqa: BLE001
            after.append((0.0, 0.0))

    print(f"\n{'worker':<34}{'queries':>10}{'hits':>10}{'hit rate':>10}")
    tq = th = 0.0
    for w, b, a in zip(args.workers, before, after):
        dq, dh = a[0] - b[0], a[1] - b[1]
        tq += dq
        th += dh
        rate = f"{dh / dq:.1%}" if dq else "n/a"
        print(f"{w:<34}{dq:>10.0f}{dh:>10.0f}{rate:>10}")

    print("\n" + "=" * 62)
    if tq == 0:
        print("VERDICT: the engines recorded no prefix-cache queries at all.")
        print("  The requests are not reaching these engines, or prefix caching")
        print("  is off. Check --enable-prefix-caching and that --workers point")
        print("  at the engines this router actually routes to.")
        problems.append("no prefix-cache queries recorded")
    elif th == 0:
        print("VERDICT: queries happen, but hits are ZERO on every worker.")
        print("  The router's hashes never match what the engines cached, so")
        print("  hits(w)=0 for all w and the weight cancels out -- which is")
        print("  exactly why 1.0 and 0.01 give identical results.")
        print("\n  Most likely, in order:")
        print("   * --kv-event-transport defaults to NATS on BOTH the engines")
        print("     and the router. With no broker the router silently never")
        print("     subscribes. Pass `--kv-event-transport zmq` to all of them.")
        print("   * --router-tokenizer-path does not match the served model, so")
        print("     the router tokenizes differently than the engine.")
        print("   * the router does not list both workers (check discovery).")
        problems.append("zero cache hits fleetwide")
    else:
        print(f"VERDICT: cache locality is working -- {th / tq:.1%} block hit rate.")
        print("  The cache term is live, so --kv-overlap-weight will change")
        print("  placement. If a SHORT-prompt test still showed no difference")
        print("  between weights, that test's prompts were under the router's")
        print("  index block size (default 768 tokens) and hashed to zero")
        print("  blocks -- size the prompts above it and re-measure.")

    if problems:
        print("\nissues found:")
        for p in problems:
            print(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
