#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""End-to-end kv-aware split test against a live two-worker Kimi-K3 fleet.

Replays multi-turn agent sessions through the router with per-session causal
pacing (a turn completes before the next turn of that session is issued) and
reports the traffic split measured the way the field report measured it: from
each engine's own ``vllm:prefix_cache_queries_total`` delta, not from router
metrics.

Sessions are built from a corpus of real K3-tokenized source files, so each
turn re-sends the conversation so far plus new content and shares a long
prefix with the previous turn.

    python3 bench/_kv_aware_k3_e2e_client.py \
        --router http://localhost:8000 \
        --workers http://node:30000 http://node:30001 \
        --requests 448 --concurrency 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from collections import defaultdict

import httpx

_QUERIES = "vllm:prefix_cache_queries_total"
_HITS = "vllm:prefix_cache_hits_total"


def _scrape(text: str, metric: str) -> float:
    """Sum every series of ``metric`` in a Prometheus exposition payload."""
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(metric):
            continue
        # metric{labels} value   |   metric value
        m = re.match(rf"{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", line.strip())
        if m:
            total += float(m.group(1))
    return total


async def counters(client: httpx.AsyncClient, url: str) -> tuple[float, float]:
    r = await client.get(f"{url}/metrics", timeout=30.0)
    r.raise_for_status()
    return _scrape(r.text, _QUERIES), _scrape(r.text, _HITS)


def load_docs(root: str, min_chars: int = 4000, limit: int = 200) -> list[str]:
    """Real source files, big enough that a prompt spans several router blocks."""
    import pathlib

    out: list[str] = []
    for f in sorted(pathlib.Path(root).rglob("*.py")):
        try:
            t = f.read_text(errors="ignore")
        except OSError:
            continue
        if len(t) >= min_chars:
            out.append(t[:12000])
        if len(out) >= limit:
            break
    return out


def build_sessions(docs: list[str], n_requests: int, seed: int) -> list[dict]:
    """Multi-turn agent sessions over real source files, as chat requests.

    A shared system prompt, then per-session divergence (one source file), then
    per-turn growth -- the shape the Mooncake toolagent trace has and the shape
    that makes turn N share a long prefix with turn N-1.

    Prompts must clear the router's index block size (768 tokens here) or they
    hash to zero blocks and the cache term has nothing to work with: the first
    run of this produced request_blocks=0 on every pick and a 0% hit rate, which
    looks like a routing result but is really an undersized prompt.
    """
    rng = random.Random(seed)
    # ~3.5k tokens of shared preamble, so even turn 0 spans several blocks and
    # every session in the run shares a real prefix.
    system = (
        "You are a senior systems engineer reviewing AMD ROCm inference code.\n"
        "Answer precisely, cite the code you are given, and prefer concrete "
        "detail over generalities.\n"
    ) + "".join(f"Review guideline {i}: check correctness, then performance.\n" for i in range(300))
    rng.shuffle(docs)

    reqs: list[dict] = []
    sid = 0
    while len(reqs) < n_requests and docs:
        doc = docs[sid % len(docs)]
        sid += 1
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Here is a file to review:\n\n{doc}"},
        ]
        for turn in range(rng.randint(3, 8)):
            messages = messages + [
                {"role": "assistant", "content": f"Reviewed section {turn}. " * 60},
                {"role": "user", "content": f"Now explain section {turn + 1} in detail. " * 60},
            ]
            reqs.append(
                {
                    "session": sid,
                    "turn": turn,
                    "payload": {
                        "model": "kimi-k3",
                        "messages": list(messages),
                        "max_tokens": 8,
                        "temperature": 0.0,
                        "stream": False,
                    },
                }
            )
            if len(reqs) >= n_requests:
                break
    return reqs[:n_requests]


async def run(args) -> int:
    docs = load_docs(args.corpus_dir)
    print(f"corpus: {len(docs)} source files from {args.corpus_dir}")
    reqs = build_sessions(docs, args.requests, args.seed)
    n_sessions = len({r["session"] for r in reqs})
    print(f"trace: {len(reqs)} requests over {n_sessions} sessions")
    print(f"       policy under test: whatever the router was launched with")
    print(f"       concurrency={args.concurrency} (1 = the reported serial case)\n")

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        before = [await counters(client, w) for w in args.workers]

        # Per-session causal pacing: turns of one session are strictly ordered,
        # and `concurrency` bounds how many sessions are in flight at once.
        by_session: dict[int, list[dict]] = defaultdict(list)
        for r in reqs:
            by_session[r["session"]].append(r)
        sem = asyncio.Semaphore(args.concurrency)
        failures = 0
        lat: list[float] = []

        done = 0

        async def one_session(turns: list[dict]) -> None:
            nonlocal failures, done
            async with sem:
                for r in turns:
                    t0 = time.perf_counter()
                    try:
                        resp = await client.post(
                            f"{args.router}/v1/chat/completions",
                            json=r["payload"],
                            timeout=args.request_timeout,
                        )
                        if resp.status_code != 200:
                            failures += 1
                            if failures <= 3:
                                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                        else:
                            lat.append(time.perf_counter() - t0)
                    except Exception as exc:  # noqa: BLE001 - report and continue
                        failures += 1
                        if failures <= 3:
                            print(f"  error: {exc}", flush=True)
                    finally:
                        done += 1
                        if done % 25 == 0:
                            print(f"  {done}/{len(reqs)} ...", flush=True)

        t_start = time.perf_counter()
        await asyncio.gather(*(one_session(t) for t in by_session.values()))
        elapsed = time.perf_counter() - t_start

        after = [await counters(client, w) for w in args.workers]

    print(f"completed {len(lat)}/{len(reqs)} in {elapsed:.1f}s ({failures} failures)\n")

    deltas = [(a[0] - b[0], a[1] - b[1]) for b, a in zip(before, after)]
    total_q = sum(d[0] for d in deltas)
    print(f"{'worker':>34} | {'queries':>9} | {'share':>6} | {'hit rate':>8}")
    print("-" * 34 + "-+-" + "-" * 9 + "-+-" + "-" * 6 + "-+-" + "-" * 8)
    for w, (dq, dh) in zip(args.workers, deltas):
        share = dq / total_q if total_q else 0.0
        hr = dh / dq if dq else 0.0
        print(f"{w:>34} | {dq:>9.0f} | {share:>5.1%} | {hr:>7.1%}")

    if not total_q:
        print("\nno prefix-cache queries recorded -- are these the right worker URLs?")
        return 2
    worst = max(d[0] for d in deltas) / total_q
    overall_hr = sum(d[1] for d in deltas) / total_q
    print(f"\noverall block hit rate: {overall_hr:.1%}")
    print(f"worst-worker share:     {worst:.1%}  (ideal {1 / len(args.workers):.1%})")
    if worst > 0.99:
        print("\nPINNED: one worker took essentially all traffic.")
        return 1
    print("\nSPREAD: traffic reached every worker.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True, help="router base URL")
    ap.add_argument("--workers", nargs="+", required=True, help="engine base URLs (/metrics)")
    ap.add_argument("--corpus-dir", default="infera")
    ap.add_argument("--requests", type=int, default=448)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--request-timeout", type=float, default=120.0)
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
