###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Replay a deterministic chat corpus against an OpenAI-compatible
endpoint; measure per-turn TTFT + wall-time and dump a JSON trace.

The bench is deliberately minimal because the *engine* is what we're
measuring — every cycle spent in the bench layer is noise. We:

1. Open one shared `httpx.AsyncClient` (HTTP keep-alive).
2. Fan out N parallel replay tasks (one per corpus session).
3. Each task replays its session's turns in order. Within a session
   turns are sequential — that's the only way the growing-prefix
   pattern works.
4. Streaming responses so first-token timing is observable; we
   ignore the body after TTFT (set max_tokens=1 to keep payload
   trivial).

Run order discipline (important for cache experiments): the engine
sees turn 0 of every session before any turn 1, because all tasks
launch concurrently and Python scheduling tends to round-robin.
That keeps L1 pressure even — no single session dominates the cache.

Usage:

    python -m bench.kvcache.deterministic_l3.replay \\
        --corpus /tmp/corpus.json \\
        --server http://localhost:30000 \\
        --model MiniMax-M2.5 \\
        --concurrency 8 \\
        --out /tmp/replay-phase1.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


async def send_turn(
    http: httpx.AsyncClient,
    server: str,
    model: str,
    messages: list[dict],
    timeout_s: float = 600.0,
) -> tuple[float | None, float, bool, str | None]:
    """Send one streaming chat completion. Returns
    `(ttft_ms, wall_ms, success, error_string)`.

    TTFT is measured from request start to the first non-empty
    `data:` line. wall_ms is end-to-end including drain (negligible
    with max_tokens=1).
    """
    url = f"{server.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        # max_tokens=2 instead of 1 to work around a vLLM v1 scheduler
        # bug: after generating the only output token the request gets
        # re-scheduled once before vLLM notices it's done, and
        # `num_new_tokens = num_tokens - num_computed_tokens` becomes 0,
        # tripping `assert num_new_tokens > 0` at scheduler.py:651.
        # Generating 2 tokens leaves a 1-token cushion that lets the
        # decode loop finish cleanly. Cost: ~1 ms per request.
        "max_tokens": 2,
        "temperature": 0.0,
        "stream": True,
    }
    start = time.perf_counter()
    ttft_ms: float | None = None
    try:
        async with http.stream(
            "POST", url, json=body, timeout=httpx.Timeout(timeout_s, connect=10.0)
        ) as resp:
            if resp.status_code != 200:
                err = await resp.aread()
                wall = (time.perf_counter() - start) * 1000
                return None, wall, False, f"HTTP {resp.status_code}: {err[:200]!r}"
            async for line in resp.aiter_lines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("data: "):
                    payload = stripped[len("data: ") :]
                    if payload == "[DONE]":
                        continue
                    # First non-empty data line is first-token signal.
                    # We don't decode the JSON — the timing is what we
                    # care about; the body is "OK" or one token.
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000
            wall = (time.perf_counter() - start) * 1000
            return ttft_ms, wall, True, None
    except Exception as exc:
        wall = (time.perf_counter() - start) * 1000
        return None, wall, False, repr(exc)


async def replay_session(
    http: httpx.AsyncClient,
    server: str,
    model: str,
    sess: dict,
    sem: asyncio.Semaphore,
    results: list[dict],
) -> None:
    """Replay one session — turns in order, one in flight at a time.
    The concurrency cap `sem` bounds the total number of in-flight
    requests across all sessions."""
    session_id = sess["session_id"]
    msgs = sess["messages"]
    # Layout is user/assistant/user/assistant/.../user — final user
    # turn has no assistant reply (that's what the engine generates).
    # We send messages[0:1], [0:3], [0:5], ... corresponding to a
    # request whose last message is each user turn in sequence.
    n_user_turns = (len(msgs) + 1) // 2
    for turn_idx in range(n_user_turns):
        prefix = msgs[: turn_idx * 2 + 1]
        async with sem:
            sent_at = time.perf_counter()
            ttft_ms, wall_ms, ok, err = await send_turn(http, server, model, prefix)
        results.append(
            {
                "session_id": session_id,
                "turn_idx": turn_idx,
                "n_messages_in_prefix": len(prefix),
                "sent_at": sent_at,
                "ttft_ms": ttft_ms,
                "wall_ms": wall_ms,
                "ok": ok,
                "err": err,
            }
        )


async def main_async(args: argparse.Namespace) -> None:
    corpus = json.loads(args.corpus.read_text())
    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    # httpx keep-alive: bound the pool so a misconfigured concurrency
    # value can't silently serialize on connection contention.
    limits = httpx.Limits(
        max_connections=args.concurrency * 2,
        max_keepalive_connections=args.concurrency * 2,
    )
    wall_start = time.perf_counter()
    async with httpx.AsyncClient(limits=limits, timeout=None) as http:
        tasks = [
            asyncio.create_task(
                replay_session(http, args.server, args.model, sess, sem, results),
                name=f"sess-{sess['session_id']}",
            )
            for sess in corpus["sessions"]
        ]
        await asyncio.gather(*tasks)
    wall_total_s = time.perf_counter() - wall_start

    args.out.write_text(json.dumps({"results": results, "wall_total_s": wall_total_s}, indent=2))

    ok_results = [r for r in results if r["ok"] and r["ttft_ms"] is not None]
    ttfts = sorted(r["ttft_ms"] for r in ok_results)
    print(f"requests sent: {len(results)}, ok: {len(ok_results)}, wall {wall_total_s:.1f}s")
    if ttfts:
        p50 = ttfts[len(ttfts) // 2]
        p90 = ttfts[int(len(ttfts) * 0.9)]
        p99 = ttfts[min(int(len(ttfts) * 0.99), len(ttfts) - 1)]
        print(f"TTFT  p50={p50:.0f}ms  p90={p90:.0f}ms  p99={p99:.0f}ms")
    if any(not r["ok"] for r in results):
        errs: dict[str, int] = {}
        for r in results:
            if not r["ok"]:
                key = (r["err"] or "?").splitlines()[0][:80]
                errs[key] = errs.get(key, 0) + 1
        for k, v in sorted(errs.items(), key=lambda x: -x[1]):
            print(f"  ERR ×{v}: {k}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--server", default="http://localhost:30000")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
