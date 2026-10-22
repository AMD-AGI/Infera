#!/usr/bin/env python3
"""Cancellation repro for a PD stack: send N concurrent long requests, cancel
half of them mid-flight (connection close before the first token), then require
the decode /health (and a small probe request) to succeed within a deadline.

Exit code 0 = healthy after cancels, 1 = unhealthy / hung, 2 = setup error.

Example:
  python3 pd_cancel_repro.yihou.py --url http://127.0.0.1:8000 \
      --health-url http://127.0.0.1:30001/health --model GLM-5.2 \
      --num-requests 40 --input-words 24000 --cancel-after 1.5
"""

import argparse
import asyncio
import json
import random
import sys
import time

import aiohttp

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu river stone cloud ember maple quartz harbor lantern"
).split()


def make_prompt(rng: random.Random, n_words: int) -> str:
    # Random words defeat the prefix cache so every request does a full prefill
    # and a full KV transfer.
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def body(args, prompt: str, max_tokens: int) -> dict:
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }


async def one_request(session, args, idx, prompt, cancel, log):
    t0 = time.monotonic()
    first_token_at = None
    status = None
    try:
        async with session.post(
            args.url + args.path, json=body(args, prompt, args.max_tokens)
        ) as resp:
            status = resp.status
            async for raw in resp.content:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if first_token_at is None:
                    first_token_at = time.monotonic() - t0
        log.append(
            {"idx": idx, "cancel": cancel, "status": status, "ttft": first_token_at,
             "outcome": "done", "elapsed": time.monotonic() - t0}
        )
    except asyncio.CancelledError:
        log.append(
            {"idx": idx, "cancel": cancel, "status": status, "ttft": first_token_at,
             "outcome": "cancelled", "elapsed": time.monotonic() - t0}
        )
        raise
    except Exception as e:  # report, do not abort the repro
        log.append(
            {"idx": idx, "cancel": cancel, "status": status, "ttft": first_token_at,
             "outcome": f"error: {type(e).__name__}: {e}",
             "elapsed": time.monotonic() - t0}
        )


async def health_ok(session, url, timeout) -> bool:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            return r.status == 200
    except Exception:
        return False


async def probe_ok(session, args, timeout) -> bool:
    try:
        payload = body(args, "Say OK.", 8)
        payload["stream"] = False
        async with session.post(
            args.url + args.path,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as r:
            return r.status == 200
    except Exception:
        return False


async def main_async(args) -> int:
    rng = random.Random(args.seed)
    prompts = [make_prompt(rng, args.input_words) for _ in range(args.num_requests)]
    cancel_set = set(rng.sample(range(args.num_requests), args.num_requests // 2))
    log = []
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        if not await health_ok(session, args.health_url, 10):
            print(f"[setup] {args.health_url} not healthy before the run", flush=True)
            return 2
        tasks = [
            asyncio.create_task(
                one_request(session, args, i, prompts[i], i in cancel_set, log)
            )
            for i in range(args.num_requests)
        ]
        await asyncio.sleep(args.cancel_after)
        cancelled = 0
        for i, t in enumerate(tasks):
            if i in cancel_set and not t.done():
                t.cancel()
                cancelled += 1
        print(
            f"[cancel] t={args.cancel_after}s cancelled {cancelled}/{len(cancel_set)} "
            f"selected requests (the rest already finished)",
            flush=True,
        )
        if args.cancel_all_after > 0:
            await asyncio.sleep(args.cancel_all_after)
            for t in tasks:
                if not t.done():
                    t.cancel()
        # Keep the survivors running while health is checked: a hang shows up
        # as both /health and the probe timing out.
        t_cancel = time.monotonic()
        healthy_at = None
        while time.monotonic() - t_cancel < args.deadline:
            await asyncio.sleep(1.0)
            if await health_ok(session, args.health_url, 5) and await probe_ok(
                session, args, args.probe_timeout
            ):
                healthy_at = time.monotonic() - t_cancel
                break
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    with open(args.log, "w") as f:
        json.dump(sorted(log, key=lambda r: r["idx"]), f, indent=1)
    outcomes = {}
    for r in log:
        key = ("cancel-set " if r["cancel"] else "keep-set ") + r["outcome"].split(":")[0]
        outcomes[key] = outcomes.get(key, 0) + 1
    print(f"[outcomes] {outcomes}  (per-request log: {args.log})", flush=True)
    if healthy_at is None:
        print(f"[FAIL] not healthy within {args.deadline}s after cancel", flush=True)
        return 1
    print(f"[PASS] healthy + probe OK {healthy_at:.1f}s after cancel", flush=True)
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", required=True, help="router base URL")
    p.add_argument("--path", default="/v1/chat/completions")
    p.add_argument("--health-url", required=True, help="decode server /health URL")
    p.add_argument("--model", default="default")
    p.add_argument("--num-requests", type=int, default=40)
    p.add_argument("--input-words", type=int, default=24000,
                   help="random words per prompt (roughly 1-2 tokens each)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--cancel-after", type=float, default=1.5,
                   help="seconds after launch to cancel half; pick it inside the "
                        "prefill+transfer window (before the first token)")
    p.add_argument("--cancel-all-after", type=float, default=0.0,
                   help="if > 0, also cancel every survivor this many seconds "
                        "after the first cancel (AgentX end-of-run shape)")
    p.add_argument("--deadline", type=float, default=60.0)
    p.add_argument("--probe-timeout", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", default="pd_cancel_repro.log.json")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
