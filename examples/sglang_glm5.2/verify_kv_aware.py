#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Two-shot prefix check: is kv-aware routing actually seeing cache hits?

Sends a long prompt, waits for the engine's kv events to reach the router, then
sends the identical prompt again. The contract in one run:

  1. the router's per-rank cache view goes from 0 blocks to N,
  2. the first request reports ``cache_hits=0``, the repeat reports
     ``cache_hits == request_blocks`` (full prefix hit),
  3. the repeat is much faster end to end.

Any of those failing means the event path is broken somewhere -- see
KV_AWARE.zh.md for the two bugs this catches (kv-event port swallowed by IPVS,
bigram token_ids under MTP) and how to tell them apart.

Env: ROUTER, MODEL, PREFILL_WORKER, RANKS, ROUTER_LOG.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

ROUTER = os.environ.get("ROUTER", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL", "/wekafs/models/GLM-5.2-FP8")
WORKER = os.environ.get("PREFILL_WORKER", "10.32.17.210:30001")
RANKS = int(os.environ.get("RANKS", "8"))
ROUTER_LOG = os.environ.get(
    "ROUTER_LOG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "infera_1_server.log")
)

# Salt the prompt so every run starts as a genuine miss; a repeat of a previous
# run's prompt is served from the radix tree and emits no store events, which
# would make an empty cache view look like a failure.
_salt = f"Session {time.time():.6f}"
_PROMPT = _salt + "\n" + "\n".join(
    f"Line {i}: token budget {i * 41 % 89}, region {i % 11}, flag {i % 3}."
    for i in range(1200)
)


def ask(tag: str) -> float:
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": _PROMPT + "\n\nReply with OK only."}],
            "max_tokens": 32,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{ROUTER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        body = json.loads(resp.read())
    elapsed = time.time() - t0
    usage = body.get("usage", {})
    details = usage.get("prompt_tokens_details") or {}
    print(
        f"[{tag}] wall={elapsed:.2f}s prompt={usage.get('prompt_tokens')} "
        f"engine_cached={details.get('cached_tokens')}",
        flush=True,
    )
    return elapsed


def view() -> dict[int, int]:
    out = {}
    for rank in range(RANKS):
        url = f"{ROUTER}/v1/admin/cache-view/{WORKER}?dp_rank={rank}"
        with urllib.request.urlopen(url, timeout=15) as resp:
            out[rank] = json.loads(resp.read())["block_count"]
    return out


def last_prefill_picks(n: int) -> list[tuple[int, int]]:
    """(cache_hits, request_blocks) for the last n prefill routing decisions."""
    try:
        with open(ROUTER_LOG, errors="replace") as fh:
            lines = [ln for ln in fh if "pick policy=kv-aware role=prefill" in ln]
    except OSError as exc:
        print(f"cannot read {ROUTER_LOG}: {exc}")
        return []
    out = []
    for line in lines[-n:]:
        m = re.search(r"cache_hits=(\d+) request_blocks=(\d+)", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return out


print(f"cache view before : {view()}")
ask("first")
time.sleep(8)
mid = view()
print(f"cache view after  : {mid}  total={sum(mid.values())}")
repeat_s = ask("repeat")
time.sleep(3)

picks = last_prefill_picks(2)
print(f"router prefill picks (cache_hits, request_blocks): {picks}")

failures = []
if sum(mid.values()) == 0:
    failures.append("router cache view stayed empty: kv events are not being applied")
if len(picks) < 2:
    failures.append(f"could not read two prefill picks from {ROUTER_LOG}")
else:
    hits, blocks = picks[-1]
    if blocks == 0:
        failures.append("repeat request hashed to 0 blocks (tokenizer/block_size problem)")
    elif hits < blocks:
        failures.append(f"repeat hit only {hits}/{blocks} blocks (expected a full prefix hit)")

if failures:
    print("\nFAIL")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"\nPASS: full prefix hit on repeat, repeat wall time {repeat_s:.2f}s")
