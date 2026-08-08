#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Reproduce the reported 448/0 kv-aware split against the real policy.

Drives KvEventAwarePolicy with the traffic shape from the field report:
two symmetric aggregated workers, multi-turn agent sessions, per-session
causal pacing (a turn finishes before the next turn of that session is
issued). Counts picks per worker, the way engine-side
prefix_cache_queries_total deltas would.

Only the KvEventClient and the tokenizer are stubbed: the client is
replaced by a model of what the engines' caches would actually hold
(blocks land on the worker that served the request), and the hasher
returns token ids directly so no model download is needed. The cost
function, the refcount bookkeeping and the lifecycle hooks are the real
ones.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo, WorkerStatus
from infera.router.kv_event.hasher import hash_request
from infera.router.policy.kv_event_aware import KvEventAwarePolicy

BLOCK_SIZE = 16


class SimulatedFleet:
    """Stands in for KvEventClient: each worker's view holds the blocks of
    the requests that worker actually served (unbounded, i.e. best case for
    locality -- a real engine evicts, which can only spread load further)."""

    def __init__(self, worker_ids: list[str]) -> None:
        self._views: dict[str, set[int]] = {w: set() for w in worker_ids}

    def cache_view(self, worker_id: str, dp_rank: int | None = None) -> set[int]:
        return self._views[worker_id]

    def store(self, worker_id: str, blocks: list[int]) -> None:
        self._views[worker_id].update(blocks)

    def on_worker_added(self, w: WorkerInfo) -> None:
        pass

    def on_worker_removed(self, worker_id: str) -> None:
        pass

    async def aclose(self) -> None:
        pass


class TokenHasher:
    """Hashes the pre-tokenized ids the trace generator produces, using the
    router's own chaining -- skips the tokenizer, keeps the hash chain."""

    def hash_for(self, body: dict, *, block_size: int, engine=None) -> list[int]:
        return hash_request(body["_tokens"], block_size)


def _worker(worker_id: str) -> WorkerInfo:
    return WorkerInfo(
        worker_id=worker_id,
        url=f"http://{worker_id}",
        model_name="moonshotai/Kimi-K3",
        engine=EngineType.VLLM,
        status=WorkerStatus.ACTIVE,
        disagg_mode=DisaggMode.MIXED,
        kv_events_endpoint=f"tcp://{worker_id}:5557",
        kv_block_size=BLOCK_SIZE,
    )


def make_trace(n_requests: int, seed: int = 0) -> list[dict]:
    """Multi-turn agent sessions: each turn re-sends the conversation so far
    plus new content, so turn N shares a long prefix with turn N-1.

    Shaped after the Mooncake toolagent trace: a shared system prompt, then
    per-session divergence, then per-turn growth.
    """
    rng = random.Random(seed)
    system = [rng.randrange(1000, 2000) for _ in range(3 * BLOCK_SIZE)]

    reqs: list[dict] = []
    session = 0
    while len(reqs) < n_requests:
        session += 1
        tokens = list(system) + [rng.randrange(10_000, 90_000) for _ in range(4 * BLOCK_SIZE)]
        for turn in range(rng.randint(3, 8)):
            tokens = tokens + [rng.randrange(10_000, 90_000) for _ in range(3 * BLOCK_SIZE)]
            reqs.append(
                {
                    "model": "moonshotai/Kimi-K3",
                    "_tokens": list(tokens),
                    "_session": session,
                    "_turn": turn,
                }
            )
            if len(reqs) >= n_requests:
                break
    return reqs[:n_requests]


def run(policy: KvEventAwarePolicy, fleet: SimulatedFleet, workers, trace) -> Counter:
    """Serial pacing: each request is picked, started and finished before the
    next -- the low-concurrency case the report describes."""
    picks: Counter = Counter()
    for req in trace:
        target, blocks = policy.pick(workers, req)
        picks[target.route_key] += 1
        policy.on_request_started(target.route_key, blocks)
        # The engine serves it and caches the blocks.
        fleet.store(target.worker.worker_id, blocks)
        policy.on_request_finished(target.route_key, blocks)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=448)
    ap.add_argument(
        "--kv-overlap-weight",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 20.0],
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    workers = [_worker("worker-A"), _worker("worker-B")]
    trace = make_trace(args.requests, seed=args.seed)

    print(f"trace: {len(trace)} requests, {len({r['_session'] for r in trace})} sessions")
    print(f"       block_size={BLOCK_SIZE}, serial pacing, 2 symmetric workers\n")
    print(f"{'overlap_weight':>15} | {'worker-A':>9} | {'worker-B':>9} | split")
    print(f"{'-' * 15}-+-{'-' * 9}-+-{'-' * 9}-+------")

    worst = 0.0
    for w in args.kv_overlap_weight:
        fleet = SimulatedFleet([x.worker_id for x in workers])
        policy = KvEventAwarePolicy(fleet, TokenHasher(), overlap_weight=w)
        picks = run(policy, fleet, workers, trace)
        a, b = picks.get("worker-A", 0), picks.get("worker-B", 0)
        share = max(a, b) / max(1, a + b)
        worst = max(worst, share)
        print(f"{w:>15g} | {a:>9} | {b:>9} | {share:>5.1%}")

    print()
    if worst > 0.99:
        print("REPRODUCED: at least one weight pins ~all traffic on one worker.")
        return 1
    print("not reproduced at these weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
