#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Replay the Mooncake FAST'25 conversation trace against the real routing policy.

    https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/traces/
        conversation_trace.jsonl

The trace gives block hashes, not text: each record is

    {"timestamp": ms, "input_length": n, "output_length": n,
     "hash_ids": [0, 1, 2, ...]}

so it drives the policy directly, with no tokenizer in the loop. That is the
point -- it isolates the routing decision from every upstream thing that can
make a prompt hash to nothing (undersized prompts, a tokenizer that failed to
load), which are separate faults with the same symptom.

Shape of this trace, measured:

    12031 requests over 58.9 min (3.4 req/s), median input 6909 tokens
    182790 distinct hash ids, ~496 tokens each
    hash id 0 appears in ALL 12031 requests -- one shared system prefix
    100% of requests share some prefix with an earlier one, 33% share >= 2 blocks

The universal prefix is what makes it a routing test rather than a cache test:
every worker can claim a hit on block 0, so a policy that weighs only locality
has a reason to send everything to whichever worker warmed up first.

    python3 bench/_kv_aware_mooncake_trace.py --trace conversation_trace.jsonl \
        --workers 2 --requests 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo, WorkerStatus
from infera.router.policy.kv_event_aware import KvEventAwarePolicy


class TraceFleet:
    """Stands in for KvEventClient. A worker's view holds the blocks of the
    requests it served, unbounded -- the best case for locality, since a real
    engine evicts and eviction can only spread load further."""

    def __init__(self, ids: list[str]) -> None:
        self._views: dict[str, set[int]] = {w: set() for w in ids}

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


class TraceHasher:
    """Returns the trace's own hash ids. The trace already chains them (a
    shared prefix is a shared id sequence), so no rehashing is needed or
    wanted -- rehashing would destroy exactly the structure under test."""

    def hash_for(self, body: dict, *, block_size: int, engine=None) -> list[int]:
        return list(body["hash_ids"])


def _worker(wid: str) -> WorkerInfo:
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="trace",
        engine=EngineType.VLLM,
        status=WorkerStatus.ACTIVE,
        disagg_mode=DisaggMode.MIXED,
        kv_events_endpoint=f"tcp://{wid}:5557",
        kv_block_size=512,
    )


def replay(rows, workers, weight, concurrency):
    """Replay with per-request pacing bounded by `concurrency`.

    concurrency=1 is the serial case the field report described: a request
    completes before the next is picked, so nothing is ever in flight at a
    decision and the in-flight half of the load term reads 0 for everyone.
    """
    fleet = TraceFleet([w.worker_id for w in workers])
    pol = KvEventAwarePolicy(fleet, TraceHasher(), overlap_weight=weight)
    picks: Counter = Counter()
    inflight: list[tuple] = []
    hits = total = 0

    for r in rows:
        while len(inflight) >= concurrency:
            rk, bl, wid = inflight.pop(0)
            fleet.store(wid, bl)
            pol.on_request_finished(rk, bl)

        body = {"model": "trace", "hash_ids": r["hash_ids"]}
        view_before = {w.worker_id: set(fleet.cache_view(w.worker_id)) for w in workers}
        target, blocks = pol.pick(workers, body)

        # Score the hit the chosen worker actually got, the way the engine would.
        v = view_before[target.worker.worker_id]
        n = 0
        for b in r["hash_ids"]:
            if b not in v:
                break
            n += 1
        hits += n
        total += len(r["hash_ids"])

        picks[target.route_key] += 1
        pol.on_request_started(target.route_key, blocks)
        inflight.append((target.route_key, blocks, target.worker.worker_id))

    while inflight:
        rk, bl, wid = inflight.pop(0)
        fleet.store(wid, bl)
        pol.on_request_finished(rk, bl)

    return picks, (hits / total if total else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="/tmp/conv_trace.jsonl")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--requests", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 64])
    ap.add_argument("--weights", type=float, nargs="+", default=[0.01, 1.0, 20.0])
    args = ap.parse_args()

    rows = []
    with open(args.trace) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.requests:
                break

    workers = [_worker(f"w{i}") for i in range(args.workers)]
    ideal = 1.0 / args.workers
    print(f"Mooncake conversation trace: {len(rows)} requests, {args.workers} workers")
    print(f"ideal share {ideal:.1%} each\n")
    print(f"{'C':>4} {'weight':>7} | {'worst share':>11} | {'hit rate':>8} | split")
    print(f"{'-' * 4}-{'-' * 7}-+-{'-' * 11}-+-{'-' * 8}-+------")

    worst_overall = 0.0
    for c in args.concurrency:
        for w in args.weights:
            picks, hr = replay(rows, workers, w, c)
            counts = [picks.get(f"w{i}", 0) for i in range(args.workers)]
            share = max(counts) / sum(counts)
            worst_overall = max(worst_overall, share)
            shown = "/".join(str(x) for x in counts[:4]) + ("/..." if len(counts) > 4 else "")
            print(f"{c:>4} {w:>7g} | {share:>10.1%} | {hr:>7.1%} | {shown}")

    print()
    if worst_overall > 0.95:
        print("PINNED: at least one configuration sends nearly everything to one worker.")
        return 1
    print(f"SPREAD: worst share across all configurations {worst_overall:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
