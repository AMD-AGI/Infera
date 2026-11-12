###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Request-level simulation of a non-autoregressive server.

``simulate_static`` runs one batch of identical requests in lockstep -- what
the reference samplers do, and the number a paper's latency table reports.

``simulate_serving`` is the serving counterpart of InferaSim's autoregressive
DES, at refinement-step granularity. Requests arrive by a Poisson process and
join the running batch at the next step boundary while the replica has room;
every step advances each running request by one of its own scheduled steps, so
a batch mixes requests at different blocks and cache phases and the step is
priced at that ragged composition. A request's first-block time is its time to
first token for a server that streams finished blocks.
"""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field
from typing import Any

from .memory import max_batch, memory_report
from .step_model import DLLMStepModel
from .workload import DLLMWorkload


@dataclass
class StaticResult:
    batch: int
    steps: int
    latency_ms: float
    first_block_ms: float
    tokens_per_s: float
    tokens_per_s_per_gpu: float
    mean_step_ms: float
    breakdown: dict[str, Any]
    memory: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _seqs(shape, wl: DLLMWorkload):
    return [shape] * wl.seqs_per_request


def simulate_static(model: DLLMStepModel, wl: DLLMWorkload, batch: int) -> StaticResult:
    if batch < 1:
        raise ValueError("[dllm-sim] batch must be >= 1")
    sched = wl.schedule()
    t = 0.0
    first_block = None
    worst = None
    for st in sched:
        b = model.step(_seqs(st.shape, wl) * batch)
        t += b.total_ms
        if worst is None or b.total_ms > worst.total_ms:
            worst = b
        if st.finishes_block == 0 and first_block is None:
            first_block = t
    tokens = batch * wl.gen_len
    tps = tokens / (t / 1e3)
    mem = memory_report(model.spec, model.par, model.hw, wl, batch)
    return StaticResult(
        batch=batch,
        steps=len(sched),
        latency_ms=t,
        first_block_ms=first_block if first_block is not None else t,
        tokens_per_s=tps,
        tokens_per_s_per_gpu=tps / model.par.gpus_per_replica,
        mean_step_ms=t / len(sched),
        breakdown=worst.as_dict() if worst else {},
        memory=mem.as_dict(),
    )


@dataclass
class _Request:
    rid: int
    arrival_ms: float
    step: int = 0
    start_ms: float = -1.0
    first_block_ms: float = -1.0
    done_ms: float = -1.0


@dataclass
class ServingResult:
    rate_per_s: float
    requests: int
    completed: int
    max_batch: int
    duration_ms: float
    tokens_per_s: float
    tokens_per_s_per_gpu: float
    mean_batch: float
    saturated: bool
    latency_ms: dict[str, float] = field(default_factory=dict)
    first_block_ms: dict[str, float] = field(default_factory=dict)
    queue_ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _pct(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    v = sorted(values)

    def q(p: float) -> float:
        i = min(len(v) - 1, max(0, round(p * (len(v) - 1))))
        return v[i]

    return {"mean": sum(v) / len(v), "p50": q(0.50), "p90": q(0.90), "p99": q(0.99)}


def simulate_serving(
    model: DLLMStepModel,
    wl: DLLMWorkload,
    *,
    rate_per_s: float,
    num_requests: int = 200,
    batch_limit: int = 0,
    seed: int = 0,
    warmup_fraction: float = 0.1,
) -> ServingResult:
    """One replica under Poisson load; ``batch_limit`` 0 = what memory allows."""
    if rate_per_s <= 0 or num_requests < 1:
        raise ValueError("[dllm-sim] rate_per_s and num_requests must be positive")
    cap = max_batch(model.spec, model.par, model.hw, wl)
    if cap < 1:
        raise ValueError("[dllm-sim] one request does not fit on the replica")
    limit = min(cap, batch_limit) if batch_limit > 0 else cap
    sched = wl.schedule()
    n_steps = len(sched)

    rng = random.Random(seed)
    t_arr = 0.0
    pending: collections.deque[_Request] = collections.deque()
    for i in range(num_requests):
        t_arr += rng.expovariate(rate_per_s) * 1e3
        pending.append(_Request(i, t_arr))

    t = 0.0
    running: list[_Request] = []
    done: list[_Request] = []
    batch_time = 0.0
    while pending or running:
        while pending and pending[0].arrival_ms <= t and len(running) < limit:
            r = pending.popleft()
            r.start_ms = t
            running.append(r)
        if not running:
            t = pending[0].arrival_ms
            continue
        seqs = []
        for r in running:
            seqs.extend(_seqs(sched[r.step].shape, wl))
        dt = model.step(seqs).total_ms
        t += dt
        batch_time += dt * len(running)
        still = []
        for r in running:
            if sched[r.step].finishes_block == 0:
                r.first_block_ms = t
            r.step += 1
            if r.step >= n_steps:
                r.done_ms = t
                done.append(r)
            else:
                still.append(r)
        running = still

    skip = int(len(done) * warmup_fraction)
    kept = sorted(done, key=lambda r: r.arrival_ms)[skip:]
    duration = max(r.done_ms for r in done) - min(r.arrival_ms for r in done)
    tps = len(done) * wl.gen_len / (duration / 1e3) if duration > 0 else 0.0
    lat = [r.done_ms - r.arrival_ms for r in kept]
    # Saturated: the queue keeps growing, so the last arrivals wait longer for a
    # slot than a request takes to be served once it has one.
    late = kept[-max(1, len(kept) // 4) :]
    late_queue = sum(r.start_ms - r.arrival_ms for r in late) / len(late)
    service = _pct([r.done_ms - r.start_ms for r in kept]).get("p50", 0.0)
    saturated = late_queue > service
    return ServingResult(
        rate_per_s=rate_per_s,
        requests=num_requests,
        completed=len(done),
        max_batch=limit,
        duration_ms=duration,
        tokens_per_s=tps,
        tokens_per_s_per_gpu=tps / model.par.gpus_per_replica,
        mean_batch=batch_time / t if t > 0 else 0.0,
        saturated=saturated,
        latency_ms=_pct(lat),
        first_block_ms=_pct([r.first_block_ms - r.arrival_ms for r in kept]),
        queue_ms=_pct([r.start_ms - r.arrival_ms for r in kept]),
    )
