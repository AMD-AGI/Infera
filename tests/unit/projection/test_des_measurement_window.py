###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""What the replay reports over, and what it is allowed to reorder.

Three decisions about the *measurement window* rather than about physics, each
of which moved a validated result further than any cost-model change did.

1. A fixed-concurrency run is bounded by a clock, not by a request count. The
   distinction bites under a closed loop because a request budget divided
   among C lanes fixes how far each lane walks into its conversation -- deep
   at low concurrency, shallow at high -- and an agentic turn's prompt grows
   with its position, so a budget-bounded replay offers a prompt-length trend
   with concurrency that the hardware never had.

2. If the opening transient is dropped at all, it has to be dropped by
   *issue* order. Every client fires at once, so the first C requests queue
   against each other and carry the run's longest waits; because they waited,
   they are also the last to retire, so dropping a fraction of the earliest
   *completions* keeps every one of them. (Whether to drop it is a separate
   question, and for this harness the answer is no -- its lane-advance
   requests are one-token requests, so the profiled window still opens with
   every lane firing together and the burst is in the reported average.)

3. A scheduler can only reorder requests that are queued. Admitting by
   longest resident prefix is the right policy for a deep queue and the wrong
   one for an empty one.

The cost kernel is a stub so that every number below follows from it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from infera.projection.core.projection.inference_projection import des as des_mod

_BASE_MS = 2.0
_PER_SEQ_MS = 0.05
_PER_PREFILL_TOKEN_MS = 0.002


@dataclass
class _Req:
    input_seq_len: int = 1024
    output_seq_len: int = 64
    max_concurrency: int = 8
    max_num_batched_tokens: int = 8192
    chunked_prefill_size: int = 256
    speculative_num_tokens: int = 0
    speculative_acceptance_rate: float = 0.0
    tokenize_overhead_us: float = 0.0
    detokenize_overhead_us: float = 0.0

    def resolved_max_concurrency(self) -> int:
        return self.max_concurrency

    def resolved_max_context_len(self) -> int:
        return self.input_seq_len + self.output_seq_len

    def resolved_prefix_cache_hit_rate(self) -> float:
        return 0.0


@dataclass
class _Cfg:
    request_config: _Req


class _Kernel:
    def decode_step_latency_ms(self, batch, ctx, q_len):
        return _BASE_MS + _PER_SEQ_MS * batch

    def mixed_step_latency_ms(self, num_decode, prefill_tokens, ctx, prefill_kv, q_len):
        return _BASE_MS + _PER_SEQ_MS * num_decode + _PER_PREFILL_TOKEN_MS * prefill_tokens


def _run(concurrency: int = 8, requests_per_client: int = 20,
         warmup_frac: float = 0.0, pool_tokens: int = 0, **kw):
    cfg = _Cfg(_Req(max_concurrency=concurrency))
    return des_mod.simulate_once(
        cfg,
        _Kernel(),
        rate_per_s=0.0,
        arrival_model="closed",
        num_requests=concurrency * requests_per_client,
        warmup_frac=warmup_frac,
        kv_cache_tokens=pool_tokens,
        closed_loop_clients=concurrency,
        **kw,
    )


# --- 1. the clock bounds the run ------------------------------------------


def test_a_duration_stops_the_run_before_the_request_budget_does():
    """With a horizon short of the budget, the budget stops mattering."""
    full = _run()
    horizon = full.makespan_ms / 4.0
    cut = _run(duration_ms=horizon)
    assert cut.num_requests < full.num_requests, (cut.num_requests, full.num_requests)
    assert cut.makespan_ms <= horizon + 1e-6, (cut.makespan_ms, horizon)


def test_a_horizon_longer_than_the_run_changes_nothing():
    """The clock is a bound, not a target: a slack horizon is inert."""
    full = _run()
    slack = _run(duration_ms=full.makespan_ms * 10.0)
    assert slack.num_requests == full.num_requests
    assert slack.ttft["mean"] == full.ttft["mean"]


def test_the_lane_walk_is_what_a_request_budget_fixes():
    """Why the clock matters: a budget sets per-lane depth from the lane count.

    This is the artifact the horizon exists to remove. Splitting one budget
    across C lanes means each lane advances ``budget / C`` turns, so the depth
    a lane reaches -- and with it the prompt it offers, for a workload whose
    turns grow -- is decided by the concurrency rather than by the workload.
    """
    budget = 400
    depths = {c: budget // c for c in (1, 8, 64)}
    assert depths[1] == 400 and depths[8] == 50 and depths[64] == 6
    # A 66x spread in how far a lane walks, for one unchanged workload.
    assert depths[1] / depths[64] > 60


# --- 2. if the opening transient is dropped, drop it by issue order --------


def _done(n: int, opening: int, opening_wait: float = 500.0):
    """A completed run whose opening requests waited long and finished last.

    This is the shape a closed loop actually produces. The first ``opening``
    requests are issued together at t=0 and queue against each other, so they
    carry the run's longest waits -- and because they waited, they are also
    the last to retire. Everything issued afterwards arrives as a slot frees
    and waits for nothing.
    """
    out = []
    for i in range(n):
        r = des_mod._Req(idx=i, arrival_ms=0.0, prompt_len=1024, output_len=8)
        burst = i < opening
        r.admit_ms = opening_wait if burst else 0.0
        r.first_token_ms = r.admit_ms + 10.0
        # Finish order is the point: the burst retires at the end of the run.
        r.finish_ms = (1e6 + i) if burst else float(i)
        r.generated = r.output_len
        out.append(r)
    out.sort(key=lambda x: x.finish_ms)
    return out


def test_a_completion_ordered_drop_retains_the_opening_burst():
    """Why ``warmup_frac`` cannot remove the transient it was meant to.

    It drops the earliest *completions*, and a request that waited a long
    time for a slot is among the last to finish. Dropping a quarter of the
    run by completion keeps all eight of the burst.
    """
    done = _done(n=80, opening=8)
    kept = des_mod._scored_sample(done, warmup_frac=0.25, warmup_requests=0)
    assert sum(1 for r in kept if r.idx < 8) == 8


def test_an_issue_ordered_drop_removes_it():
    """``warmup_requests`` drops by the order the clients issued, which works."""
    done = _done(n=80, opening=8)
    kept = des_mod._scored_sample(done, warmup_frac=0.0, warmup_requests=8)
    assert sum(1 for r in kept if r.idx < 8) == 0
    assert len(kept) == 72


def test_the_two_drops_report_very_different_queue_waits():
    """The size of the correction, on the shape the harness actually produces."""
    done = _done(n=80, opening=8, opening_wait=500.0)

    def mean_wait(sample):
        return sum(r.admit_ms - r.arrival_ms for r in sample) / len(sample)

    by_completion = mean_wait(
        des_mod._scored_sample(done, warmup_frac=0.25, warmup_requests=0)
    )
    by_issue = mean_wait(
        des_mod._scored_sample(done, warmup_frac=0.0, warmup_requests=8)
    )
    assert by_issue == 0.0
    assert by_completion > 60.0, by_completion


def test_the_issue_order_drop_keeps_at_least_half_the_run():
    """A trace shorter than the requested warmup still reports over something."""
    done = _done(n=20, opening=4)
    kept = des_mod._scored_sample(done, warmup_frac=0.0, warmup_requests=10_000)
    assert len(kept) >= 10


# --- 3. only a real queue is reorderable ----------------------------------


def _reuse(waiting_depth: int, resident_cap: int, blocks_per_req: int = 4):
    """Reuse ``_route_and_warm`` derives for a synthetic interleaved stream.

    The resolved order is not observable from the outside -- the function
    deliberately hands the stream back in arrival order, because the loops
    that consume it are arrival-driven -- so the ordering shows up in the one
    place it has an effect: how much of each prompt the cache covered.
    """
    convs = 4
    reqs = []
    for i in range(convs * 5):
        r = des_mod._Req(idx=i, arrival_ms=float(i), prompt_len=1024, output_len=16)
        # More live conversations than the cache holds, round-robin the way C
        # lanes arrive. Consecutive turns of one conversation share blocks, so
        # whether that reuse survives depends entirely on how many other
        # conversations get admitted in between.
        conv = i % convs
        r.blocks = [conv * 1000 + b for b in range(blocks_per_req)]
        reqs.append(r)
    _, summary = des_mod._route_and_warm(
        reqs,
        policy="kv",
        num_instances=1,
        block_size=64,
        cache_blocks=blocks_per_req + 1,
        rng=random.Random(0),
        overlap_weight=1.0,
        waiting_depth=waiting_depth,
        resident_cap=resident_cap,
    )
    return summary["hit_rate"]


def test_reuse_is_lower_when_nothing_is_queued_to_reorder():
    """The finding: an empty queue leaves reuse to the arrival interleaving.

    Below the cap every lane is resident at once and they evict each other,
    which is the regime where measured reuse collapses. Handing the policy
    the whole client count let it reorder that empty queue and held modelled
    reuse high through exactly the band where the hardware thrashed.
    """
    unqueued = _reuse(waiting_depth=20, resident_cap=20)
    queued = _reuse(waiting_depth=20, resident_cap=1)
    assert queued > unqueued, (queued, unqueued)


def test_the_cap_counts_what_the_pool_holds():
    """``_resident_cap`` divides the pool by a length-biased context.

    Length-biased rather than the plain mean: a request occupies the pool for
    a time proportional to its own length, so what is resident at any instant
    is sampled in proportion to length, and the arithmetic mean overstates how
    many fit.
    """
    reqs = [des_mod._Req(idx=i, arrival_ms=0.0, prompt_len=n, output_len=0)
            for i, n in enumerate([100] * 9 + [900])]
    # Plain mean is 180, so 1800 tokens would look like room for 10.
    assert des_mod._resident_cap(reqs, 1800, 0, enabled=True) < 10
    # No pool to divide leaves the policy unconstrained.
    assert des_mod._resident_cap(reqs, 0, 0, enabled=True) == 0
    # And it is opt-in: by default the window is not constrained at all.
    assert des_mod._resident_cap(reqs, 1800, 0) == 0
