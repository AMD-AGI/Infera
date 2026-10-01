###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""``--prefix-cache-hit-rate`` has to reach the single-engine DES.

The multi-instance router derives each request's hit from a content-addressed
block cache and hands the seeded requests over in ``prebuilt``. The single-engine
path had no equivalent: it accepted the flag, built every request with
``num_computed = 0``, and reprefilled the whole prompt. On an agentic shape that
is a 130k prefill where the analytical path does 10k, and it showed up as the DES
saturating at 0.08 req/s against 1.78 req/s for the same configuration -- a
disagreement that reads as a physics dispute and is actually a dropped flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from infera.projection.core.projection.inference_projection import des as des_mod


@dataclass
class _Req:
    input_seq_len: int = 130000
    output_seq_len: int = 900
    hit: float = 0.92

    def resolved_prefix_cache_hit_rate(self) -> float:
        return self.hit


@dataclass
class _Cfg:
    request_config: _Req = field(default_factory=_Req)


def _seed(hit: float, prompt_len: int = 130000, n: int = 8):
    """Run the seeding block the way ``simulate_once`` does, in isolation."""
    pending = [
        des_mod._Req(idx=i, arrival_ms=0.0, prompt_len=prompt_len, output_len=900) for i in range(n)
    ]
    cfg = _Cfg(request_config=_Req(input_seq_len=prompt_len, hit=hit))
    h = cfg.request_config.resolved_prefix_cache_hit_rate()
    if h > 0.0:
        for r in pending:
            cached = max(0, min(int(r.prompt_len * h), r.prompt_len - 1))
            if cached > 0:
                r.cached_prefix = cached
                r.num_computed = cached
    return pending


def test_a_reused_prompt_is_not_reprefilled():
    """92% reuse must leave 8% of the prompt to compute, not all of it."""
    reqs = _seed(0.92)
    for r in reqs:
        assert r.num_computed == 119600, r.num_computed
        assert r.prompt_len - r.num_computed == 10400


def test_no_hit_rate_leaves_the_whole_prompt_to_compute():
    """The default path must project exactly as it always did."""
    for r in _seed(0.0):
        assert r.num_computed == 0
        assert r.cached_prefix == 0


def test_a_fully_cached_prompt_still_runs_one_forward():
    """Otherwise the request never emits a first token."""
    for r in _seed(1.0, prompt_len=4096):
        assert r.num_computed == 4095
        assert r.prompt_len - r.num_computed == 1


def test_the_scheduler_only_charges_for_the_uncached_suffix():
    """The seeding is only worth anything if step scheduling reads it."""
    r = _seed(0.92)[0]
    need = r.prompt_len - r.num_computed
    assert need == 10400
    # And the cache blocks count as resident context immediately.
    assert r.kv_len == 119600


def test_a_shared_prefix_is_charged_to_the_pool_once_and_not_per_sharer():
    """Admission has to read the hit too, not just prefill.

    The pool gate charged every request its whole context, so a corpus whose
    requests all continue from the same conversation paid for the one resident
    copy of that prefix once per sharer. At the 96-98% reuse an agentic replay
    carries that inflates a request's footprint 25-50x: a measured 7.67M-token
    pool looked full at ~19 concurrent and the replay queued from there, while
    the MI355X ladder reports its pool at 0.1-0.8% utilization at every
    concurrency and never binds. The uncached suffix plus what the request
    generates is what it actually adds.
    """
    r = _seed(0.92)[0]
    assert r.reserved_kv == (r.prompt_len - r.cached_prefix) + r.output_len == 11300
    # A cold request is untouched, which is every fixed-sequence workpoint.
    cold = _seed(0.0)[0]
    assert cold.reserved_kv == cold.prompt_len + cold.output_len


def _sweep(cache: des_mod._BlockCache, n_convs: int, depth: int, rounds: int):
    """Interleave ``n_convs`` conversations that share a head and diverge.

    The shape every long-context corpus has: a common prefix, then per-thread
    continuation. Each round revisits every thread, so a thread's own blocks
    are reused a full sweep after they were written.
    """
    head = [0, 1, 2]
    hits = 0
    for _ in range(rounds):
        for c in range(n_convs):
            blocks = head + [1000 + c * depth + d for d in range(depth)]
            hits += cache.prefix_match(blocks)
            cache.insert(blocks)
    return hits


def test_the_shared_head_of_a_corpus_survives_a_working_set_it_cannot_hold():
    """Leaf-first eviction is the whole difference between 0% reuse and most of it.

    Blocks are touched in prefix order, so a corpus larger than the pool is a
    cyclic sweep -- the access pattern that makes a flat LRU evict every block
    exactly before it is reused. A radix cache cannot do that: it may only
    reclaim a block no resident block continues from, so the shared head stays
    and the divergent tails are what go.
    """
    # Room for the head and roughly half the threads, so the sweep cannot fit.
    cache = des_mod._BlockCache(capacity_blocks=3 + 10 * 4)
    hits = _sweep(cache, n_convs=20, depth=4, rounds=4)

    assert hits > 0, "a flat LRU scores exactly zero here; that was the bug"
    assert cache.prefix_match([0, 1, 2]) == 3, "the shared head must still be resident"


def test_a_block_is_never_dropped_while_something_continues_from_it():
    """The invariant that makes it a radix cache rather than a priority order."""
    cache = des_mod._BlockCache(capacity_blocks=4)
    cache.insert([0, 1, 2, 3])
    cache.insert([0, 1, 2, 4])  # diverges at the last block, forcing eviction

    assert cache.prefix_match([0, 1, 2]) == 3, "an interior block was reclaimed"
    assert len(cache) <= 4


def test_the_cache_still_honours_its_capacity():
    """Protecting interior blocks must not become "never evict"."""
    cache = des_mod._BlockCache(capacity_blocks=8)
    _sweep(cache, n_convs=30, depth=4, rounds=2)
    assert len(cache) <= 8


def test_an_unbounded_cache_evicts_nothing():
    cache = des_mod._BlockCache(capacity_blocks=0)
    _sweep(cache, n_convs=10, depth=4, rounds=2)
    assert cache.evictions == 0
    assert cache.prefix_match([0, 1, 2]) == 3


def _lanes(n_lanes: int, turns: int, blocks_per_turn: int = 6):
    """A closed-loop corpus: ``n_lanes`` conversations taking turns round-robin.

    Each lane extends its own context every turn, so turn ``t`` of a lane can
    reuse everything that lane wrote at ``t - 1`` -- if it is still resident a
    full sweep of the other lanes later. This is the shape of every agentic
    trace, and the distance between a lane's turns is the offered concurrency.
    """
    reqs = []
    for t in range(turns):
        for lane in range(n_lanes):
            blocks = [lane * 10_000 + b for b in range((t + 1) * blocks_per_turn)]
            reqs.append(
                des_mod._Req(
                    idx=len(reqs),
                    arrival_ms=float(len(reqs)),
                    prompt_len=len(blocks) * 16,
                    output_len=16,
                    blocks=blocks,
                )
            )
    return reqs


def _warm(reqs, capacity: int, depth: int) -> float:
    import random

    _, summary = des_mod._route_and_warm(
        [
            des_mod._Req(
                idx=r.idx,
                arrival_ms=r.arrival_ms,
                prompt_len=r.prompt_len,
                output_len=r.output_len,
                blocks=list(r.blocks),
            )
            for r in reqs
        ],
        policy="kv",
        num_instances=1,
        block_size=16,
        cache_blocks=capacity,
        rng=random.Random(0),
        waiting_depth=depth,
    )
    return summary["block_hit_rate"]


def test_reuse_does_not_fall_off_a_cliff_once_the_offered_load_stops_fitting():
    """Oldest-first admission turns "does not fit" into "reuses nothing".

    Past the point where the pool cannot hold every outstanding context, taking
    the oldest waiting request walks the lanes round-robin, which guarantees a
    lane's context is evicted before its next turn: every request then pays a
    full reprefill and reuse is not merely degraded, it is exactly zero. Real
    hardware degrades gradually instead, because a scheduler picks off its
    waiting queue by longest resident prefix and keeps working on whatever is
    still resident. The measured ladders show exactly that -- reuse holding
    near 0.93 up to the admission ceiling and never collapsing past it.
    """
    reqs = _lanes(n_lanes=24, turns=6)
    # Room for a few lanes' contexts, nowhere near all 24.
    capacity = 6 * 21

    oldest_first = _warm(reqs, capacity, depth=0)
    prefix_first = _warm(reqs, capacity, depth=24)

    assert oldest_first == 0.0, "the cliff being fixed; if this moves, re-derive it"
    assert prefix_first > 0.6, prefix_first


def test_admission_order_changes_nothing_while_the_load_still_fits():
    """The correction has to be invisible below the pressure point.

    With room for every outstanding context nothing is evicted between a lane's
    turns, so both orders touch the same blocks and must score identically.
    Were that not so this would be a tuning knob rather than a missing
    behaviour, and it would be moving results in regimes it has no business in.
    """
    reqs = _lanes(n_lanes=8, turns=6)
    capacity = 0  # unbounded: everything offered fits

    assert _warm(reqs, capacity, depth=0) == _warm(reqs, capacity, depth=8)


def test_the_slot_a_served_request_frees_goes_back_to_its_own_client():
    """Which client gets the freed slot is most of the correction.

    Serving a request frees the client that issued it, and that client's next
    turn continues the context just served. Hand the slot to whichever request
    is next in arrival order instead and the closed loop degrades into a
    sliding window: the reuse that was about to be collected is dropped, and
    on the dsv4 ladder that is the difference between reproducing the measured
    0.93 at C=128 and reporting 0.83.
    """
    reqs = _lanes(n_lanes=24, turns=6)
    capacity = 6 * 21

    # Same policy, same capacity; the only difference is that a depth which
    # does not match the interleave misattributes the freed slot.
    tied = _warm(reqs, capacity, depth=24)
    untied = _warm(reqs, capacity, depth=23)

    assert tied > untied, (tied, untied)


def test_the_arrival_stream_each_instance_replays_is_still_in_arrival_order():
    """Schedule order says how much to prefill, not when a request showed up.

    The per-instance loops downstream are arrival-driven, so resolving prefixes
    in a different order must not leak into the stream they replay.
    """
    import random

    reqs = _lanes(n_lanes=8, turns=4)
    per_inst, _ = des_mod._route_and_warm(
        reqs,
        policy="kv",
        num_instances=1,
        block_size=16,
        cache_blocks=32,
        rng=random.Random(0),
        waiting_depth=8,
    )
    arrivals = [r.arrival_ms for r in per_inst[0]]
    assert arrivals == sorted(arrivals)


def test_a_split_reports_the_prefix_cache_it_actually_warmed(tmp_path):
    """The disaggregated branch warmed a cache and threw the summary away.

    Every split row then reported no hit rate, which reads as a split that
    reuses nothing rather than one whose reuse was never recorded -- and it
    was the disaggregated rows that the reuse question was being asked about.
    """
    import json

    from .test_des_disaggregated import _Cfg, _Pool, _Projector, _Req

    # Two passes over the same prompts, so there is reuse to find.
    trace = tmp_path / "trace.jsonl"
    with trace.open("w") as f:
        for turn in range(2):
            for lane in range(4):
                f.write(
                    json.dumps(
                        {
                            "timestamp": turn * 1000 + lane,
                            "input_length": 2048,
                            "output_length": 64,
                            "hash_ids": [0, 1, 2, 100 + lane, 200 + lane],
                        }
                    )
                    + "\n"
                )

    point = des_mod.run_des(
        _Cfg(_Req(max_concurrency=4, chunked_prefill_size=256, output_seq_len=64)),
        _Projector(_Pool(), _Pool()),
        arrival_model="closed",
        rate_per_s=0.0,
        num_requests=8,
        warmup_frac=0.0,
        closed_loop=True,
        mooncake_trace=str(trace),
        block_size=512,
        cache_blocks=64,
    )["point"]

    assert point.packing["disaggregated"] == 1.0, "this must be the split path"
    assert point.prefix, "a split reported no prefix-cache summary at all"
    assert point.prefix["hit_rate"] > 0.0
