###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Routing outcomes that are decidable by hand, so a wrong one is obviously wrong.

The pipeline tests answer "does a hit happen". These answer "does the RIGHT
worker get it", on inputs where the correct answer is not a matter of degree:
everything to one worker, an exact half-and-half split, a strict subset losing
to a superset. A policy that quietly degenerates to a fixed tiebreak passes a
hit-rate test on one worker and fails these.

That degeneration is not hypothetical. In production kv-aware routing sent
32 of 32 requests to a single worker while the second node sat idle, and every
decision logged ``cache_hits=0``. Nothing in the suite would have caught it,
because no test asked where requests went when there was a choice.

Cost, from ``KvEventAwarePolicy.pick``:

    cost(t) = w_overlap * (blocks_in_request - blocks_t_already_has)
              + w_mm * images_t_lacks
              + in_flight_blocks_on_t

with ties broken by the lower in-flight count. At the default weight of 1.0,
one uncached block and one in-flight request are worth the same, which is what
makes the load-versus-locality cases below decidable rather than a judgement.
"""

from __future__ import annotations

import pytest

from infera.common.worker_pool import (
    DisaggMode,
    EngineType,
    WorkerInfo,
    WorkerStatus,
)
from infera.router.kv_event.client import KvEventClient
from infera.router.kv_event.events import BlockStored
from infera.router.policy.kv_event_aware import KvEventAwarePolicy

BS = 4


def _worker(wid: str) -> WorkerInfo:
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.VLLM,
        status=WorkerStatus.ACTIVE,
        disagg_mode=DisaggMode.MIXED,
        kv_events_endpoint=f"tcp://{wid}:5555",
        kv_block_size=BS,
    )


class _IdentityHasher:
    # BlockHasher's gate for `spawn_probe`; these doubles always render.
    def can_render(self, model_id, engine=None) -> bool:
        return True

    def hash_for(self, body: dict, *, block_size: int, engine=None) -> list[int]:
        from infera.router.kv_event.hasher import hash_request

        return hash_request(body.get("token_ids", []), block_size)


@pytest.fixture
async def rig():
    """Two workers behind one policy, with their event subscriptions stubbed out
    so blocks can be placed directly and the outcome is not a race."""
    client = KvEventClient()
    policy = KvEventAwarePolicy(client, _IdentityHasher())
    a, b = _worker("a:1"), _worker("b:1")
    for w in (a, b):
        policy.on_worker_added(w)
        for t in client._subs[w.worker_id].tasks:
            t.cancel()
    return policy, client, a, b


def _store(client, wid: str, tokens: list[int], first_hash: int = 0) -> None:
    """Give a worker the blocks for `tokens`, as one aligned event."""
    n = len(tokens) // BS
    client._handle_event(
        client._subs[wid],
        BlockStored(
            block_hashes=[bytes([first_hash + i]) for i in range(n)],
            parent_block_hash=None,
            token_ids=list(tokens),
            block_size=BS,
            lora_id=None,
            group_idx=0,
            kv_cache_spec_kind="full_attention",
        ),
        rank=None,
    )


def _pick(policy, workers, tokens):
    target, _ = policy.pick(list(workers), {"model": "m", "token_ids": list(tokens)})
    return target.worker.worker_id


# --- everything to one worker ------------------------------------------------


async def test_every_request_goes_to_the_only_worker_that_has_the_prefix(rig):
    """One worker holds the whole prompt, the other holds nothing. There is no
    tradeoff to weigh: 20/20 to the holder."""
    policy, client, a, b = rig
    prompt = list(range(40))  # 10 blocks
    _store(client, "a:1", prompt)

    picks = [_pick(policy, (a, b), prompt) for _ in range(20)]
    assert picks.count("a:1") == 20, f"{picks.count('b:1')}/20 went to the cold worker"


async def test_the_holder_still_wins_when_it_is_the_second_candidate(rig):
    """Candidate order must not decide this. A degenerate tiebreak looks correct
    whenever the holder happens to be listed first."""
    policy, client, a, b = rig
    prompt = list(range(40))
    _store(client, "b:1", prompt)

    assert _pick(policy, (a, b), prompt) == "b:1"
    assert _pick(policy, (b, a), prompt) == "b:1"


# --- an exact half-and-half split -------------------------------------------


async def test_two_disjoint_prefixes_split_exactly_down_the_middle(rig):
    """Each worker owns one prefix. Requests must sort themselves 10/10 by
    content, not alternate: a round-robin router also produces 10/10 overall,
    but sends half of each prefix to the wrong worker."""
    policy, client, a, b = rig
    p1 = list(range(40))
    p2 = list(range(1000, 1040))
    _store(client, "a:1", p1, first_hash=0)
    _store(client, "b:1", p2, first_hash=100)

    by_prefix = {"p1": [], "p2": []}
    for _ in range(10):
        by_prefix["p1"].append(_pick(policy, (a, b), p1))
        by_prefix["p2"].append(_pick(policy, (a, b), p2))

    assert by_prefix["p1"] == ["a:1"] * 10, "prefix 1 must always follow its holder"
    assert by_prefix["p2"] == ["b:1"] * 10, "prefix 2 must always follow its holder"


# --- more of the prefix wins ------------------------------------------------


@pytest.mark.parametrize(
    "a_blocks,b_blocks,winner",
    [
        (2, 8, "b:1"),  # strict subset loses to the longer match
        (8, 2, "a:1"),  # and the same the other way round
        (0, 1, "b:1"),  # even one block beats nothing
        (9, 10, "b:1"),  # a single block of difference still decides
    ],
)
async def test_the_longer_cached_prefix_wins(rig, a_blocks, b_blocks, winner):
    policy, client, a, b = rig
    prompt = list(range(40))  # 10 blocks
    if a_blocks:
        _store(client, "a:1", prompt[: a_blocks * BS], first_hash=0)
    if b_blocks:
        _store(client, "b:1", prompt[: b_blocks * BS], first_hash=100)

    assert _pick(policy, (a, b), prompt) == winner


# --- cold start must spread, not pile up ------------------------------------


async def test_a_cold_fleet_spreads_instead_of_piling_onto_one_worker(rig):
    """Neither worker has anything, so every candidate costs the same and the
    in-flight count is the only signal left. This is the exact shape of the
    production failure -- 32/32 to one worker -- and it is what the tiebreak
    exists to prevent."""
    policy, _client, a, b = rig
    counts = {"a:1": 0, "b:1": 0}
    for i in range(20):
        prompt = list(range(i * 100, i * 100 + 40))  # a fresh prefix each time
        target, blocks = policy.pick([a, b], {"model": "m", "token_ids": prompt})
        wid = target.worker.worker_id
        counts[wid] += 1
        policy.on_request_started(target.route_key, blocks)  # stays in flight

    assert counts == {"a:1": 10, "b:1": 10}, f"cold fleet did not spread: {counts}"


async def test_load_outweighs_a_one_block_cache_edge(rig):
    """One cached block and one in-flight request are both worth 1 at the
    default weight, so a worker holding one extra block but carrying two more
    in-flight requests must lose. Stated as a test because it is the tradeoff
    someone will change the weight to alter."""
    policy, client, a, b = rig
    prompt = list(range(40))
    _store(client, "a:1", prompt[: 1 * BS], first_hash=0)  # A: 1 block, 9 misses

    # Put 2 blocks' worth of in-flight work on A. Cost(A) = 9 + 2 = 11 > Cost(B) = 10.
    ta, blocks = policy.pick([a], {"model": "m", "token_ids": prompt})
    policy.on_request_started(ta.route_key, blocks[:2])

    assert _pick(policy, (a, b), prompt) == "b:1"


async def test_finishing_a_request_returns_the_worker_to_contention(rig):
    """In-flight cost must be released, or a worker that served a burst is
    written off long after it went idle."""
    policy, client, a, b = rig
    prompt = list(range(40))
    _store(client, "a:1", prompt[: 1 * BS], first_hash=0)

    ta, blocks = policy.pick([a], {"model": "m", "token_ids": prompt})
    policy.on_request_started(ta.route_key, blocks[:2])
    assert _pick(policy, (a, b), prompt) == "b:1"

    policy.on_request_finished(ta.route_key, blocks[:2])
    assert _pick(policy, (a, b), prompt) == "a:1", (
        "A holds a block and is idle again; it must win once its load is released"
    )


# --- load must survive between requests, not just during them ----------------


async def test_serial_traffic_does_not_pin_every_request_to_one_worker(rig):
    """The reported production failure, reduced to its smallest form.

    Each request finishes before the next is picked, so nothing is ever in
    flight when a decision is made. If in-flight blocks are the only load
    signal, every worker reads 0, the first cold pick wins on candidate order,
    and the cache it gains re-elects it for the rest of the run -- 100/0 on a
    symmetric pair, at any overlap weight.

    Distinct prefixes per request, so there is no locality reason to prefer
    either worker.
    """
    policy, _client, a, b = rig
    counts = {"a:1": 0, "b:1": 0}
    for i in range(40):
        prompt = list(range(i * 100, i * 100 + 40))
        target, blocks = policy.pick([a, b], {"model": "m", "token_ids": prompt})
        counts[target.worker.worker_id] += 1
        # Started AND finished before the next pick: the paced-traffic shape.
        policy.on_request_started(target.route_key, blocks)
        policy.on_request_finished(target.route_key, blocks)

    assert min(counts.values()) > 0, f"one worker got everything: {counts}"
    assert abs(counts["a:1"] - counts["b:1"]) <= 4, f"serial traffic did not spread: {counts}"


async def test_serial_traffic_spreads_even_at_a_high_overlap_weight(rig):
    """A high --kv-prefill-overlap-weight must not reintroduce the pin.

    The load term is unweighted, so a fix that keeps it bounded independently
    of request size loses to one block of cache edge once the weight exceeds
    that bound. 20.0 is the documented production prefill weight.
    """
    _policy, client, a, b = rig
    policy = KvEventAwarePolicy(client, _IdentityHasher(), overlap_weight=20.0)
    counts = {"a:1": 0, "b:1": 0}
    for i in range(40):
        prompt = list(range(i * 100, i * 100 + 40))
        target, blocks = policy.pick([a, b], {"model": "m", "token_ids": prompt})
        counts[target.worker.worker_id] += 1
        policy.on_request_started(target.route_key, blocks)
        policy.on_request_finished(target.route_key, blocks)

    assert min(counts.values()) > 0, f"high weight pinned the fleet: {counts}"


async def test_a_fully_cached_prompt_stays_on_its_holder_under_serial_traffic(rig):
    """Balance must not cost affinity.

    The recent-dispatch charge is the blocks the winner had to COMPUTE, so a
    worker serving a prompt it already holds accrues nothing and keeps winning
    it. Were the charge the request's total size instead, repeat requests would
    ping-pong and every one after the first would miss.
    """
    policy, client, a, b = rig
    prompt = list(range(40))
    _store(client, "a:1", prompt)

    for _ in range(20):
        target, blocks = policy.pick([a, b], {"model": "m", "token_ids": prompt})
        assert target.worker.worker_id == "a:1"
        policy.on_request_started(target.route_key, blocks)
        policy.on_request_finished(target.route_key, blocks)


async def test_a_saturated_incumbent_yields_new_work_to_an_idle_worker(rig):
    """Recent load has to be able to outweigh a shared-prefix edge.

    A worker that has served a long run of traffic carries a large recent
    total; a cold worker carries none. A new prompt sharing only a short prefix
    with the incumbent must go to the idle worker.
    """
    policy, client, a, b = rig
    shared = list(range(8))  # 2 blocks both would hit on
    for i in range(20):
        prompt = shared + list(range(1000 + i * 100, 1000 + i * 100 + 32))
        target, blocks = policy.pick([a], {"model": "m", "token_ids": prompt})
        policy.on_request_started(target.route_key, blocks)
        # first_hash stays a single byte: _store packs it with bytes([...]).
        _store(client, "a:1", prompt, first_hash=i * 10)
        policy.on_request_finished(target.route_key, blocks)

    fresh = shared + list(range(9000, 9032))
    assert _pick(policy, (a, b), fresh) == "b:1", (
        "a saturated worker kept new work on the strength of a 2-block prefix"
    )


async def test_recent_load_decays_so_an_idle_worker_returns_to_contention(rig):
    """The charge is transient. A worker that took a burst and then went quiet
    must not be written off permanently -- otherwise the fix trades one
    starvation mode for another."""
    policy, _client, a, b = rig
    for i in range(20):
        prompt = list(range(i * 100, i * 100 + 40))
        target, blocks = policy.pick([a], {"model": "m", "token_ids": prompt})
        policy.on_request_started(target.route_key, blocks)
        policy.on_request_finished(target.route_key, blocks)

    assert policy._recent_blocks.get("a:1", 0.0) > 0.0
    for i in range(400):
        prompt = list(range(50_000 + i * 100, 50_000 + i * 100 + 40))
        target, blocks = policy.pick([b], {"model": "m", "token_ids": prompt})
        policy.on_request_started(target.route_key, blocks)
        policy.on_request_finished(target.route_key, blocks)

    assert "a:1" not in policy._recent_blocks, "an idle worker never returned to 0"


async def test_prompts_too_short_to_hash_still_spread(rig):
    """A prompt under the index block size hashes to nothing. The load charge
    must not be zero for those, or the term stays 0 on every candidate and the
    tie sends the whole run to whichever worker sorts first -- the 100/0 split
    reappearing on short prompts, which is what a user hit after the load-term
    fix landed.

    token_ids shorter than BS produce no blocks at all, which is the same state
    the real hasher reaches on a sub-768-token prompt.
    """
    policy, _client, a, b = rig
    counts = {"a:1": 0, "b:1": 0}
    for i in range(30):
        # Two tokens: fewer than BS, so hash_request returns [].
        target, blocks = policy.pick([a, b], {"model": "m", "token_ids": [i, i + 1]})
        assert blocks == [], "precondition: this prompt must hash to zero blocks"
        counts[target.worker.worker_id] += 1
        policy.on_request_started(target.route_key, blocks)
        policy.on_request_finished(target.route_key, blocks)

    assert min(counts.values()) > 0, f"unhashable prompts pinned one worker: {counts}"
    assert abs(counts["a:1"] - counts["b:1"]) <= 2, f"did not spread: {counts}"


async def test_a_fully_cached_pick_is_still_charged_nothing(rig):
    """The zero-block floor must not leak into the fully-cached case: a worker
    serving a prompt it already holds does no prefill, so it accrues no load and
    keeps winning that prompt. Distinguishing the two is the whole point of
    passing request_blocks alongside the miss count."""
    policy, client, a, b = rig
    prompt = list(range(40))
    _store(client, "a:1", prompt)

    for _ in range(15):
        target, blocks = policy.pick([a, b], {"model": "m", "token_ids": prompt})
        assert target.worker.worker_id == "a:1"
        policy.on_request_started(target.route_key, blocks)
        policy.on_request_finished(target.route_key, blocks)
    assert policy._recent_blocks.get("a:1", 0.0) == 0.0, "fully-cached picks accrued load"
