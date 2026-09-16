###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A split is two stations on one clock, and the DES had neither.

The event loop knew about one engine. A disaggregated candidate handed to it was
scheduled as a single unified batch and priced with the parent's parallelism, so
what came back described a colocated deployment -- and nothing in the output
said so. That is why trace-driven results excluded the topology rather than
reporting it: the alternative was a number for a machine nobody had configured.

Two properties have to hold together, and either one alone is a different bug:

* **The pools do not share a step.** No decode batch carries a prefill chunk.
  That is the argument for splitting at all.
* **The pools do not share GPUs.** They run at the same time on disjoint
  hardware. An engine that merely alternated prefill-only and decode-only steps
  would also report no mixed steps while serialising work that overlaps, which
  inflates TTFT and TPOT together and would look like the topology losing.

The cost kernels here are stubs, deliberately -- these assert the *scheduling*,
not the model's physics, so every number below follows from an explicit step
cost. The two pools are given deliberately different kernels, because a single
shared one cannot tell "priced per pool" from "priced once".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from infera.projection.core.projection.inference_projection import des as des_mod

_BASE_MS = 2.0
_PER_SEQ_MS = 0.05
_PER_PREFILL_TOKEN_MS = 0.002


@dataclass
class _Req:
    input_seq_len: int = 1024
    output_seq_len: int = 256
    max_concurrency: int = 8
    max_num_batched_tokens: int = 8192
    chunked_prefill_size: int = 0
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
class _Disagg:
    enabled: bool = True
    prefill_replicas: int = 1
    decode_replicas: int = 1


@dataclass
class _MP:
    tensor_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1


@dataclass
class _PoolCfg:
    model_parallel_config: _MP = field(default_factory=_MP)


@dataclass
class _Cfg:
    request_config: _Req
    disaggregation_config: _Disagg = field(default_factory=_Disagg)


class _Pool:
    """One pool's cost model, which records what it was asked to price."""

    def __init__(self, scale: float = 1.0, tp: int = 4):
        self.scale = scale
        self.cfg = _PoolCfg(_MP(tensor_model_parallel_size=tp))
        self.decode_calls = 0
        self.mixed_calls = 0

    def decode_step_latency_ms(self, batch, ctx, q_len):
        self.decode_calls += 1
        return self.scale * (_BASE_MS + _PER_SEQ_MS * batch)

    def mixed_step_latency_ms(self, num_decode, prefill_tokens, ctx, prefill_kv, q_len):
        self.mixed_calls += 1
        return self.scale * (
            _BASE_MS + _PER_SEQ_MS * num_decode + _PER_PREFILL_TOKEN_MS * prefill_tokens
        )


class _Projector:
    """Stands in for the disaggregated projector's two-pool interface."""

    def __init__(self, prefill: _Pool, decode: _Pool, handoff_ms: float = 1.0):
        self.prefill = prefill
        self.decode = decode
        self.handoff_ms = handoff_ms

    def pool_projectors(self):
        return self.prefill, self.decode

    def kv_handoff_ms(self, decode_proj, context_len):
        return self.handoff_ms


def _run(
    concurrency: int = 8,
    *,
    requests_per_client: int = 6,
    decode_scale: float = 1.0,
    prefill_scale: float = 1.0,
    decode_replicas: int = 1,
    handoff_ms: float = 1.0,
    chunk: int = 256,
    output_len: int = 256,
    projector: _Projector | None = None,
):
    cfg = _Cfg(
        _Req(
            max_concurrency=concurrency,
            chunked_prefill_size=chunk,
            output_seq_len=output_len,
        ),
        _Disagg(decode_replicas=decode_replicas),
    )
    proj = projector or _Projector(_Pool(prefill_scale), _Pool(decode_scale), handoff_ms=handoff_ms)
    res = des_mod.simulate_disaggregated(
        cfg,
        proj,
        rate_per_s=0.0,
        num_requests=concurrency * requests_per_client,
        warmup_frac=0.5,
        closed_loop_clients=concurrency,
    )
    return res, proj


def test_a_split_says_it_was_simulated_as_one():
    """The absent flag is the bug: a colocated schedule looked identical."""
    res, _ = _run()
    assert res.packing["disaggregated"] == 1.0
    assert res.num_requests > 0


def test_run_des_sends_a_split_to_the_two_station_loop():
    """The routing, at the entry point ``--des-*`` actually reaches.

    Without this the flags are accepted, a simulation runs, and it is of the
    wrong machine.
    """
    cfg = _Cfg(_Req(max_concurrency=8, chunked_prefill_size=256))
    point = des_mod.run_des(
        cfg,
        _Projector(_Pool(), _Pool()),
        arrival_model="closed",
        rate_per_s=0.0,
        num_requests=48,
        warmup_frac=0.5,
        closed_loop=True,
    )["point"]
    assert point.packing["disaggregated"] == 1.0
    assert point.packing["closed_loop_clients"] == 8.0


def test_the_decode_pool_is_never_asked_to_price_a_mixed_step():
    """Moving prefill off these GPUs is the entire topology.

    Asserted against the kernel rather than the reported fraction, because the
    fraction is something this module writes and the call is something the
    schedule forces.
    """
    res, proj = _run()
    assert proj.decode.mixed_calls == 0
    assert proj.decode.decode_calls > 0
    assert res.packing["mixed_step_fraction"] == 0.0


def test_the_prefill_pool_is_never_asked_to_price_a_decode_step():
    """The other half of the same property."""
    _res, proj = _run()
    assert proj.prefill.decode_calls == 0
    assert proj.prefill.mixed_calls > 0


def _open(rate_per_s: float, *, num_requests: int = 200, chunk: int = 1024, **kw):
    """An arrival stream rather than a client loop.

    A closed load of identical requests runs in lockstep here -- everyone
    prefills together, hands off together and decodes together -- so it is the
    wrong instrument for an overlap question. Staggered arrivals put one
    request in each pool at once, which is the state the topology exists for.
    """
    cfg = _Cfg(_Req(max_concurrency=64, chunked_prefill_size=chunk, **kw))
    proj = _Projector(_Pool(), _Pool())
    return des_mod.simulate_disaggregated(
        cfg,
        proj,
        rate_per_s=rate_per_s,
        arrival_model="poisson",
        num_requests=num_requests,
        warmup_frac=0.2,
        seed=3,
    )


def test_the_pools_run_at_the_same_time_rather_than_taking_turns():
    """Disjoint GPUs, so the busy periods overlap and can exceed the makespan.

    This is what separates a two-station model from one engine alternating
    prefill-only and decode-only steps. The alternating engine would report the
    same zero mixed steps while serialising work that in fact overlaps, which
    inflates TTFT and TPOT together -- and its two utilisations could not sum
    past 1, because it only has the one set of GPUs to be busy on.
    """
    # Sized so both stations carry real load rather than one waiting on the
    # other: at this rate the prefill pool runs about a third busy while the
    # decode pool never empties. Measured 0.37 + 0.99.
    res = _open(15.0, input_seq_len=8192, output_seq_len=64)
    both = res.packing["prefill_utilization"] + res.packing["decode_utilization"]
    assert res.packing["prefill_utilization"] > 0.2, res.packing
    assert res.packing["decode_utilization"] > 0.8, res.packing
    assert both > 1.2, res.packing


def test_one_client_has_nothing_to_overlap_with():
    """The control for the test above: the overlap has to come from load.

    A single client is in exactly one pool at a time, so the busy periods are
    disjoint by construction and must sum to at most the makespan. If this also
    exceeded 1 the overlap above would be an accounting error rather than
    concurrency.
    """
    res, _ = _run(concurrency=1, requests_per_client=24)
    both = res.packing["prefill_utilization"] + res.packing["decode_utilization"]
    assert both <= 1.0 + 1e-9, res.packing


def test_each_pool_is_priced_by_its_own_cost_model():
    """A TP4 prefill pool beside TP2 decode replicas costs what those shapes do.

    Pricing both from the parent's parallelism -- the colocated layout, which
    belongs to neither pool -- is what the single-engine path did.
    """
    base, _ = _run(concurrency=16)
    slow_decode, _ = _run(concurrency=16, decode_scale=4.0)
    slow_prefill, _ = _run(concurrency=16, prefill_scale=4.0)
    assert slow_decode.tpot["mean"] > 2.0 * base.tpot["mean"]
    assert slow_prefill.ttft["mean"] > 2.0 * base.ttft["mean"]
    # And the cost does not leak across the handoff.
    assert slow_decode.ttft["mean"] < 1.5 * base.ttft["mean"]


def test_the_handoff_between_the_pools_is_charged():
    """KV has to cross the link, and the request waits for it."""
    cheap, _ = _run(concurrency=8, handoff_ms=1.0)
    dear, _ = _run(concurrency=8, handoff_ms=500.0)
    assert dear.packing["kv_handoff_ms"] == 500.0
    assert dear.e2e["mean"] > cheap.e2e["mean"]


def test_the_client_slot_is_released_at_decode_retirement_not_at_the_handoff():
    """Otherwise one client holds two requests and C is not the concurrency.

    With a single client the prefill pool can only ever see one request, so a
    slot released at the handoff would show up here as prefill batches larger
    than one while the previous request was still decoding.
    """
    res, _ = _run(concurrency=1, requests_per_client=24)
    assert res.packing["avg_prefill_reqs"] == 1.0, res.packing


def test_every_request_is_served_exactly_once():
    """Two stations and a link are three places to drop or duplicate work."""
    res, _ = _run(concurrency=8, requests_per_client=5)
    assert res.num_requests == 40


def test_the_decode_pool_spreads_the_population_over_its_replicas():
    """Replicas are why the decode pool can run narrower than prefill.

    Two replicas hold half the sequences each, and the stub step cost grows with
    the batch, so splitting has to show up as throughput.
    """
    one, _ = _run(concurrency=32, decode_replicas=1)
    two, _ = _run(concurrency=32, decode_replicas=2)
    assert two.system_throughput_tps > one.system_throughput_tps, (
        one.system_throughput_tps,
        two.system_throughput_tps,
    )


def test_the_loop_stops_instead_of_spinning_when_nothing_can_progress():
    """A prompt longer than the model's context never finishes prefilling.

    Both step functions decline to advance their own clock when they schedule
    nothing, so "stuck" and "idle" are indistinguishable from outside the
    station -- and re-picking the stalled station at an unchanged clock spins
    the loop until its iteration guard, reporting zero completed requests for a
    run that had work to do.
    """
    cfg = _Cfg(_Req(max_concurrency=4, chunked_prefill_size=256, input_seq_len=1024))
    # A context shorter than the prompt: the request can never complete.
    cfg.request_config.resolved_max_context_len = lambda: 64
    res = des_mod.simulate_disaggregated(
        cfg,
        _Projector(_Pool(), _Pool()),
        rate_per_s=0.0,
        num_requests=8,
        warmup_frac=0.0,
        closed_loop_clients=4,
    )
    assert res.num_requests == 0
    assert res.makespan_ms == 0.0


def _run_split(concurrency: int, *, requests_per_client: int = 4,
               prefill_scale: float = 0.1, chunk: int = 256,
               output_len: int = 512, range_ratio: float = 0.0):
    """A split whose prefill pool finishes well ahead of its decode pool.

    That ordering is the one that matters below: prefill drains its queue and
    goes idle while decode is still working, which is exactly when the two
    stations' clocks come apart.
    """
    cfg = _Cfg(
        _Req(max_concurrency=concurrency, chunked_prefill_size=chunk,
             output_seq_len=output_len),
        _Disagg(decode_replicas=1),
    )
    proj = _Projector(_Pool(prefill_scale), _Pool(1.0), handoff_ms=1.0)
    return des_mod.simulate_disaggregated(
        cfg,
        proj,
        rate_per_s=0.0,
        num_requests=concurrency * requests_per_client,
        warmup_frac=0.0,
        range_ratio=range_ratio,
        closed_loop_clients=concurrency,
        seed=7,
    )


def test_a_closed_loop_keeps_its_clients_in_flight():
    """The two stations keep separate clocks, and only one was feeding itself.

    A closed-loop client retires on the *decode* clock and submits its
    replacement stamped with that time, but arrivals were released against the
    *prefill* clock. Whenever prefill drained its queue its clock stopped while
    decode's kept moving, so every request reissued in that window was invisible
    to the pool that has to prefill it. The population in flight collapsed from
    C to about one, and with nothing to batch with the decode pool ran at batch
    1 however many clients were offered -- which reads as a split that cannot
    use concurrency rather than as a bug in the loop.
    """
    res = _run_split(32)
    # Nearly all of a request's life is decode here, so most of the 32 clients
    # should be resident there at any moment. This was 1.29.
    assert res.packing["avg_decode_reqs"] > 16.0
    assert res.packing["closed_loop_clients"] == 32.0


def test_the_decode_batch_grows_with_the_client_count():
    """Offering more clients has to put more of them in the batch.

    The failure this guards is specifically a *flat* response: the collapsed
    loop returned about the same batch, and so about the same throughput, at
    every concurrency, which is indistinguishable from a deployment that is
    genuinely saturated. Before the fix a fourfold rise in clients moved the
    batch from 1.08 to 1.29.
    """
    few = _run_split(8)
    many = _run_split(32)
    assert many.packing["avg_decode_reqs"] > 2.5 * few.packing["avg_decode_reqs"]
    assert many.system_throughput_tps > 2.0 * few.system_throughput_tps


def test_the_stall_does_not_depend_on_the_lengths_being_uniform():
    """Staggered lengths keep decode occupied, which is the harder case.

    With every request the same size the pools empty together, and an empty
    system takes the loop's "nothing can run" branch, which jumps both clocks
    forward and hides the drift. A spread of lengths keeps decode busy across
    the gap, so the drift never gets collected.
    """
    res = _run_split(32, range_ratio=0.8)
    assert res.packing["avg_decode_reqs"] > 16.0


def test_no_request_is_prefilled_before_its_client_submitted_it():
    """Feeding prefill from the simulation clock must not outrun arrival.

    Releasing against the global clock is what fixes the stall, and the risk it
    introduces is the mirror image: serving a request at a time the prefill pool
    has not reached yet, which would surface as a negative queue wait and an
    understated TTFT.
    """
    res = _run_split(8)
    assert res.queue_wait["p50"] >= 0.0
    assert res.queue_wait["p99"] >= 0.0
