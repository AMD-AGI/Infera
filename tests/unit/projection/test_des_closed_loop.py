###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A fixed-concurrency benchmark is a closed loop, and TTFT has to be observed.

The steady-state path prices TTFT as prefill *service* time: how long the
forward pass over the prompt takes. A harness run with ``--max-concurrency C``
reports *response* time instead, and under a colocated engine the two diverge
for a reason that is physics rather than scheduling noise -- a prefill chunk
rides a scheduler step that is simultaneously carrying the resident decodes, so
the step it waits on dilates with the decode batch. Measured TTFT is therefore
close to a constant number of steps across three decades of concurrency, while
a standalone-prefill model holds it constant in milliseconds and reads
progressively early as load rises (0.23-0.40x on the fleet corpus).

``closed_loop_clients`` drives the existing unified-batch step loop from ``C``
clients that resubmit on completion, which makes TTFT the difference between two
simulated timestamps and puts that dilation in the answer for free.

The cost kernel here is a stub, deliberately: these assert the *scheduling*
mechanism, not the model's physics, so the step cost is an explicit function of
batch and prefill tokens and every number below follows from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from infera.projection.core.projection.inference_projection import des as des_mod

# Stub step cost, in ms. Decode dominates once a handful of sequences are
# resident, which is the regime the dilation claim is about.
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
class _Cfg:
    request_config: _Req


class _Kernel:
    """Step cost as a closed form of the step's composition."""

    def decode_step_latency_ms(self, batch, ctx, q_len):
        return _BASE_MS + _PER_SEQ_MS * batch

    def mixed_step_latency_ms(self, num_decode, prefill_tokens, ctx, prefill_kv, q_len):
        return _BASE_MS + _PER_SEQ_MS * num_decode + _PER_PREFILL_TOKEN_MS * prefill_tokens


def _run(concurrency: int, chunk: int = 256, requests_per_client: int = 6, **kw):
    cfg = _Cfg(_Req(max_concurrency=concurrency, chunked_prefill_size=chunk, **kw))
    return des_mod.simulate_once(
        cfg,
        _Kernel(),
        rate_per_s=0.0,
        arrival_model="closed",
        num_requests=concurrency * requests_per_client,
        # Every client starts at once, so the opening burst is a transient the
        # steady-state claims below are not about. The harness does report it,
        # which is a separate finding; here it is dropped.
        warmup_frac=0.5,
        closed_loop_clients=concurrency,
    )


def _decode_step_ms(concurrency: int) -> float:
    return _BASE_MS + _PER_SEQ_MS * concurrency


def test_a_closed_load_holds_every_client_in_flight():
    """The population is the client count -- there is no admission queue.

    Little's law on the fleet corpus said the same thing (running batch 504
    against concurrency 512), which is why the TTFT gap could be ruled out as an
    admission cap and localised to step dilation instead.
    """
    for c in (4, 16, 64):
        res = _run(c)
        assert res.packing["closed_loop_clients"] == float(c)
        # Averaged over the run, near enough every client is resident.
        assert res.packing["avg_batch_size"] > 0.9 * c, (c, res.packing["avg_batch_size"])
        assert res.queue_wait["mean"] < 1e-6


def test_ttft_dilates_with_the_decode_batch():
    """The whole point: TTFT must grow with load, not stay flat in ms."""
    ttfts = [_run(c).ttft["mean"] for c in (4, 16, 64, 256)]
    assert all(later > earlier for earlier, later in zip(ttfts, ttfts[1:])), ttfts
    # And it is a large effect, not a rounding one.
    assert ttfts[-1] > 3.0 * ttfts[0], ttfts


def test_the_growth_is_step_dilation_and_not_an_unbounded_queue():
    """TTFT must grow with load, but no faster than the step it rides.

    The distinction is the finding. Under a closed load the population is capped
    at ``C``, so there is no admission backlog to grow without bound -- what
    grows is the duration of the scheduler step a prefill chunk shares with the
    resident decodes. So TTFT expressed in *steps* stays bounded while TTFT in
    milliseconds climbs, which is the measured shape (13-17 steps from
    concurrency 2 to 512) and the one a standalone-prefill model inverts.
    """
    lo, hi = 16, 256
    ttft_growth = _run(hi).ttft["mean"] / _run(lo).ttft["mean"]
    step_growth = _decode_step_ms(hi) / _decode_step_ms(lo)
    assert ttft_growth > 1.0, ttft_growth
    assert ttft_growth <= step_growth, (ttft_growth, step_growth)


def test_the_declared_prefill_allowance_sets_the_step_count():
    """``chunked_prefill_size`` is the input the exports never recorded.

    It is the per-request per-step prefill allowance, so halving it doubles the
    steps a prompt takes and therefore its TTFT. Solving measured rows for it
    gave 30-300 tokens/step, nowhere near the nominal token budget, which is why
    it has to be declared rather than derived.
    """
    coarse = _run(64, chunk=512).ttft["mean"]
    fine = _run(64, chunk=256).ttft["mean"]
    assert fine > 1.5 * coarse, (coarse, fine)


def test_an_unset_allowance_prefills_in_one_step():
    """Which is the behaviour that reads early, so it must stay the default."""
    one_step = _run(64, chunk=0).ttft["mean"]
    chunked = _run(64, chunk=256).ttft["mean"]
    assert one_step < chunked, (one_step, chunked)


def test_a_single_client_reduces_to_unloaded_prefill():
    """With nothing else resident there is no dilation left to model.

    Four chunks of a 1024-token prompt against a step carrying one sequence.
    """
    res = _run(1, chunk=256, requests_per_client=32)
    expected = 4 * (_BASE_MS + _PER_SEQ_MS * 1 + _PER_PREFILL_TOKEN_MS * 256)
    assert abs(res.ttft["mean"] - expected) < 0.05 * expected, (res.ttft["mean"], expected)


def test_a_closed_load_never_reports_saturation():
    """There is no offered rate to outrun: the population is bounded by C."""
    res = _run(256)
    assert res.saturated is False
    assert res.offered_rate == 0.0
    assert res.achieved_rate > 0.0


def test_the_drain_tail_is_not_charged_to_the_engine():
    """Throughput is rated over the scored window, where the population is full.

    Rating the whole run would include the tail where the last requests finish
    against a shrinking population, which understates throughput by more the
    larger C is.
    """
    res = _run(64)
    # Steady-state decode rate: 64 sequences committing one token per step.
    ideal_tps = 64 * 1000.0 / _decode_step_ms(64)
    assert 0.5 * ideal_tps < res.system_throughput_tps < ideal_tps, (
        res.system_throughput_tps,
        ideal_tps,
    )


def test_every_request_is_served_exactly_once():
    """Client slots must reissue, not duplicate or drop work."""
    res = _run(16, requests_per_client=5)
    assert res.num_requests == 16 * 5


def _trace(tmp_path, n=48, prompt=1024, output=64, shared_blocks=8, block=64):
    """A Mooncake trace whose requests share a leading run of blocks.

    Shared leading hashes are what a multi-turn agentic trace actually looks
    like, and they are the reason the closed-loop composition below has to
    preserve request order rather than rebuild the workload.
    """
    import json

    path = tmp_path / "trace.jsonl"
    with path.open("w") as fh:
        for i in range(n):
            nblocks = prompt // block
            hids = list(range(shared_blocks)) + [
                1000 + i * nblocks + j for j in range(nblocks - shared_blocks)
            ]
            fh.write(
                json.dumps(
                    {
                        "timestamp": i * 1000,
                        "input_length": prompt,
                        "output_length": output,
                        "hash_ids": hids,
                    }
                )
                + "\n"
            )
    return str(path)


def _trace_run(tmp_path, concurrency, closed, **kw):
    cfg = _Cfg(_Req(input_seq_len=4096, max_concurrency=concurrency, chunked_prefill_size=256))
    return des_mod.run_des(
        cfg,
        _Kernel(),
        arrival_model="poisson",
        rate_per_s=1.0,
        num_requests=48,
        warmup_frac=0.0,
        mooncake_trace=_trace(tmp_path, **kw),
        block_size=64,
        num_instances=1,
        routing="kv",
        closed_loop=closed,
    )["point"]


def test_a_traces_cache_hits_survive_being_driven_as_a_closed_loop(tmp_path):
    """The composition that makes a trace answer a capacity question.

    A trace carries three things: per-request lengths, block hashes, and the
    arrival times it was recorded at. The first two are the workload; the third
    is the load, and replaying it at the rate it was captured at can leave an
    engine almost idle. Replacing only the arrivals keeps the workload and asks
    what the configuration can do, and the hit rate is what proves the
    substitution was lossless -- hits come from walking requests through a block
    cache in *order*, and client slots issue in the same order the trace does.
    """
    opened = _trace_run(tmp_path, 8, closed=False)
    closed = _trace_run(tmp_path, 8, closed=True)
    assert opened.prefix["trace_driven"] == 1.0
    assert closed.prefix["trace_driven"] == 1.0
    assert closed.prefix["hit_rate"] == opened.prefix["hit_rate"]
    assert closed.prefix["block_hit_rate"] == opened.prefix["block_hit_rate"]
    # And the hit is real, not a degenerate zero that would match trivially.
    assert closed.prefix["block_hit_rate"] > 0.1
    assert closed.num_requests == opened.num_requests == 48


def test_an_underloaded_replay_leaves_concurrency_inert_and_a_closed_loop_does_not(tmp_path):
    """Why the composition is needed at all, rather than a nicety.

    At one request per second against this stub kernel the engine is never
    asked for more than a couple of concurrent sequences, so every concurrency
    returns the same throughput -- and a matrix that ranks configurations on
    throughput would score them all identically. The closed loop is what puts
    the engine at full utilisation and makes concurrency a real axis.
    """
    flat = [_trace_run(tmp_path, c, closed=False).system_throughput_tps for c in (2, 8, 32)]
    assert max(flat) - min(flat) < 0.01 * max(flat), flat

    live = [_trace_run(tmp_path, c, closed=True).system_throughput_tps for c in (2, 8, 32)]
    assert all(later > earlier for earlier, later in zip(live, live[1:])), live
    assert live[-1] > 3.0 * live[0], live


def test_a_closed_loop_over_a_trace_runs_the_engine_full(tmp_path):
    """Utilisation is the check that the load is no longer the trace's rate."""
    opened = _trace_run(tmp_path, 8, closed=False)
    closed = _trace_run(tmp_path, 8, closed=True)
    assert opened.utilization < 0.5
    assert closed.utilization > 0.95


def test_the_client_count_cannot_exceed_the_requests_the_trace_supplies(tmp_path):
    """Asking for 256 clients from a 48-request trace is 48 clients.

    Left uncapped the extra slots hold requests that never exist, and the run
    reports a concurrency it never reached.
    """
    point = _trace_run(tmp_path, 256, closed=True, n=48)
    assert point.packing["closed_loop_clients"] == 48.0
    assert point.num_requests == 48


def test_run_des_takes_the_client_count_from_the_configured_concurrency():
    """The launcher entry point, which is what ``--des-closed-loop`` reaches.

    There is no offered rate to sweep -- concurrency *is* the load axis here --
    so the sweep is skipped rather than run against a rate of zero.
    """
    cfg = _Cfg(_Req(max_concurrency=32, chunked_prefill_size=256))
    out = des_mod.run_des(
        cfg,
        _Kernel(),
        arrival_model="closed",
        rate_per_s=0.0,
        num_requests=128,
        warmup_frac=0.5,
        sweep=True,
        closed_loop=True,
    )
    point = out["point"]
    assert "curve" not in out
    assert point.arrival_model == "closed"
    assert point.packing["closed_loop_clients"] == 32.0
    assert point.ttft["mean"] > 0.0


def test_exclusive_prefill_reaches_the_path_a_block_cache_dispatches_to():
    """Whether prefill excludes decode has to survive the trip through ``run_des``.

    The mechanism is implemented in ``simulate_once``, but a closed-loop run
    with a block cache to model -- a prefix pool or a trace to replay, which is
    every agentic replay -- is dispatched through ``simulate_multi_instance``,
    and that driver did not take the flag at all. A parameter dropped one frame
    up is indistinguishable from one that was never passed: SGLang and Atom
    replays scheduled a unified batch while the flag that was meant to
    serialise them sat unused, which dissolves the herd a closed-loop
    population forms and reads TTFT early. Assert the effect at the entry point
    the harness calls, not at the loop that implements it.
    """
    clients = 8
    kw = dict(
        arrival_model="closed",
        rate_per_s=0.0,
        num_requests=clients * 8,
        closed_loop=True,
        warmup_frac=0.0,
        # Enough of a prefix pool to put the run on the multi-instance path.
        num_prefixes=4,
        prefix_len=256,
        block_size=16,
        cache_blocks=1 << 14,
    )
    cfg = _Cfg(_Req(max_concurrency=clients, chunked_prefill_size=256))
    unified = des_mod.run_des(cfg, _Kernel(), **kw)["point"]
    exclusive = des_mod.run_des(cfg, _Kernel(), prefill_exclusive=True, **kw)["point"]

    # Serialising prefill against the resident decodes can only make a request
    # wait longer for its first token.
    assert exclusive.ttft["mean"] > unified.ttft["mean"], (
        exclusive.ttft["mean"], unified.ttft["mean"]
    )
