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
