###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Lightweight single-engine discrete-event simulation (DES) for serving.

This is Phase 3 of the inference roadmap: an opt-in event loop that turns the
analytical steady-state projector into a *time-driven* simulation.  It answers
the questions the closed-form model structurally cannot:

  * **percentiles** — p50/p90/p99 of TTFT, TPOT/ITL, and end-to-end latency,
  * **request rate / arrivals** — open-loop Poisson / gamma-bursty /
    deterministic arrivals (or a caller-supplied / file-based workload) with
    real queueing and admission, and
  * **throughput-vs-latency curve** — sweep offered load to find the knee.

**Closed-loop arrivals (``closed_loop_clients``).**  A serving benchmark run
with ``--max-concurrency C`` is not an open-loop stream: ``C`` clients each
submit one request, block until it completes, then immediately submit the next.
Driving the DES that way is what makes **TTFT an observed quantity** — the
difference between two simulated timestamps — rather than a closed-form prefill
service time.  It matters because under a colocated engine a prompt's prefill
chunks ride scheduler steps that are also carrying the resident decodes, so the
step a prefill waits on *dilates with the decode batch*.  Measured TTFT is
therefore roughly a constant number of scheduler steps across three decades of
concurrency, while a model that prices prefill as standalone work holds it
constant in milliseconds and reads progressively early as load rises.  The step
loop below already mixes prefill chunks with decodes, so that dilation is
structural here and needs no separate term.

The per-step prefill allocation is the one input this needs and no export
records: ``chunked_prefill_size`` is the per-request cap on prefill tokens per
step (vLLM's ``long_prefill_token_threshold``), and it sets how many steps a
prompt takes.  It is a *declared* input, not a fitted one — leaving it unset
lets a prompt prefill in a single step, which is the behaviour that reads early.

**Scheduler fidelity (vLLM V1 unified batch).**  The step scheduler mirrors the
``tools/serving_sim`` token-step model: each forward pass first advances every
already-running request (decodes + in-progress prefill chunks) under a shared
``max_num_batched_tokens`` budget (**Phase 1**), then admits new waiting
requests subject to the resident-sequence cap and a **full-ISL KV reservation**
against a flat ``kv_cache_tokens`` pool (**Phase 2**).  Chunked prefill emerges
naturally from the budget; a step can therefore mix prefill chunks and decodes.
Per-request prompt/output lengths are heterogeneous (uniform sampling or a
workload file), and per-step batch composition (query/KV shapes) is recorded.

**The time axis ("benchmark calibration inside a DES").**  On top of that
packing model, every simulated step's *duration* is drawn from the existing
:class:`InferencePerformanceProjector` cost kernel
(:meth:`decode_step_latency_ms` / :meth:`mixed_step_latency_ms`), which is itself
analytical *or* benchmark-calibrated — so the DES turns accurate per-pass costs
into accurate latency **distributions**.  Speculative decoding is modelled as a
per-request draft→verify→commit cycle so accept variance is preserved.

The steady-state model (``arrival_model == "closed"``) remains the validation
path: at low utilisation the DES means should agree with it.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
import random
from dataclasses import dataclass, field

from infera.projection.core.projection.training_config import InferenceConfig

from .performance import InferencePerformanceProjector

# Context-length bucket (tokens) for memoising step-cost kernel calls. Decode
# step latency varies slowly with context, so bucketing keeps the number of
# (expensive) profiler evaluations small while the DES iterates many steps.
_CTX_BUCKET = 256
# Query-token bucket for prefill chunk sizes (heterogeneous per step); keeps the
# mixed-step cost cache small without materially changing the composed cost.
_TOK_BUCKET = 64


def _slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope dy/dx; 0.0 for degenerate input."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / sxx


def _pct(xs: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0, 1]); 0.0 for empty input."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


@dataclass
class _Req:
    """One request's live scheduling state (vLLM-V1 style token accounting)."""

    idx: int
    arrival_ms: float
    prompt_len: int
    output_len: int
    num_computed: int = 0  # prompt tokens already processed (prefill progress)
    prefix_id: int = -1  # shared-prefix identity (-1 = unique / no pool)
    cached_prefix: int = 0  # prompt tokens served from a prefix-cache hit
    blocks: list[int] = field(default_factory=list)  # ordered KV block-hash ids
    generated: int = 0  # output tokens emitted so far
    prefill_done: bool = False
    status: str = "WAITING"  # WAITING | RUNNING | FINISHED
    admit_ms: float = -1.0  # when it left the waiting queue (server admit)
    first_token_ms: float = -1.0
    finish_ms: float = -1.0
    itls: list[float] = field(default_factory=list)
    # Whether the credit below applies. It is a property of *who the sharers
    # are*, not of the engine: see ``reserved_kv``.
    shared_prefix_credit: bool = True

    @property
    def reserved_kv(self) -> int:
        # Full-ISL reservation: the whole sequence is reserved up front, less
        # the leading blocks that are already resident. A prefix-cache hit
        # means those blocks are *in* the pool -- some earlier request put them
        # there and this one attends to the same physical KV -- so charging
        # them again per request double-counts the one copy the engine keeps.
        # On an agentic replay that is not a small correction: at the 96-98%
        # block reuse these corpora carry it inflates a request's footprint
        # 25-50x, so a measured 7.67M-token pool looks full at ~19 concurrent
        # and the replay queues from there, while the MI355X ladder reports its
        # pool at 0.1-0.8% utilization at every concurrency it was run at and
        # never binds at all.
        #
        # The credit is not refcounted: when the request that warmed a block
        # retires, its reservation is released even though the block stays
        # resident for reuse. Both directions are approximations of a
        # refcounted pool, and undercharging shared blocks lands far nearer the
        # measured occupancy than charging every sharer in full. A cold run is
        # untouched -- ``cached_prefix`` is zero without a cache to hit, which
        # is every fixed-sequence workpoint.
        #
        # The credit accounts for one physical copy only where the sharers are
        # resident *at the same time*. On the agentic corpora they are not:
        # reuse there is a conversation hitting its own previous turn, and a
        # closed loop gives each client one turn in flight, so the requests
        # running together are always different conversations whose contexts
        # are disjoint past a short shared head. Credited anyway, C concurrent
        # 200k-token histories cost a few thousand tokens each and the pool
        # never binds at any concurrency these ladders reach -- which is why
        # reuse holds at its ceiling in the replay and degrades on every one
        # of these systems as C approaches what the pool holds.
        if not self.shared_prefix_credit:
            return self.prompt_len + self.output_len
        return self.prompt_len - self.cached_prefix + self.output_len

    @property
    def in_prefill(self) -> bool:
        return not self.prefill_done

    @property
    def kv_len(self) -> int:
        """Context length attended to right now."""
        return self.prompt_len + self.generated if self.prefill_done else self.num_computed


@dataclass
class DESResult:
    arrival_model: str
    offered_rate: float  # req/s requested
    achieved_rate: float  # req/s completed over makespan
    utilization: float  # busy time / makespan
    num_requests: int
    makespan_ms: float
    system_throughput_tps: float  # output tokens/s
    saturated: bool
    # latency distributions (ms)
    ttft: dict[str, float] = field(default_factory=dict)  # from admission (queue-excluded)
    ttft_arrival: dict[str, float] = field(default_factory=dict)  # from arrival (incl. queue wait)
    queue_wait: dict[str, float] = field(default_factory=dict)  # arrival -> admission
    tpot: dict[str, float] = field(default_factory=dict)
    itl: dict[str, float] = field(default_factory=dict)
    e2e: dict[str, float] = field(default_factory=dict)
    # batch-composition / packing summary (serving_sim-style)
    packing: dict[str, float] = field(default_factory=dict)
    # multi-instance routing / prefix-cache summary (only when instances>1 or a
    # prefix pool is configured)
    prefix: dict[str, float] = field(default_factory=dict)
    # optional per-step records (only when record_steps=True)
    steps: list[dict] | None = None
    # optional raw per-request latency samples (only when return_samples=True);
    # used to pool distributions across instances in the multi-instance driver.
    samples: dict[str, list[float]] | None = None


class _CostKernel:
    """Memoised view over the projector's step-cost methods."""

    def __init__(self, projector: InferencePerformanceProjector, q_len: int):
        self._p = projector
        self._q = q_len
        self._decode: dict[tuple, float] = {}
        self._mixed: dict[tuple, float] = {}

    @staticmethod
    def _bucket(ctx: int) -> int:
        return max(_CTX_BUCKET, int(round(ctx / _CTX_BUCKET)) * _CTX_BUCKET)

    @staticmethod
    def _tok(n: int) -> int:
        return max(_TOK_BUCKET, int(round(n / _TOK_BUCKET)) * _TOK_BUCKET)

    def decode_step_ms(self, batch: int, ctx: int) -> float:
        key = (batch, self._bucket(ctx))
        v = self._decode.get(key)
        if v is None:
            v = self._p.decode_step_latency_ms(batch, key[1], self._q)
            self._decode[key] = v
        return v

    def mixed_step_ms(
        self, num_decode: int, prefill_tokens: int, ctx: int, prefill_kv: int
    ) -> float:
        key = (num_decode, self._tok(prefill_tokens), self._bucket(ctx), self._bucket(prefill_kv))
        v = self._mixed.get(key)
        if v is None:
            v = self._p.mixed_step_latency_ms(num_decode, key[1], key[2], key[3], self._q)
            self._mixed[key] = v
            if os.getenv("INFERASIM_DEBUG_DES_STEPS"):
                print(
                    f"[dbg-des] mixed num_decode={key[0]} prefill_tok={key[1]} "
                    f"ctx={key[2]} prefill_kv={key[3]} -> {v:.2f} ms"
                )
        return v


def _resident_cap(
    reqs: list[_Req], kv_cache_tokens: int, max_running: int, enabled: bool = False
) -> int:
    """How many of these requests the KV pool holds at once.

    Length-biased, not the arithmetic mean: a request occupies the pool for a
    time proportional to its own length, so the set resident at any instant is
    sampled in proportion to length and the plain mean overstates how many
    fit. Zero when there is no pool to divide, which leaves the caller's
    ordering policy unconstrained.

    Off unless ``enabled``, because measuring the reordering window against
    this is a modelling choice that did not pay for itself on the AgentX
    corpus. It is the right account of what a scheduler can reorder, and it
    does improve ITL ordering (8 of 11 model-engine pairs at 0.90 or better,
    against 9), but it costs more than that in throughput (8 pairs to 7) --
    reuse at the low rungs falls further than the hardware's did. Kept
    available because the effect it describes is real and the band it applies
    to is narrow; a caller that wants it has to ask.
    """
    if not enabled:
        return 0
    if kv_cache_tokens <= 0 or not reqs:
        return 0
    ctx = [max(1, r.prompt_len + r.output_len) for r in reqs]
    biased = sum(c * c for c in ctx) / max(1, sum(ctx))
    fits = int(kv_cache_tokens / max(1.0, biased))
    return max(1, min(fits, max_running if max_running > 0 else fits))


def _free_cache_blocks(
    kv_cache_tokens: int, live_tokens: float, block_size: int, cap_blocks: int
) -> int:
    """Block-cache capacity left over once the running set has its KV.

    There is one pool. ``_Req.reserved_kv`` already declines to charge a
    request for the leading blocks it hit on, on the grounds that an earlier
    request is holding them -- which is right, and which only balances if
    whatever *is* holding them is charged instead. The block cache is that
    holder, and it was being handed the whole pool at the same time as the
    running set, so both structures booked the same memory and neither ever
    saw it run out.

    The error is invisible wherever the pool is large against the working set
    and total wherever it is not. On the AgentX ladders DeepSeek-V4 has room
    for 44-132 conversations and reuse holds near its ceiling at every
    concurrency run; GLM-5.2 on MI325X has room for 5.5, and the hardware's
    reuse falls 0.79 -> 0.30 -> 0.12 across C=4,5,6 as the live contexts crowd
    the cached prefixes out, taking TTFT from 3s to 198s. Double-booked, the
    replay kept reporting a 0.96 hit rate and a flat TTFT through the whole
    collapse.

    Residency does not depend on reuse -- a hit skips prefill compute, it does
    not free blocks -- so the live figure measured with the cache at one
    capacity is still the live figure at another, and a single corrective pass
    lands on the answer rather than approaching it.

    Never returns zero: ``_BlockStore`` reads a zero capacity as unbounded, and
    a pool with no room left is the opposite of that.
    """
    if kv_cache_tokens <= 0 or block_size <= 0:
        return cap_blocks
    free = max(0, int(kv_cache_tokens) - int(max(0.0, live_tokens)))
    blocks = free // int(block_size)
    if cap_blocks > 0:
        blocks = min(blocks, cap_blocks)
    return max(1, int(blocks))


def _scored_sample(
    done: list[_Req], warmup_frac: float, warmup_requests: int
) -> list[_Req]:
    """The requests whose latencies get reported.

    ``warmup_requests`` drops the opening transient by *issue* order, which is
    the only order that removes it. A closed loop starts every one of its C
    clients at once, so the first C requests queue against each other and wait
    far longer than anything that follows -- a mean of 17 s against a run
    median of 0 on glm5.2 at C=8, and 16% of all the queue wait in the run.
    Dropping a fraction of the *earliest completions* instead, as
    ``warmup_frac`` does, keeps every one of them: a request that waited 40 s
    for a slot is among the last to finish, not the first.

    The harness has the same transient and excludes it the same way -- it
    advances each lane by ``AIPERF_WARMUP_REQUESTS_PER_LANE`` requests, waits
    for them to drain, and only then starts profiling -- so matching it is a
    matter of dropping the same requests rather than a comparable number of
    them. At least half the run is always kept, so a short trace still reports
    over something.
    """
    keep = done
    if warmup_requests > 0:
        cut = min(int(warmup_requests), len(done) // 2)
        keep = [r for r in done if r.idx >= cut]
        if len(keep) >= 8:
            return keep
        keep = done
    drop = int(len(keep) * max(0.0, min(0.9, warmup_frac)))
    return keep[drop:] if len(keep) - drop >= 8 else keep


def _generate_arrivals(
    n: int, rate_per_s: float, model: str, rng: random.Random, burstiness: float = 1.0
) -> list[float]:
    """Arrival timestamps (ms) for ``n`` requests at ``rate_per_s`` req/s.

    ``deterministic`` → fixed spacing; ``poisson`` → gamma inter-arrivals with
    shape ``burstiness`` (1.0 = exponential / Poisson, <1 = burstier, >1 =
    smoother). ``rate_per_s`` non-positive / infinite ⇒ all arrive at t=0.
    """
    if rate_per_s <= 0 or math.isinf(rate_per_s):
        return [0.0] * n
    mean_dt_ms = 1000.0 / rate_per_s
    t = 0.0
    out: list[float] = []
    for _ in range(n):
        if model == "deterministic":
            dt = mean_dt_ms
        else:  # poisson / gamma-bursty
            shape = max(1e-3, burstiness)
            dt = rng.gammavariate(shape, mean_dt_ms / shape)  # mean = mean_dt_ms
        t += dt
        out.append(t)
    return out


def _sample_accepted(rng: random.Random, k: int, accept: float, cap: int) -> int:
    """Tokens committed in one verify step under speculative decoding.

    Bonus token (always) + a run of accepted drafts until the first rejection
    (each accepted independently w.p. ``accept``), capped at ``k+1`` and at the
    remaining ``cap`` tokens. Reproduces the per-step *variable-commit* variance
    that the analytical scalar-expectation model averages away.
    """
    if k <= 0:
        return min(1, cap) if cap > 0 else 1
    accepted = 1
    if accept >= 1.0:
        accepted = k + 1
    else:
        for _ in range(k):
            if rng.random() < accept:
                accepted += 1
            else:
                break
    return max(1, min(accepted, cap if cap > 0 else accepted))


def _sample_len(rng: random.Random, max_len: int, range_ratio: float) -> int:
    """Uniform length in ``[range_ratio*max_len, max_len]`` (inclusive)."""
    max_len = max(1, int(max_len))
    lo = max(1, int(max_len * max(0.0, min(1.0, range_ratio))))
    return rng.randint(lo, max_len)


def _load_workload_file(path: str) -> list[tuple[float, int, int]]:
    """Load ``(arrival_ms, isl, osl)`` rows from JSON (list of dicts) or CSV.

    Keys/columns are case-insensitive: ``arrival`` (ms; aliases arrival_ms/time/
    step), ``isl`` (aliases input_len/prompt_len), ``osl`` (aliases output_len).
    Missing arrival ⇒ 0; missing lengths ⇒ 1.
    """
    if path.endswith(".json"):
        with open(path) as f:
            rows = json.load(f)
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

    def get(row: dict, *names: str, default: float = 0.0) -> float:
        lower = {str(k).lower(): v for k, v in row.items()}
        for nm in names:
            if nm in lower and lower[nm] not in (None, ""):
                return float(lower[nm])
        return default

    out: list[tuple[float, int, int]] = []
    for row in rows:
        arrival = get(row, "arrival", "arrival_ms", "time", "step", default=0.0)
        isl = int(get(row, "isl", "input_len", "prompt_len", default=1.0))
        osl = int(get(row, "osl", "output_len", default=1.0))
        out.append((float(arrival), max(1, isl), max(1, osl)))
    return out


def _build_workload(
    n: int,
    arrivals: list[float],
    input_len: int,
    output_len: int,
    range_ratio: float,
    rng: random.Random,
) -> list[_Req]:
    """Construct ``n`` requests with per-request sampled lengths (homogeneous
    when ``range_ratio >= 1``)."""
    reqs: list[_Req] = []
    for i in range(n):
        isl = _sample_len(rng, input_len, range_ratio)
        osl = _sample_len(rng, output_len, range_ratio)
        reqs.append(_Req(idx=i, arrival_ms=arrivals[i], prompt_len=isl, output_len=osl))
    return reqs


def simulate_once(
    inference_config: InferenceConfig,
    projector: InferencePerformanceProjector,
    *,
    rate_per_s: float,
    arrival_model: str = "poisson",
    num_requests: int = 400,
    warmup_frac: float = 0.1,
    warmup_requests: int = 0,
    seed: int = 0,
    arrivals: list[float] | None = None,
    burstiness: float = 1.0,
    range_ratio: float = 1.0,
    kv_cache_tokens: int = 0,
    workload_file: str | None = None,
    record_steps: bool = False,
    prebuilt: list[_Req] | None = None,
    return_samples: bool = False,
    closed_loop_clients: int = 0,
    closed_loop_think_ms: float = 0.0,
    whole_context_residency: bool = False,
    prefill_exclusive: bool = False,
    new_seqs_per_step: int = 0,
    duration_ms: float = 0.0,
) -> DESResult:
    """Run one single-engine DES at a fixed offered load.

    vLLM-V1 unified-batch scheduler: each forward pass advances running requests
    (Phase 1) then admits waiting ones under the resident-sequence cap + full-ISL
    KV reservation (Phase 2), all sharing a per-step ``max_num_batched_tokens``
    budget; each step's duration comes from the (optionally benchmark-calibrated)
    cost kernel.

    ``closed_loop_clients > 0`` replaces the arrival stream with that many
    clients, each of which resubmits the moment its previous request finishes,
    reproducing a fixed-concurrency benchmark harness. All ``C`` start at once,
    which is what the harness does and what makes its *mean* TTFT carry an
    opening-burst transient; ``warmup_frac`` decides whether that transient is
    scored, so match it to whatever the harness reports over.

    ``duration_ms`` stops the run on the clock instead of on a request count,
    which is how a fixed-concurrency harness is actually bounded: it runs for
    a set wall time and reports whatever completed. The distinction is not
    cosmetic under a closed loop. A request budget split across ``C`` lanes
    fixes how far *each lane* walks into its conversation -- 400 turns deep at
    C=1 against 40 at C=16 for the same budget -- and an agentic turn's prompt
    grows with its position in the conversation, so a budget-bounded replay
    offers systematically different prompt lengths at each concurrency than
    the run it is being compared against. A clock-bounded one lets the lane
    depth fall out of how fast the engine actually is, which is the same thing
    that decided it on the hardware.
    """
    req = inference_config.request_config
    input_len = max(1, req.input_seq_len)
    output_len = max(1, req.output_seq_len)
    max_running = max(1, req.resolved_max_concurrency())
    token_budget = int(req.max_num_batched_tokens or 0)  # 0 = unlimited
    long_prefill = int(req.chunked_prefill_size or 0)  # per-request chunk cap
    max_model_len = max(2, int(req.resolved_max_context_len()))
    spec_k = int(req.speculative_num_tokens or 0)
    accept = float(req.speculative_acceptance_rate or 0.0)
    q_len = (spec_k + 1) if spec_k > 0 else 1
    kv_pool = int(kv_cache_tokens or 0)  # 0 = unlimited

    rng = random.Random(seed)
    clients = max(0, int(closed_loop_clients))
    closed_loop = clients > 0
    think_ms = max(0.0, float(closed_loop_think_ms))

    # ---- workload (arrivals + per-request lengths) ----
    if closed_loop:
        # No arrival process: a client slot releases the next request when its
        # previous one retires. Everything past the first ``clients`` is issued
        # from the retirement handler below, so its arrival is unknown up front.
        #
        # A caller-supplied request list is kept and only its arrivals are
        # replaced. That is what lets a trace's real lengths and its
        # content-addressed cache hits be measured at a fixed concurrency
        # instead of at the rate the trace happens to have been recorded at: an
        # under-loaded replay leaves concurrency inert and every configuration
        # tied on throughput. The hits survive the substitution because
        # ``_route_and_warm`` derives them from request *order*, which the
        # retirement handler preserves -- it issues ``pending`` by index -- and
        # not from arrival timestamps.
        if prebuilt is not None:
            pending = prebuilt
            clients = min(clients, len(pending))
        else:
            n_total = max(clients, int(num_requests))
            pending = _build_workload(
                n_total, [0.0] * n_total, input_len, output_len, range_ratio, rng
            )
        for i, r in enumerate(pending):
            r.arrival_ms = 0.0 if i < clients else math.inf
    elif prebuilt is not None:
        # Caller supplied a fully-formed request list (multi-instance router:
        # arrivals, lengths, prefix ids + seeded ``num_computed`` for hits).
        pending = prebuilt
    elif workload_file:
        rows = _load_workload_file(workload_file)
        rows.sort(key=lambda r: r[0])
        pending = [
            _Req(idx=i, arrival_ms=a, prompt_len=isl, output_len=osl)
            for i, (a, isl, osl) in enumerate(rows)
        ]
    else:
        if arrivals is None:
            arrivals = _generate_arrivals(num_requests, rate_per_s, arrival_model, rng, burstiness)
        pending = _build_workload(len(arrivals), arrivals, input_len, output_len, range_ratio, rng)
    if prebuilt is None:
        # Seed the flat prefix-cache hit rate. The multi-instance router derives
        # each request's hit from a content-addressed block cache and hands the
        # result over in ``prebuilt``; the single-engine path had no equivalent,
        # so ``--prefix-cache-hit-rate`` was accepted and then ignored and every
        # request reprefilled its whole prompt. On an agentic shape that is the
        # difference between a 10k prefill and a 130k one, which showed up as the
        # DES saturating at 0.08 req/s where the analytical path sustained 1.78.
        # Same rule as ``_prefix_cached_tokens``: at least one token stays
        # uncached so a fully-cached prompt still runs a forward to emit token 1.
        hit = inference_config.request_config.resolved_prefix_cache_hit_rate()
        if hit > 0.0:
            for r in pending:
                cached = max(0, min(int(r.prompt_len * hit), r.prompt_len - 1))
                if cached > 0:
                    r.cached_prefix = cached
                    r.num_computed = cached
    if whole_context_residency:
        for r in pending:
            r.shared_prefix_credit = False
    pending.sort(key=lambda r: (r.arrival_ms, r.idx))
    n = len(pending)

    kernel = _CostKernel(projector, q_len)
    next_arrival = 0
    # Closed loop only: the first request no client slot has issued yet. Every
    # index below it carries a finite arrival, which is what lets the ingest
    # pointer above stay a simple in-order scan.
    next_unissued = clients if closed_loop else n
    waiting: list[_Req] = []
    running: list[_Req] = []
    done: list[_Req] = []
    kv_used = 0

    now = 0.0
    busy_ms = 0.0
    backlog: list[tuple] = []  # (time_ms, waiting depth) for saturation test
    step_records: list[dict] = [] if record_steps else []
    # packing accumulators
    pk_steps = pk_batch = pk_maxbatch = pk_prefill_reqs = pk_decode_reqs = 0
    pk_qtokens = 0
    pk_prefill_steps = 0
    pk_kv_peak = 0
    pk_kv_sum = 0

    # Exact worst-case iteration bound (batch=1: every token its own step). Both
    # the per-request chunk cap and the shared token budget can split a prefill,
    # so the bound has to follow whichever is tighter.
    pf_cap = min(
        long_prefill if long_prefill > 0 else math.inf,
        token_budget if token_budget > 0 else math.inf,
    )
    total_work = 0
    for r in pending:
        pf = 1 if math.isinf(pf_cap) else max(1, math.ceil(r.prompt_len / pf_cap))
        total_work += pf + r.output_len
    max_steps = total_work + n + 16

    steps = 0
    horizon = duration_ms if duration_ms and duration_ms > 0 else math.inf
    while len(done) < n and steps < max_steps and now < horizon:
        steps += 1
        # 1) Ingest arrivals due by ``now`` into the FCFS waiting queue.
        while next_arrival < n and pending[next_arrival].arrival_ms <= now + 1e-9:
            waiting.append(pending[next_arrival])
            next_arrival += 1

        # 2) Nothing resident and nothing waiting → jump to the next arrival.
        if not running and not waiting:
            if next_arrival < n and math.isfinite(pending[next_arrival].arrival_ms):
                now = pending[next_arrival].arrival_ms
                continue
            break

        # 3) Build the step: Phase 1 (running) then Phase 2 (admit), sharing the
        #    per-step token budget. ``scheduled`` = (req, q, is_prefill, kv_start).
        budget = token_budget if token_budget > 0 else math.inf
        scheduled: list[tuple[_Req, int, bool, int]] = []

        def _schedule_tokens(r: _Req, budget: float) -> int:
            if r.prefill_done:
                # decode: compute q_len speculative tokens (>=1) if not finished
                if r.generated >= r.output_len:
                    return 0
                return int(min(q_len, budget)) if budget >= 1 else 0
            need = r.prompt_len - r.num_computed
            if need <= 0:
                return 0
            if long_prefill > 0:
                need = min(need, long_prefill)
            need = min(need, budget)
            need = min(need, max_model_len - 1 - r.num_computed)
            return int(max(need, 0))

        # Phase 0 — exclusive prefill. Some engines do not co-schedule prefill
        # with decode: a prefill batch takes one request and the whole token
        # budget, and the resident decodes wait. Atom logs exactly that for
        # every prefill it ran on MI355X ("Scheduled prefill batch: 1 reqs,
        # 8192 new tokens" against a budget of 8192, 1270 times out of 1270),
        # and the consequence is not a small one. Because prefill blocks
        # decode, a closed-loop population re-synchronises every round: nobody
        # advances while one request prefills, so all C clients start decoding
        # together, finish together, and resubmit together. The queue that
        # forms is therefore a standing one rather than an opening transient,
        # which is what makes measured TTFT uniform on [0, 2*median] -- the
        # signature the MI355X ladder shows at every concurrency (std/mean
        # 0.564 against 0.577 for a uniform, p99/median 1.96 against 1.98).
        # Chunked co-scheduling dissolves that herd and reads TTFT an order of
        # magnitude early at high concurrency.
        # A prefill step excludes decode but is not limited to one request: it
        # packs as many as the token budget holds, which is why the same engine
        # logs "1 reqs, 8192 new tokens" for 8192-token prompts and "8 reqs,
        # 8192" for 1024-token ones. Serialising one per step regardless would
        # make short prompts queue C-deep when they in fact clear in C*ISL/budget
        # steps.
        # How many *new* sequences may enter this step. The token budget is an
        # upper bound on a prefill batch, not a target: engines admit far fewer
        # requests than would fill it, because every admission reserves KV for a
        # prompt whose length they must assume in full, and over-admitting risks
        # having to retract a running request later. Across 5.4M prefill batches
        # logged by vLLM and SGLang on MI355X, the median batch takes 1 new
        # sequence with an empty queue and 2 with a backlog (mean 2.7, p90 4) --
        # never the 16 that a 16384-token budget would hold at ISL 1024.
        #
        # It changes TTFT twice over, which is why the budget alone read it an
        # order of magnitude early at high concurrency. A wave of C requests
        # needs C/admit prefill batches instead of C*ISL/budget, and each batch
        # is small enough to run at a much worse token rate: those same logs put
        # a 1024-token prefill batch at 9.2k tok/s against 42.7k for a
        # 16384-token one. Both effects push real TTFT up.
        admit_cap = new_seqs_per_step if new_seqs_per_step > 0 else (1 << 30)
        admitted = 0

        prefill_step = False
        if prefill_exclusive:
            for r in running:
                if r.prefill_done or budget < 1:
                    continue
                q = _schedule_tokens(r, budget)
                if q > 0:
                    scheduled.append((r, q, True, r.kv_len))
                    budget -= q
                    prefill_step = True
            while waiting and len(running) < max_running and budget >= 1 and admitted < admit_cap:
                head = waiting[0]
                if kv_pool > 0 and kv_used + head.reserved_kv > kv_pool:
                    break
                q = _schedule_tokens(head, budget)
                if q <= 0:
                    break
                waiting.pop(0)
                head.status = "RUNNING"
                head.admit_ms = now
                running.append(head)
                kv_used += head.reserved_kv
                scheduled.append((head, q, True, head.kv_len))
                budget -= q
                admitted += 1
                prefill_step = True

        # Phase 1 — already-running requests.
        for r in [] if prefill_step else running:
            q = _schedule_tokens(r, budget)
            if q <= 0:
                continue
            scheduled.append((r, q, r.in_prefill, r.kv_len))
            budget -= q

        # Phase 2 — admit new waiting requests (full-ISL KV reservation gate).
        # Under exclusive prefill, admission is Phase 0's job: admitting here
        # would start a second prefill in a step that is meant to hold one.
        while (
            not prefill_exclusive
            and waiting
            and len(running) < max_running
            and budget >= 1
            and admitted < admit_cap
        ):
            cand = waiting[0]
            if kv_pool > 0 and kv_used + cand.reserved_kv > kv_pool:
                break  # head-of-line block until KV frees up
            q = _schedule_tokens(cand, budget)
            if q <= 0:
                break
            waiting.pop(0)
            cand.status = "RUNNING"
            cand.admit_ms = now
            running.append(cand)
            kv_used += cand.reserved_kv
            scheduled.append((cand, q, True, cand.kv_len))
            budget -= q
            admitted += 1

        if not scheduled:
            # Budget/KV starved this step with nothing runnable; advance to the
            # next arrival if possible, else we are stuck (bound will trip).
            if next_arrival < n:
                now = pending[next_arrival].arrival_ms
                continue
            break

        # 4) Step duration from the cost kernel (composition → time).
        pref = [(r, q, kv) for (r, q, p, kv) in scheduled if p]
        dec = [(r, q, kv) for (r, q, p, kv) in scheduled if not p]
        num_decode = len(dec)
        prefill_q = sum(q for _, q, _ in pref)
        if prefill_q > 0:
            prefill_kv = int(sum(kv + q for _, q, kv in pref) / len(pref))
            decode_ctx = int(sum(kv for _, _, kv in dec) / len(dec)) if dec else input_len
            step_dt = kernel.mixed_step_ms(num_decode, prefill_q, decode_ctx, prefill_kv)
        else:
            decode_ctx = int(sum(kv for _, _, kv in dec) / len(dec)) if dec else input_len
            step_dt = kernel.decode_step_ms(num_decode, decode_ctx)

        # 5) Advance the clock.
        now += step_dt
        busy_ms += step_dt

        # 6) Apply the forward pass (prefill progress; decode commits w/ spec).
        for r, q, is_prefill, _kv in scheduled:
            if not r.prefill_done:
                r.num_computed += q
                if r.num_computed >= r.prompt_len:
                    r.prefill_done = True
                    r.first_token_ms = now
                    r.generated = 1  # last prefill chunk emits token 1
                    r.itls.append(step_dt)
                    if r.generated >= r.output_len:
                        r.status = "FINISHED"
                        r.finish_ms = now
            else:
                cap = r.output_len - r.generated
                if cap <= 0:
                    continue
                acc = _sample_accepted(rng, spec_k, accept, cap)
                r.generated += acc
                per_tok = step_dt / acc
                r.itls.extend([per_tok] * acc)
                if r.generated >= r.output_len:
                    r.status = "FINISHED"
                    r.finish_ms = now

        # 7) Retire finished requests, free their KV reservation. Under a closed
        #    load the freed client slot immediately submits its next request, so
        #    the population stays at ``clients`` rather than draining.
        still: list[_Req] = []
        for r in running:
            if r.status == "FINISHED":
                kv_used -= r.reserved_kv
                done.append(r)
                if closed_loop and next_unissued < n:
                    pending[next_unissued].arrival_ms = now + think_ms
                    next_unissued += 1
            else:
                still.append(r)
        running = still

        # 8) Bookkeeping: waiting depth + packing stats (+ optional records).
        backlog.append((now, len(waiting)))
        bs = len(scheduled)
        pk_steps += 1
        pk_batch += bs
        pk_maxbatch = max(pk_maxbatch, bs)
        pk_prefill_reqs += len(pref)
        pk_decode_reqs += num_decode
        pk_qtokens += prefill_q + num_decode * q_len
        pk_prefill_steps += 1 if pref else 0
        pk_kv_peak = max(pk_kv_peak, kv_used)
        pk_kv_sum += kv_used
        if record_steps:
            step_records.append(
                {
                    "step": pk_steps,
                    "time_ms": round(now, 4),
                    "step_ms": round(step_dt, 4),
                    "batch_size": bs,
                    "num_prefill_reqs": len(pref),
                    "num_decode_reqs": num_decode,
                    "total_query_tokens": prefill_q + num_decode * q_len,
                    "kv_tokens_in_use": kv_used,
                    "requests": [
                        {
                            "request_id": r.idx,
                            "phase": "prefill" if is_prefill else "decode",
                            "query_len": q,
                            "kv_len": kv,
                        }
                        for (r, q, is_prefill, kv) in scheduled
                    ],
                }
            )

    # ---- aggregate latency metrics (drop the opening transient) ----
    done.sort(key=lambda r: r.finish_ms)
    sample = _scored_sample(done, warmup_frac, warmup_requests)

    # TTFT is measured from *admission* (server start), not arrival, so the
    # client-side wait for a concurrency slot is excluded -- matching the vLLM /
    # InferenceX serving harness, whose TTFT clock (``st``) starts after the
    # request acquires its semaphore. The queue wait and the arrival-relative
    # TTFT are reported separately (queue_wait, ttft_arrival) so the deployment
    # view is not lost.
    # Host prompt-tokenization cost (per prompt token). The prompt is sent as
    # text and tokenized server-side after the TTFT clock starts, so it lands in
    # TTFT. Latency-only -- the server scheduler/makespan above are unchanged.
    tok_ms_pt = max(0.0, req.tokenize_overhead_us) / 1000.0

    def _admit(r):
        return r.admit_ms if r.admit_ms >= 0 else r.arrival_ms

    ttft = [
        (r.first_token_ms - _admit(r)) + tok_ms_pt * r.prompt_len
        for r in sample
        if r.first_token_ms >= 0
    ]
    ttft_arrival = [
        (r.first_token_ms - r.arrival_ms) + tok_ms_pt * r.prompt_len
        for r in sample
        if r.first_token_ms >= 0
    ]
    queue_wait = [_admit(r) - r.arrival_ms for r in sample if r.first_token_ms >= 0]
    # Per-output-token detokenization + streaming (client-side host cost).
    # Latency-only: added to ITL/TPOT/e2e but not to the server step, so the
    # scheduler, makespan and throughput above are unchanged. Matches the
    # serving harness, which measures ITL client-side.
    detok_ms = max(0.0, req.detokenize_overhead_us) / 1000.0
    e2e = [
        (r.finish_ms - r.arrival_ms) + tok_ms_pt * r.prompt_len + detok_ms * r.generated
        for r in sample
        if r.finish_ms >= 0
    ]
    tpot = [
        (r.finish_ms - r.first_token_ms) / max(1, r.generated - 1) + detok_ms
        for r in sample
        if r.finish_ms >= 0 and r.generated > 1
    ]
    itl_all: list[float] = []
    for r in sample:
        itl_all.extend(x + detok_ms for x in r.itls)

    makespan = max((r.finish_ms for r in done), default=0.0)
    total_out = sum(r.generated for r in done)
    if closed_loop and len(sample) >= 2:
        # The tail of a closed run drains: once the last request is issued the
        # population falls below ``clients`` and throughput with it. Rating the
        # whole run would charge that drain against the engine, so the rate is
        # taken over the scored sample's own span, where the population is full.
        span_ms = max(r.finish_ms for r in sample) - min(r.arrival_ms for r in sample)
        out_sample = sum(r.generated for r in sample)
        achieved_rate = (len(sample) * 1000.0 / span_ms) if span_ms > 0 else 0.0
        sys_tps = (out_sample * 1000.0 / span_ms) if span_ms > 0 else 0.0
    else:
        achieved_rate = (len(done) * 1000.0 / makespan) if makespan > 0 else 0.0
        sys_tps = (total_out * 1000.0 / makespan) if makespan > 0 else 0.0
    utilization = (busy_ms / makespan) if makespan > 0 else 0.0

    # Saturation: the waiting queue diverges (grows ~linearly) while arrivals are
    # still coming, rather than staying stationary. Restricted to the arrival
    # window because a finite run always drains to empty afterwards. N-invariant,
    # unlike achieved-vs-offered rate (drain-tail biased) or utilisation (~1
    # whenever ≥1 request is resident, common for memory-bound decode).
    # A closed load has no offered rate to outrun -- the population is bounded by
    # the client count -- so the test does not apply and the flag stays clear.
    saturated = False
    last_arrival = pending[-1].arrival_ms if pending else 0.0
    win = [(t, float(d)) for (t, d) in backlog if t <= last_arrival] if rate_per_s > 0 else []
    if rate_per_s > 0 and len(win) >= 30:
        cut = int(0.2 * len(win))
        bs_win = win[cut:]
        ts = [t for t, _ in bs_win]
        ds = [d for _, d in bs_win]
        span = ts[-1] - ts[0]
        if span > 0:
            growth = _slope(ts, ds) * span
            saturated = growth > max_running and max(ds) > max_running
    if rate_per_s > 0 and achieved_rate < 0.5 * rate_per_s:
        saturated = True

    def dist(xs: list[float]) -> dict[str, float]:
        return {
            "mean": (sum(xs) / len(xs)) if xs else 0.0,
            "p50": _pct(xs, 0.50),
            "p90": _pct(xs, 0.90),
            "p99": _pct(xs, 0.99),
        }

    packing = {
        "num_steps": float(pk_steps),
        "avg_batch_size": (pk_batch / pk_steps) if pk_steps else 0.0,
        "max_batch_size": float(pk_maxbatch),
        "avg_prefill_reqs": (pk_prefill_reqs / pk_steps) if pk_steps else 0.0,
        "avg_decode_reqs": (pk_decode_reqs / pk_steps) if pk_steps else 0.0,
        "avg_query_tokens": (pk_qtokens / pk_steps) if pk_steps else 0.0,
        "prefill_step_fraction": (pk_prefill_steps / pk_steps) if pk_steps else 0.0,
        "kv_peak_tokens": float(pk_kv_peak),
        "kv_utilization": (pk_kv_peak / kv_pool) if kv_pool > 0 else 0.0,
        # Occupancy averaged over steps, not the high-water mark. The peak is
        # set by the opening transient: the cache is cold, nothing is a hit,
        # and the first batch reserves whole prompts, so the peak pins to the
        # pool on any run whose cold working set exceeds it and says nothing
        # about the run that follows. What the cache has to live alongside is
        # the steady occupancy.
        "kv_mean_tokens": (pk_kv_sum / pk_steps) if pk_steps else 0.0,
        "closed_loop_clients": float(clients),
    }

    samples = None
    if return_samples:
        samples = {
            "ttft": ttft,
            "ttft_arrival": ttft_arrival,
            "queue_wait": queue_wait,
            "tpot": tpot,
            "itl": itl_all,
            "e2e": e2e,
        }

    return DESResult(
        arrival_model=arrival_model,
        offered_rate=rate_per_s,
        achieved_rate=achieved_rate,
        utilization=utilization,
        num_requests=len(done),
        makespan_ms=makespan,
        system_throughput_tps=sys_tps,
        saturated=saturated,
        ttft=dist(ttft),
        ttft_arrival=dist(ttft_arrival),
        queue_wait=dist(queue_wait),
        tpot=dist(tpot),
        itl=dist(itl_all),
        e2e=dist(e2e),
        packing=packing,
        steps=step_records if record_steps else None,
        samples=samples,
    )


_ROUTING_POLICIES = ("round_robin", "random", "prefix_aware", "kv")

# Default KV block size (tokens per paged block). 512 matches the Mooncake
# trace convention; also used to blockify synthetic shared prefixes.
_DEFAULT_BLOCK_SIZE = 512


class _BlockCache:
    """Per-instance paged-KV block store (content-addressed, leaf-first eviction).

    Models engine-style automatic prefix caching: a prompt is an ordered
    sequence of block-hash ids; a **hit** is the longest *contiguous prefix* of
    that sequence already resident (prefix caching only reuses a leading run of
    matching blocks). This is the reuse store the KV-aware router scores routes
    against, and what a single engine hits across a sequential stream.

    Eviction is leaf-first, least-recently-used among the leaves, which is what
    a radix prefix cache does -- a block cannot be dropped while a resident
    block continues from it, so the shared head of a corpus survives pressure
    and only the divergent tails are reclaimed.

    Modelling it as a flat LRU instead was not a smaller approximation of that,
    it was the LRU pathology. Blocks are touched in prefix order, so a corpus
    whose working set exceeds the pool is a cyclic sweep, and a cyclic sweep
    evicts every block exactly before it is reused. Replaying the AgentX c256
    trace against DeepSeek-V4's real 231,336-block pool, flat LRU returned a
    0.0% hit rate over 10.5M evictions -- not a degraded rate, every single
    access -- where the same trace and pool under leaf-first eviction return
    90.6%, against the corpus's own published reuse of ~92%. Downstream that
    was throughput at 0.08x of measured on MI355X/SGLang at C=256, and a
    throughput rank correlation of 0.11 on a curve that otherwise orders at
    0.93.

    Blocks are content-addressed over their prefix, so a block id implies its
    predecessor and the parent recorded on first insert is stable.
    """

    def __init__(self, capacity_blocks: int = 0) -> None:
        self.capacity = int(capacity_blocks or 0)  # 0 = unbounded
        self._parent: dict[int, int | None] = {}
        self._children: dict[int, int] = {}  # resident children; 0 => a leaf
        self._access: dict[int, int] = {}
        # Candidate leaves by access time. Entries go stale when a block is
        # touched again or stops being a leaf, so they are filtered on pop
        # rather than removed in place.
        self._leaves: list[tuple[int, int]] = []
        self._tick = 0
        self.evictions = 0

    def prefix_match(self, blocks: list[int]) -> int:
        """Number of leading blocks already resident (contiguous from the head)."""
        m = 0
        for b in blocks:
            if b in self._children:
                m += 1
            else:
                break
        return m

    def insert(self, blocks: list[int]) -> None:
        """Warm a request's blocks, then reclaim leaves beyond capacity."""
        prev: int | None = None
        for b in blocks:
            self._tick += 1
            if b in self._children:
                self._access[b] = self._tick
                # Only leaves are eviction candidates, so only a leaf's touch
                # needs re-filing; an interior block is not reclaimable anyway.
                if self._children[b] == 0:
                    heapq.heappush(self._leaves, (self._tick, b))
            else:
                self._parent[b] = prev
                self._children[b] = 0
                self._access[b] = self._tick
                if prev is not None and prev in self._children:
                    self._children[prev] += 1
                heapq.heappush(self._leaves, (self._tick, b))
            prev = b
        self._evict()

    def _evict(self) -> None:
        if self.capacity <= 0:
            return
        while len(self._children) > self.capacity:
            victim = None
            while self._leaves:
                t, b = heapq.heappop(self._leaves)
                if b not in self._children:
                    continue  # already reclaimed
                if self._access[b] != t:
                    continue  # touched since; a newer entry stands
                if self._children[b]:
                    continue  # no longer a leaf
                victim = b
                break
            if victim is None:
                # Every resident block is interior. Nothing is reclaimable
                # without orphaning a continuation, which is the one thing a
                # radix cache will not do.
                return
            p = self._parent.pop(victim)
            del self._children[victim]
            del self._access[victim]
            self.evictions += 1
            if p is not None and p in self._children:
                self._children[p] -= 1
                if self._children[p] == 0:
                    heapq.heappush(self._leaves, (self._access[p], p))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._children)


class _BlockHasher:
    """Deterministic ``tuple -> small int`` block-hash id allocator."""

    def __init__(self) -> None:
        self._m: dict[tuple, int] = {}

    def hid(self, key: tuple) -> int:
        h = self._m.get(key)
        if h is None:
            h = len(self._m)
            self._m[key] = h
        return h


def _blocks_from_prefix(
    idx: int,
    prompt_len: int,
    prefix_id: int,
    prefix_len: int,
    block_size: int,
    hasher: _BlockHasher,
) -> list[int]:
    """Blockify a synthetic prompt: leading blocks shared by same-``prefix_id``
    requests, trailing blocks unique to this request.

    Reproduces shared-prefix reuse at block granularity so the block cache sees
    the same overlap a real workload would, without needing token content.
    """
    bs = max(1, block_size)
    n_blocks = max(1, math.ceil(prompt_len / bs))
    n_shared = (
        min(n_blocks, math.ceil(prefix_len / bs)) if (prefix_id >= 0 and prefix_len > 0) else 0
    )
    blocks = [hasher.hid(("P", prefix_id, b)) for b in range(n_shared)]
    blocks += [hasher.hid(("U", idx, b)) for b in range(n_blocks - n_shared)]
    return blocks


def _load_mooncake_trace(path: str) -> list[tuple[float, int, int, list[int]]]:
    """Load a Mooncake-format trace: ``(arrival_ms, isl, osl, hash_ids)`` rows.

    Accepts JSON-lines (one object per line) or a JSON array. Each record uses
    ``timestamp`` (ms), ``input_length``, ``output_length`` and ``hash_ids`` (the
    ordered list of block-hash ids for the prompt). The block hashes drive
    content-addressed prefix-cache matching directly -- consecutive requests that
    share a system prompt share leading ``hash_ids``.
    """
    records: list[dict] = []
    with open(path) as f:
        text = f.read()
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(stripped)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    def _g(r: dict, *names: str, default=0):
        low = {str(k).lower(): v for k, v in r.items()}
        for nm in names:
            if nm in low and low[nm] not in (None, ""):
                return low[nm]
        return default

    rows: list[tuple[float, int, int, list[int]]] = []
    for r in records:
        ts = float(_g(r, "timestamp", "arrival", "arrival_ms", "time", default=0.0))
        isl = int(_g(r, "input_length", "isl", "input_len", "prompt_len", default=1))
        osl = int(_g(r, "output_length", "osl", "output_len", default=1))
        hids = _g(r, "hash_ids", "block_hashes", "blocks", default=[]) or []
        rows.append((ts, max(1, isl), max(1, osl), [int(x) for x in hids]))
    return rows


def _draw_prefix_ids(n: int, num_prefixes: int, zipf: float, rng: random.Random) -> list[int]:
    """Assign each of ``n`` requests a shared-prefix id in ``[0, num_prefixes)``.

    ``zipf <= 0`` → uniform popularity; ``zipf > 0`` → power-law skew (a few
    hot prefixes dominate, as with shared system prompts / popular templates).
    """
    if num_prefixes <= 0:
        return [-1] * n
    if zipf and zipf > 0:
        weights = [1.0 / ((k + 1) ** zipf) for k in range(num_prefixes)]
        total = sum(weights)
        cum, acc = [], 0.0
        for w in weights:
            acc += w / total
            cum.append(acc)
        out = []
        for _ in range(n):
            u = rng.random()
            lo = 0
            for k, c in enumerate(cum):
                if u <= c:
                    lo = k
                    break
            else:
                lo = num_prefixes - 1
            out.append(lo)
        return out
    return [rng.randrange(num_prefixes) for _ in range(n)]


# Per-pick decay on each instance's recent-miss total, and the load charged for
# a request carrying no block information. Both mirror the serving router, whose
# policy module explains why each value is what it is.
_KV_RECENT_DECAY = 0.97
_KV_UNKNOWN_COST_BLOCKS = 1.0


def _route_and_warm(
    reqs: list[_Req],
    *,
    policy: str,
    num_instances: int,
    block_size: int,
    cache_blocks: int,
    rng: random.Random,
    overlap_weight: float = 1.0,
    waiting_depth: int = 0,
    resident_cap: int = 0,
) -> tuple[list[list[_Req]], dict[str, float]]:
    """Route requests across instances and derive per-request prefix-cache hits
    from a content-addressed block cache (as real serving engines do).

    Each instance owns a :class:`_BlockCache`. For each request the router picks
    a target, the number of resident leading blocks (longest contiguous prefix
    match) becomes the cache hit -- its ``cached_prefix`` tokens are seeded into
    ``num_computed`` so the scheduler only prefills the uncached suffix -- and
    the request's blocks are then warmed into that instance (evicted under
    ``cache_blocks`` capacity).

    Routing policies: ``kv`` trades cache overlap against load by the serving
    router's own cost function, with ``overlap_weight`` as the dial between them
    (``0`` routes purely by load); ``prefix_aware`` consistently hashes the
    leading block so same-prefix requests co-locate; ``round_robin``/``random``
    ignore locality.

    ``waiting_depth`` is how many requests the engine has outstanding at once,
    and it sets the order they are resolved in. With it unset they are resolved
    oldest-first, which is only the order an engine admits in while everything
    offered fits; once the waiting queue is deep a scheduler chooses from it by
    longest resident prefix rather than by age (SGLang's default policy, and
    what vLLM's prefix-caching scheduler approximates). The distinction does
    not matter below the pressure point -- with room for every outstanding
    context both orders touch the same blocks -- but above it the two diverge
    completely: oldest-first walks the offered lanes round-robin, so a lane's
    context is always evicted before its next turn and every request pays a
    full reprefill, while prefix-first keeps working on what is already
    resident and reuse falls off gradually instead of to zero.

    ``resident_cap`` is how many of those requests fit in the KV pool at once,
    and it is what the reordering window has to be measured against: a
    scheduler can only reorder requests that are actually queued. Below the
    cap nothing waits, so there is no choice to make and admission is just the
    order the lanes arrived in -- which is the regime where reuse degrades,
    because C lanes all resident at once evict each other. Handing the policy
    the full client count instead let it reorder an empty queue and kept reuse
    high everywhere: on kimik3 at C=14 the hardware thrashed to 0.67 and the
    replay reported 0.93, and the band where that happens is exactly where C
    meets the cap. Above the cap the queue is genuinely deep, prefix-first is
    the real policy, and measured reuse recovers -- 0.92 by C=44 -- which the
    replay only gets right if the window grows with the queue and not with C.
    """
    per_inst: list[list[_Req]] = [[] for _ in range(num_instances)]
    caches = [_BlockCache(cache_blocks) for _ in range(num_instances)]
    # Decayed history of blocks each instance had to compute. The router's load
    # term also counts in-flight blocks, which are unknowable here (routing is
    # decided before the simulation runs) and are 0 for everyone anyway whenever
    # a request finishes before the next is picked.
    recent = [0.0] * num_instances
    hits = 0
    cached_total = 0
    blocks_total = 0
    blocks_hit = 0
    inst_hits = [0] * num_instances
    inst_reqs = [0] * num_instances

    ordered = sorted(reqs, key=lambda x: (x.arrival_ms, x.idx))

    def _admission_order():
        """The order the engine takes requests off its waiting queue.

        A closed loop holds one request per client, so the stream is the
        clients interleaved: client ``i`` owns arrival positions ``i``,
        ``i + depth``, ``i + 2 * depth``. Serving a request frees its own
        client, and the turn that client issues next is the one continuing the
        context just served -- the whole reason its prefix is worth keeping.
        Refilling the slot from a global pointer instead hands it to some other
        client and quietly turns the loop into a sliding window over arrival
        order, which throws that reuse away.
        """
        # Only the backlog is reorderable: what fits in the pool is already
        # running and was admitted in the order it arrived.
        depth = int(waiting_depth or 0) - max(0, int(resident_cap or 0))
        if depth <= 1:
            yield from ordered
            return
        window = list(range(min(depth, len(ordered))))
        while window:
            # Longest resident prefix anywhere in the fleet: the scheduler is
            # choosing what to run next, not where to run it, so the routing
            # decision below is still the router's to make.
            pick = max(
                window,
                key=lambda i: max(c.prefix_match(ordered[i].blocks or []) for c in caches),
            )
            window.remove(pick)
            if pick + depth < len(ordered):
                window.append(pick + depth)
            yield ordered[pick]

    for r in _admission_order():
        blocks = r.blocks or []
        if num_instances <= 1:
            inst = 0
        elif policy == "kv":
            # The serving router's cost function, so the simulated fleet splits
            # traffic the way the real one does:
            #   cost(i) = overlap_weight * (blocks - hits(i)) + load(i)
            best_i, best_cost = 0, None
            for i in range(num_instances):
                miss = len(blocks) - caches[i].prefix_match(blocks)
                cost = overlap_weight * miss + recent[i]
                if best_cost is None or cost < best_cost:
                    best_i, best_cost = i, cost
            inst = best_i
        elif policy == "prefix_aware":
            key = blocks[0] if blocks else r.idx
            inst = key % num_instances
        elif policy == "random":
            inst = rng.randrange(num_instances)
        else:  # round_robin
            inst = r.idx % num_instances

        inst_reqs[inst] += 1
        matched = caches[inst].prefix_match(blocks)
        if policy == "kv":
            # Charging misses rather than whole requests is what lets load
            # coexist with affinity: an instance serving a fully-cached prompt
            # accrues nothing and keeps winning it, while one handed a cold
            # prompt accrues it and the next cold prompt goes elsewhere.
            for i in range(num_instances):
                recent[i] *= _KV_RECENT_DECAY
            recent[inst] += float(len(blocks) - matched) if blocks else _KV_UNKNOWN_COST_BLOCKS
        cached = max(0, min(matched * block_size, r.prompt_len - 1))
        if cached > 0:
            r.cached_prefix = cached
            r.num_computed = cached
            hits += 1
            inst_hits[inst] += 1
            cached_total += cached
        blocks_total += len(blocks)
        blocks_hit += matched
        caches[inst].insert(blocks)
        per_inst[inst].append(r)

    # Resolving prefixes in schedule order says how much each request has to
    # prefill; it does not say when each one arrived. The per-instance loops
    # below are arrival-driven, so hand them back the stream they expect.
    for sub in per_inst:
        sub.sort(key=lambda x: (x.arrival_ms, x.idx))

    n = len(reqs)
    summary = {
        "num_instances": float(num_instances),
        "num_prefixes": float(max(0, max((r.prefix_id for r in reqs), default=-1) + 1)),
        "prefix_len": float(cached_total / max(1, hits)) if hits else 0.0,
        "block_size": float(block_size),
        "cache_blocks": float(cache_blocks),
        "evictions": float(sum(c.evictions for c in caches)),
        "block_hit_rate": (blocks_hit / blocks_total) if blocks_total else 0.0,
        "hit_rate": (hits / n) if n else 0.0,
        "avg_cached_tokens": (cached_total / n) if n else 0.0,
        "min_inst_hit_rate": min(
            (inst_hits[i] / inst_reqs[i]) for i in range(num_instances) if inst_reqs[i]
        )
        if any(inst_reqs)
        else 0.0,
        "max_inst_hit_rate": max(
            (inst_hits[i] / inst_reqs[i]) for i in range(num_instances) if inst_reqs[i]
        )
        if any(inst_reqs)
        else 0.0,
    }
    return per_inst, summary


def _aggregate_instances(results: list[DESResult], prefix_summary: dict[str, float]) -> DESResult:
    """Pool per-instance DESResults into one fleet-level DESResult.

    Latency distributions are recomputed from the pooled raw samples; system
    throughput / achieved rate sum across instances; makespan is the slowest
    instance; utilization is the per-instance mean.

    Packing is averaged across instances rather than dropped. Every field in it
    describes one engine's steps -- batch size, prefill-step share, KV
    utilisation, client count -- so a fleet mean is the meaningful pooling and a
    single instance passes through unchanged.
    """

    def _pool(key: str) -> list[float]:
        out: list[float] = []
        for r in results:
            if r.samples and r.samples.get(key):
                out.extend(r.samples[key])
        return out

    def dist(xs: list[float]) -> dict[str, float]:
        return {
            "mean": (sum(xs) / len(xs)) if xs else 0.0,
            "p50": _pct(xs, 0.50),
            "p90": _pct(xs, 0.90),
            "p99": _pct(xs, 0.99),
        }

    n_inst = max(1, len(results))
    makespan = max((r.makespan_ms for r in results), default=0.0)
    packing: dict[str, float] = {}
    for key in {k for r in results for k in (r.packing or {})}:
        vals = [float(r.packing[key]) for r in results if r.packing and key in r.packing]
        packing[key] = sum(vals) / len(vals) if vals else 0.0
    return DESResult(
        arrival_model=results[0].arrival_model if results else "poisson",
        offered_rate=sum(r.offered_rate for r in results),
        achieved_rate=sum(r.achieved_rate for r in results),
        utilization=sum(r.utilization for r in results) / n_inst,
        num_requests=sum(r.num_requests for r in results),
        makespan_ms=makespan,
        system_throughput_tps=sum(r.system_throughput_tps for r in results),
        saturated=any(r.saturated for r in results),
        ttft=dist(_pool("ttft")),
        ttft_arrival=dist(_pool("ttft_arrival")),
        queue_wait=dist(_pool("queue_wait")),
        tpot=dist(_pool("tpot")),
        itl=dist(_pool("itl")),
        e2e=dist(_pool("e2e")),
        packing=packing,
        prefix=prefix_summary,
    )


def simulate_disaggregated(
    inference_config: InferenceConfig,
    projector: InferencePerformanceProjector,
    *,
    rate_per_s: float,
    arrival_model: str = "poisson",
    num_requests: int = 400,
    warmup_frac: float = 0.1,
    warmup_requests: int = 0,
    seed: int = 0,
    burstiness: float = 1.0,
    range_ratio: float = 1.0,
    kv_cache_tokens: int = 0,
    prebuilt: list[_Req] | None = None,
    closed_loop_clients: int = 0,
    closed_loop_think_ms: float = 0.0,
    return_samples: bool = False,
    duration_ms: float = 0.0,
) -> DESResult:
    """Run a prefill pool and a decode pool as two stations on one clock.

    The single-engine loop above cannot represent this topology, and did not try
    to: it has no notion of pools, so a disaggregated candidate handed to it was
    scheduled as one unified batch and priced with the parent's parallelism. The
    result was a colocated schedule wearing disaggregated step costs -- a number
    for a deployment that does not exist -- which is why the topology was kept
    out of the trace-driven results rather than reported from that path.

    Two things have to be true at once for the simulation to mean anything, and
    they pull against each other:

    * **The pools do not share a step.** Moving prefill off the decode GPUs is
      the entire argument for splitting, so no decode batch here ever carries a
      prefill chunk. That is the property the colocated path spends 6% of its
      steps and a third of its TPOT on.
    * **The pools do not share GPUs either.** They run *concurrently* on
      disjoint hardware. Time-sharing one engine between prefill-only and
      decode-only steps would also produce no mixed steps, and would be wrong
      in the other direction -- it serialises work that overlaps in the real
      deployment, inflating TTFT and TPOT together. So each station keeps its
      own clock, and they advance independently.

    A request therefore traverses: arrival, prefill queue, prefill batches on
    the prefill pool (whose last chunk emits token 1), a KV handoff charged per
    request, then a decode queue and pure decode steps on one replica of the
    decode pool. Each station is priced by its own pool's projector, so a TP4
    prefill pool beside two TP2 decode replicas costs what those shapes cost.

    Under a closed load the loop spans *both* stations: a client slot is
    released when its request retires from decode, not from prefill. That is
    what makes this a single event loop rather than two sequential replays --
    running the stations one after the other cannot represent the feedback,
    because each one's arrivals are the other's completions.
    """
    from .performance import _replica_gpus

    req = inference_config.request_config
    input_len = max(1, req.input_seq_len)
    output_len = max(1, req.output_seq_len)
    max_running = max(1, req.resolved_max_concurrency())
    token_budget = int(req.max_num_batched_tokens or 0)  # 0 = unlimited
    long_prefill = int(req.chunked_prefill_size or 0)
    max_model_len = max(2, int(req.resolved_max_context_len()))
    spec_k = int(req.speculative_num_tokens or 0)
    accept = float(req.speculative_acceptance_rate or 0.0)
    q_len = (spec_k + 1) if spec_k > 0 else 1

    disagg = inference_config.disaggregation_config
    replicas = max(1, int(getattr(disagg, "decode_replicas", 1) or 1))
    p_replicas = max(1, int(getattr(disagg, "prefill_replicas", 1) or 1))
    prefill_proj, decode_proj = projector.pool_projectors()
    p_kernel = _CostKernel(prefill_proj, q_len)
    d_kernel = _CostKernel(decode_proj, q_len)
    p_gpus = _replica_gpus(prefill_proj.cfg) * p_replicas
    d_gpus = _replica_gpus(decode_proj.cfg) * replicas

    rng = random.Random(seed)
    clients = max(0, int(closed_loop_clients))
    closed_loop = clients > 0
    think_ms = max(0.0, float(closed_loop_think_ms))

    # ---- workload ----
    if prebuilt is not None:
        pending = prebuilt
    else:
        arrivals = (
            [0.0] * num_requests
            if closed_loop
            else _generate_arrivals(num_requests, rate_per_s, arrival_model, rng, burstiness)
        )
        pending = _build_workload(len(arrivals), arrivals, input_len, output_len, range_ratio, rng)
    n = len(pending)
    if not n:
        return DESResult(
            arrival_model=arrival_model,
            offered_rate=rate_per_s,
            achieved_rate=0.0,
            utilization=0.0,
            num_requests=0,
            makespan_ms=0.0,
            system_throughput_tps=0.0,
            saturated=False,
        )
    if closed_loop:
        # All C clients start at once and the rest are issued on retirement, so
        # their arrival is unknown up front. Ordering is preserved, which is what
        # keeps a trace's content-addressed cache hits intact.
        clients = min(clients, n)
        for i, r in enumerate(pending):
            r.arrival_ms = 0.0 if i < clients else math.inf
        next_unissued = clients
    else:
        pending.sort(key=lambda r: r.arrival_ms)
        next_unissued = n

    # The KV pool is per *pool*, not per fleet. A prefill pool holds a prompt
    # only until the handoff, while a decode replica holds the whole sequence
    # for the generation's duration, so splitting the budget evenly would
    # mis-state both. Each station is given the pool it would actually have.
    pool_tokens = int(kv_cache_tokens or 0)
    p_pool = pool_tokens  # 0 = unlimited
    d_pool = pool_tokens

    p_waiting: list[_Req] = []
    p_running: list[_Req] = []
    p_kv = 0
    now_p = 0.0
    p_busy = 0.0
    xfer: list[tuple[float, _Req]] = []  # (ready_ms, req) in flight over the link
    d_waiting: list[list[_Req]] = [[] for _ in range(replicas)]
    d_running: list[list[_Req]] = [[] for _ in range(replicas)]
    d_kv = [0] * replicas
    now_d = [0.0] * replicas
    d_busy = [0.0] * replicas
    done: list[_Req] = []
    handoff: dict[int, float] = {}
    next_arrival = 0
    p_steps = 0
    d_steps = 0
    pk_p_batch = 0
    pk_d_batch = 0
    pk_q_tokens = 0
    pk_maxbatch = 0
    pk_kv_peak = 0
    backlog: list[tuple[float, int]] = []

    def _prefill_tokens(r: _Req, budget: float) -> int:
        need = r.prompt_len - r.num_computed
        if need <= 0:
            return 0
        if long_prefill > 0:
            need = min(need, long_prefill)
        need = min(need, budget)
        need = min(need, max_model_len - 1 - r.num_computed)
        return int(max(need, 0))

    def _release_arrivals(t: float) -> int:
        nonlocal next_arrival
        got = 0
        while next_arrival < n and pending[next_arrival].arrival_ms <= t:
            p_waiting.append(pending[next_arrival])
            next_arrival += 1
            got += 1
        return got

    def _deliver_handoffs(t: float) -> int:
        """KV that has landed by ``t`` joins the least-loaded decode replica."""
        ready = [(ts, r) for (ts, r) in xfer if ts <= t]
        if not ready:
            return 0
        xfer[:] = [(ts, r) for (ts, r) in xfer if ts > t]
        for ts, r in sorted(ready, key=lambda x: x[0]):
            k = min(range(replicas), key=lambda i: (len(d_running[i]) + len(d_waiting[i]), i))
            d_waiting[k].append(r)
            now_d[k] = max(now_d[k], ts)
        return len(ready)

    def _step_prefill() -> bool:
        """One prefill-only batch on the prefill pool."""
        nonlocal now_p, p_kv, p_busy, p_steps, pk_p_batch, pk_maxbatch, pk_kv_peak
        nonlocal pk_q_tokens
        # An idle prefill pool advances to meet the next request instead of
        # holding its clock where its last batch left it. The two stations keep
        # separate clocks and in a split it is decode that leads: a closed-loop
        # client retires on the decode clock and submits its replacement stamped
        # with that time, so a prefill pool that has drained its queue is behind
        # by however long it sat idle. The decode station already has this rule
        # -- ``_deliver_handoffs`` moves ``now_d`` up to the handoff it is given
        # -- and without the match here every request reissued while prefill was
        # idle stays invisible to it until its clock crawls forward a step at a
        # time, which is what collapsed a 32-client closed loop to one request
        # in flight and pinned the decode batch at 1.
        if not p_running and p_waiting:
            now_p = max(now_p, p_waiting[0].arrival_ms)
        budget: float = token_budget if token_budget > 0 else math.inf
        scheduled: list[tuple[_Req, int, int]] = []
        for r in p_running:
            if budget < 1:
                break
            q = _prefill_tokens(r, budget)
            if q > 0:
                scheduled.append((r, q, r.num_computed))
                budget -= q
        while p_waiting and budget >= 1 and len(p_running) < max_running:
            head = p_waiting[0]
            if head.arrival_ms > now_p:
                # Queued against the simulation clock but not yet submitted on
                # this station's, so it waits rather than being prefilled before
                # the client asked for it.
                break
            # A prefill pool reserves the prompt it is about to compute; it hands
            # the KV off and frees it, so it is not charged for the generation.
            need_kv = head.prompt_len
            if p_pool > 0 and p_kv + need_kv > p_pool:
                break
            q = _prefill_tokens(head, budget)
            if q <= 0:
                break
            p_waiting.pop(0)
            head.status = "RUNNING"
            head.admit_ms = now_p
            p_running.append(head)
            p_kv += need_kv
            pk_kv_peak = max(pk_kv_peak, p_kv)
            scheduled.append((head, q, head.num_computed))
            budget -= q
        if not scheduled:
            return False
        prefill_q = sum(q for _, q, _ in scheduled)
        avg_kv = int(sum(kv + q for _, q, kv in scheduled) / len(scheduled))
        # num_decode=0: this station never co-schedules a decode, by construction.
        dt = p_kernel.mixed_step_ms(0, prefill_q, input_len, avg_kv)
        now_p += dt
        p_busy += dt
        p_steps += 1
        pk_p_batch += len(scheduled)
        pk_q_tokens += prefill_q
        pk_maxbatch = max(pk_maxbatch, len(scheduled))
        finished: list[_Req] = []
        for r, q, _kv in scheduled:
            r.num_computed += q
            if r.num_computed >= r.prompt_len:
                r.prefill_done = True
                r.first_token_ms = now_p  # the last prefill chunk emits token 1
                r.generated = 1
                r.itls.append(dt)
                finished.append(r)
        for r in finished:
            p_running.remove(r)
            p_kv -= r.prompt_len
            if r.generated >= r.output_len:
                # A one-token request never reaches the decode pool.
                r.status = "FINISHED"
                r.finish_ms = now_p
                done.append(r)
                if closed_loop:
                    _reissue(now_p)
                continue
            ms = projector.kv_handoff_ms(decode_proj, r.prompt_len)
            handoff[r.idx] = ms
            xfer.append((now_p + ms, r))
        return True

    def _step_decode(k: int) -> bool:
        """One pure decode step on replica ``k``."""
        nonlocal d_steps, pk_d_batch, pk_maxbatch, pk_kv_peak, pk_q_tokens
        while d_waiting[k] and len(d_running[k]) < max_running:
            head = d_waiting[k][0]
            need_kv = head.reserved_kv
            if d_pool > 0 and d_kv[k] + need_kv > d_pool:
                break
            d_waiting[k].pop(0)
            head.status = "RUNNING"
            d_running[k].append(head)
            d_kv[k] += need_kv
            pk_kv_peak = max(pk_kv_peak, d_kv[k])
        if not d_running[k]:
            return False
        ctx = int(sum(r.kv_len for r in d_running[k]) / len(d_running[k]))
        dt = d_kernel.decode_step_ms(len(d_running[k]), ctx)
        now_d[k] += dt
        d_busy[k] += dt
        d_steps += 1
        pk_d_batch += len(d_running[k])
        pk_q_tokens += len(d_running[k]) * q_len
        pk_maxbatch = max(pk_maxbatch, len(d_running[k]))
        still: list[_Req] = []
        for r in d_running[k]:
            cap = r.output_len - r.generated
            if cap > 0:
                acc = _sample_accepted(rng, spec_k, accept, cap)
                r.generated += acc
                r.itls.extend([dt / acc] * acc)
            if r.generated >= r.output_len:
                r.status = "FINISHED"
                r.finish_ms = now_d[k]
                d_kv[k] -= r.reserved_kv
                done.append(r)
                if closed_loop:
                    _reissue(now_d[k])
            else:
                still.append(r)
        d_running[k] = still
        return True

    def _reissue(t: float) -> None:
        """A retired request frees its client slot, which submits the next one.

        The slot is released on *decode* retirement. Releasing it at the handoff
        would let a client hold two requests at once and report a concurrency
        the harness never ran.
        """
        nonlocal next_unissued
        if next_unissued < n:
            pending[next_unissued].arrival_ms = t + think_ms
            next_unissued += 1

    # ---- event loop ----
    # A station that cannot move is held out until something changes rather than
    # re-picked at the same clock, which would spin the loop forever: both step
    # functions decline to advance their own clock when they schedule nothing,
    # so "stuck" and "idle" look identical from outside. It happens for real --
    # a prompt longer than the model's context never finishes prefilling, and a
    # request whose reservation exceeds the whole KV pool is never admitted --
    # and the single-engine loop handles the same case by breaking out.
    stalled_p = False
    stalled_d = [False] * replicas

    def _unstall() -> None:
        nonlocal stalled_p
        stalled_p = False
        for i in range(replicas):
            stalled_d[i] = False

    guard = 0
    bound = 4000 * max(1, n)
    horizon = duration_ms if duration_ms and duration_ms > 0 else math.inf
    while guard < bound:
        guard += 1
        # Both stations are fed against the simulation clock rather than the
        # prefill station's own. Gating arrivals on ``now_p`` hid every request
        # a client reissued at a decode-clock time from the pool that has to
        # prefill it.
        now_any = max([now_p, *now_d])
        if now_any >= horizon:
            break
        moved = _release_arrivals(now_any) + _deliver_handoffs(now_any)
        if moved:
            _unstall()

        runnable: list[tuple[float, int]] = []
        if (p_running or p_waiting) and not stalled_p:
            runnable.append((now_p, -1))
        for k in range(replicas):
            if (d_running[k] or d_waiting[k]) and not stalled_d[k]:
                runnable.append((now_d[k], k))
        if not runnable:
            # Nothing resident anywhere: jump to the next thing that can happen.
            nxt = [
                t
                for t in (
                    pending[next_arrival].arrival_ms if next_arrival < n else None,
                    min((ts for ts, _ in xfer), default=None),
                )
                if t is not None and t < math.inf
            ]
            if not nxt:
                break
            t = min(nxt)
            now_p = max(now_p, t)
            for k in range(replicas):
                now_d[k] = max(now_d[k], t)
            continue
        _t, who = min(runnable)
        did = _step_prefill() if who < 0 else _step_decode(who)
        if did:
            # Any progress can unblock the other station -- a retiring decode
            # frees the KV a waiting prefill needs, and vice versa.
            _unstall()
        elif who < 0:
            stalled_p = True
        else:
            stalled_d[who] = True
        backlog.append((now_p, len(p_waiting)))

    # ---- aggregate ----
    done.sort(key=lambda r: r.finish_ms)
    sample = _scored_sample(done, warmup_frac, warmup_requests)
    tok_ms_pt = max(0.0, req.tokenize_overhead_us) / 1000.0
    detok_ms = max(0.0, req.detokenize_overhead_us) / 1000.0

    def _admit(r):
        return r.admit_ms if r.admit_ms >= 0 else r.arrival_ms

    ttft = [
        (r.first_token_ms - _admit(r)) + tok_ms_pt * r.prompt_len
        for r in sample
        if r.first_token_ms >= 0
    ]
    ttft_arrival = [
        (r.first_token_ms - r.arrival_ms) + tok_ms_pt * r.prompt_len
        for r in sample
        if r.first_token_ms >= 0
    ]
    queue_wait = [_admit(r) - r.arrival_ms for r in sample if r.first_token_ms >= 0]
    e2e = [
        (r.finish_ms - r.arrival_ms) + tok_ms_pt * r.prompt_len + detok_ms * r.generated
        for r in sample
        if r.finish_ms >= 0
    ]
    # TPOT spans token 1 to the last, so it carries the KV handoff and any wait
    # for a decode slot. Both are real delays between two tokens the client
    # sees, and pricing them anywhere else would hide the cost of the handoff.
    tpot = [
        (r.finish_ms - r.first_token_ms) / max(1, r.generated - 1) + detok_ms
        for r in sample
        if r.finish_ms >= 0 and r.generated > 1
    ]
    itl_all: list[float] = []
    for r in sample:
        itl_all.extend(x + detok_ms for x in r.itls)

    makespan = max((r.finish_ms for r in done), default=0.0)
    total_out = sum(r.generated for r in done)
    if closed_loop and len(sample) >= 2:
        span_ms = max(r.finish_ms for r in sample) - min(r.arrival_ms for r in sample)
        out_sample = sum(r.generated for r in sample)
        achieved_rate = (len(sample) * 1000.0 / span_ms) if span_ms > 0 else 0.0
        sys_tps = (out_sample * 1000.0 / span_ms) if span_ms > 0 else 0.0
    else:
        achieved_rate = (len(done) * 1000.0 / makespan) if makespan > 0 else 0.0
        sys_tps = (total_out * 1000.0 / makespan) if makespan > 0 else 0.0

    # Fleet utilisation, GPU-weighted. An unweighted mean of the two stations
    # would rate a 4-GPU prefill pool and a 12-GPU decode pool as equals.
    total_gpus = max(1, p_gpus + d_gpus)
    busy_gpu_ms = p_busy * p_gpus + sum(d_busy) * (d_gpus / replicas)
    utilization = (busy_gpu_ms / (makespan * total_gpus)) if makespan > 0 else 0.0

    saturated = False
    if rate_per_s > 0 and achieved_rate < 0.5 * rate_per_s:
        saturated = True

    def dist(xs: list[float]) -> dict[str, float]:
        return {
            "mean": (sum(xs) / len(xs)) if xs else 0.0,
            "p50": _pct(xs, 0.50),
            "p90": _pct(xs, 0.90),
            "p99": _pct(xs, 0.99),
        }

    steps = p_steps + d_steps
    packing = {
        "num_steps": float(steps),
        "avg_batch_size": ((pk_p_batch + pk_d_batch) / steps) if steps else 0.0,
        "max_batch_size": float(pk_maxbatch),
        # Per *station*: a prefill batch and a decode batch are different
        # populations here, unlike the unified loop where one step holds both.
        "avg_prefill_reqs": (pk_p_batch / p_steps) if p_steps else 0.0,
        "avg_decode_reqs": (pk_d_batch / d_steps) if d_steps else 0.0,
        "avg_query_tokens": (pk_q_tokens / steps) if steps else 0.0,
        # Every prefill step is prefill-only and every decode step is
        # decode-only, so there is no mixed step to report a fraction of. This
        # is the topology's defining property, not a missing measurement.
        "prefill_step_fraction": (p_steps / steps) if steps else 0.0,
        "mixed_step_fraction": 0.0,
        "kv_peak_tokens": float(pk_kv_peak),
        "kv_utilization": (pk_kv_peak / pool_tokens) if pool_tokens > 0 else 0.0,
        "closed_loop_clients": float(clients),
        "disaggregated": 1.0,
        "prefill_utilization": (p_busy / makespan) if makespan > 0 else 0.0,
        "decode_utilization": ((sum(d_busy) / replicas / makespan) if makespan > 0 else 0.0),
        "kv_handoff_ms": (sum(handoff.values()) / len(handoff) if handoff else 0.0),
        "prefill_pool_gpus": float(p_gpus),
        "decode_pool_gpus": float(d_gpus),
    }

    samples = None
    if return_samples:
        samples = {
            "ttft": ttft,
            "ttft_arrival": ttft_arrival,
            "queue_wait": queue_wait,
            "tpot": tpot,
            "itl": itl_all,
            "e2e": e2e,
        }

    return DESResult(
        arrival_model="closed" if closed_loop else arrival_model,
        offered_rate=rate_per_s,
        achieved_rate=achieved_rate,
        utilization=utilization,
        num_requests=len(done),
        makespan_ms=makespan,
        system_throughput_tps=sys_tps,
        saturated=saturated,
        ttft=dist(ttft),
        ttft_arrival=dist(ttft_arrival),
        queue_wait=dist(queue_wait),
        tpot=dist(tpot),
        itl=dist(itl_all),
        e2e=dist(e2e),
        packing=packing,
        samples=samples,
    )


def simulate_multi_instance(
    inference_config: InferenceConfig,
    projector: InferencePerformanceProjector,
    *,
    rate_per_s: float,
    arrival_model: str,
    num_requests: int,
    seed: int,
    warmup_frac: float,
    warmup_requests: int,
    burstiness: float,
    range_ratio: float,
    kv_cache_tokens: int,
    num_instances: int,
    routing: str,
    overlap_weight: float,
    num_prefixes: int,
    prefix_len: int,
    prefix_zipf: float,
    block_size: int,
    cache_blocks: int,
    admit_backlog_only: bool = False,
    mooncake_rows: list[tuple[float, int, int, list[int]]] | None = None,
    closed_loop_clients: int = 0,
    duration_ms: float = 0.0,
    prefill_exclusive: bool = False,
    cache_shares_pool: bool = False,
    closed_loop_think_ms: float = 0.0,
    whole_context_residency: bool = False,
) -> DESResult:
    """Route one arrival stream across ``num_instances`` replicas and pool.

    Prefix-cache hits are *derived* from a content-addressed block cache + the
    routing policy (not a static rate): ``kv`` / ``prefix_aware`` routing
    co-locate requests that share leading KV blocks on one instance, so a prefix
    is warmed once and reused; ``round_robin`` / ``random`` scatter them, forcing
    every instance to re-warm (more cold misses, higher TTFT). Block sequences
    come from a Mooncake trace (``mooncake_rows``) or are synthesised from the
    shared-prefix pool.
    """
    req = inference_config.request_config
    input_len = max(1, req.input_seq_len)
    output_len = max(1, req.output_seq_len)
    rng = random.Random(seed)
    bs = int(block_size) if block_size and block_size > 0 else _DEFAULT_BLOCK_SIZE
    hasher = _BlockHasher()

    def _trace_requests() -> list[_Req]:
        rows = sorted(mooncake_rows or [], key=lambda x: x[0])
        return [
            _Req(idx=i, arrival_ms=a, prompt_len=isl, output_len=osl, blocks=list(hids))
            for i, (a, isl, osl, hids) in enumerate(rows)
        ]

    if mooncake_rows is not None:
        # Trace-driven: arrivals, lengths and block hashes all come from the file.
        reqs = _trace_requests()
    else:
        arrivals = _generate_arrivals(num_requests, rate_per_s, arrival_model, rng, burstiness)
        reqs = _build_workload(len(arrivals), arrivals, input_len, output_len, range_ratio, rng)
        # A prefix pool with ``prefix_len == 0`` defaults to half the prompt.
        eff_prefix_len = (
            prefix_len if prefix_len > 0 else (input_len // 2 if num_prefixes > 0 else 0)
        )
        for r, pid in zip(reqs, _draw_prefix_ids(len(reqs), num_prefixes, prefix_zipf, rng)):
            r.prefix_id = pid
            r.blocks = _blocks_from_prefix(r.idx, r.prompt_len, pid, eff_prefix_len, bs, hasher)

    def _warm_and_run(pool_reqs: list[_Req], cap_blocks: int) -> DESResult:
        per_inst, prefix_summary = _route_and_warm(
            pool_reqs,
            policy=routing,
            num_instances=num_instances,
            block_size=bs,
            cache_blocks=cap_blocks,
            rng=random.Random(seed),
            overlap_weight=overlap_weight,
            waiting_depth=closed_loop_clients,
            resident_cap=_resident_cap(
                pool_reqs,
                kv_cache_tokens,
                inference_config.request_config.resolved_max_concurrency(),
                admit_backlog_only,
            ),
        )
        prefix_summary["routing"] = (
            float(_ROUTING_POLICIES.index(routing)) if routing in _ROUTING_POLICIES else -1.0
        )
        prefix_summary["trace_driven"] = 1.0 if mooncake_rows is not None else 0.0

        results: list[DESResult] = []
        for i, sub in enumerate(per_inst):
            if not sub:
                continue
            # Per-instance offered rate ≈ its share of the global stream.
            inst_rate = rate_per_s * (len(sub) / len(pool_reqs)) if pool_reqs else rate_per_s
            results.append(
                simulate_once(
                    inference_config,
                    projector,
                    rate_per_s=inst_rate,
                    arrival_model=arrival_model,
                    seed=seed + i,
                    warmup_frac=warmup_frac,
                    warmup_requests=warmup_requests,
                    kv_cache_tokens=kv_cache_tokens,
                    prebuilt=sub,
                    return_samples=True,
                    # Concurrency is per engine, so every replica runs the full
                    # client count rather than a share of it.
                    closed_loop_clients=closed_loop_clients,
                    closed_loop_think_ms=closed_loop_think_ms,
                    whole_context_residency=whole_context_residency,
                    # Every replica stops on the same clock: the window is the
                    # harness's, so it is not divided across instances the way
                    # the request stream is.
                    duration_ms=duration_ms,
                    # Whether prefill excludes decode is a property of the engine,
                    # so every replica schedules the same way.
                    prefill_exclusive=prefill_exclusive,
                )
            )
        return _aggregate_instances(results, prefix_summary)

    agg = _warm_and_run(reqs, cache_blocks)
    # The cache and the running set are the same memory, and the warm pass runs
    # before there is a running set to measure. So measure one, then re-warm
    # against what it left free. Residency is independent of reuse, so the
    # second pass is the answer and not a step towards it.
    if cache_shares_pool and mooncake_rows is not None and kv_cache_tokens > 0:
        live = float((agg.packing or {}).get("kv_mean_tokens") or 0.0)
        free_blocks = _free_cache_blocks(kv_cache_tokens, live, bs, cache_blocks)
        if free_blocks != cache_blocks:
            agg = _warm_and_run(_trace_requests(), free_blocks)
            agg.prefix["cache_shares_pool"] = 1.0
            agg.prefix["cache_free_tokens"] = float(max(0.0, kv_cache_tokens - live))
    return agg


def run_des(
    inference_config: InferenceConfig,
    projector: InferencePerformanceProjector,
    *,
    arrival_model: str,
    rate_per_s: float,
    num_requests: int = 400,
    seed: int = 0,
    warmup_frac: float = 0.1,
    warmup_requests: int = 0,
    sweep: bool = False,
    burstiness: float = 1.0,
    range_ratio: float = 1.0,
    kv_cache_tokens: int = 0,
    workload_file: str | None = None,
    record_steps: bool = False,
    num_instances: int = 1,
    routing: str = "round_robin",
    overlap_weight: float = 1.0,
    num_prefixes: int = 0,
    prefix_len: int = 0,
    prefix_zipf: float = 0.0,
    cache_slots: int = 0,
    block_size: int = 0,
    cache_blocks: int = 0,
    mooncake_trace: str | None = None,
    duration_ms: float = 0.0,
    admit_backlog_only: bool = False,
    prefill_exclusive: bool = False,
    new_seqs_per_step: int = 0,
    closed_loop: bool = False,
    closed_loop_think_ms: float = 0.0,
    cache_shares_pool: bool = False,
    whole_context_residency: bool = False,
) -> dict[str, object]:
    """Run the DES at the configured load and (optionally) a load sweep.

    Returns ``{"point": DESResult, "curve": [DESResult, ...],
    "max_sustainable_rate": mu}``. The sweep derives the engine's max-sustainable
    rate ``mu`` from the steady-state projection and samples fractions of it to
    trace the throughput-vs-latency knee.

    ``closed_loop`` runs the fixed-concurrency form instead: the resolved
    concurrency becomes a client count, there is no offered rate, and the load
    sweep is meaningless (concurrency is the load axis), so it is skipped.
    """
    out: dict[str, object] = {}
    # ``cache_blocks`` is the block-cache capacity; fall back to the legacy
    # ``cache_slots`` flag when the new one is unset.
    eff_cache_blocks = max(0, cache_blocks or cache_slots or 0)
    mooncake_rows = _load_mooncake_trace(mooncake_trace) if mooncake_trace else None

    # A split is a different topology, not a different setting, so it gets the
    # two-station loop rather than the unified-batch one. Routed before every
    # other branch because the alternative is not a worse simulation of this
    # deployment -- it is a simulation of a colocated one, which the single
    # engine has no way to say it is doing.
    if getattr(getattr(inference_config, "disaggregation_config", None), "enabled", False):
        reqs = None
        prefix_summary: dict[str, float] | None = None
        if mooncake_rows is not None:
            hasher = _BlockHasher()
            bs = int(block_size) if block_size and block_size > 0 else _DEFAULT_BLOCK_SIZE
            rows = sorted(mooncake_rows, key=lambda x: x[0])
            reqs = [
                _Req(idx=i, arrival_ms=a, prompt_len=isl, output_len=osl, blocks=list(hids))
                for i, (a, isl, osl, hids) in enumerate(rows)
            ]
            # The prefill pool is what owns a prefix cache here, so the hits are
            # warmed against one station rather than routed across a fleet.
            _, prefix_summary = _route_and_warm(
                reqs,
                policy="kv",
                num_instances=1,
                block_size=bs,
                cache_blocks=eff_cache_blocks,
                rng=random.Random(seed),
                overlap_weight=overlap_weight,
                waiting_depth=(
                    inference_config.request_config.resolved_max_concurrency()
                    if closed_loop
                    else 0
                ),
                resident_cap=_resident_cap(
                    reqs,
                    kv_cache_tokens,
                    inference_config.request_config.resolved_max_concurrency(),
                    admit_backlog_only,
                ),
            )
            prefix_summary["routing"] = float(_ROUTING_POLICIES.index("kv"))
            prefix_summary["trace_driven"] = 1.0
            del hasher
        eff_rate = rate_per_s
        if mooncake_rows and not closed_loop:
            span_s = (max(r[0] for r in mooncake_rows) - min(r[0] for r in mooncake_rows)) / 1000.0
            eff_rate = (len(mooncake_rows) / span_s) if span_s > 0 else rate_per_s
        out["point"] = simulate_disaggregated(
            inference_config,
            projector,
            rate_per_s=eff_rate,
            arrival_model=arrival_model
            if arrival_model in ("poisson", "deterministic")
            else "poisson",
            num_requests=num_requests,
            seed=seed,
            warmup_frac=warmup_frac,
            warmup_requests=warmup_requests,
            burstiness=burstiness,
            range_ratio=range_ratio,
            kv_cache_tokens=kv_cache_tokens,
            prebuilt=reqs,
            closed_loop_clients=(
                inference_config.request_config.resolved_max_concurrency() if closed_loop else 0
            ),
            closed_loop_think_ms=closed_loop_think_ms,
            duration_ms=duration_ms,
        )
        # The split warms a prefix cache exactly as the colocated path does,
        # but discarded the summary, so every disaggregated row reported no
        # hit rate at all -- indistinguishable, to anyone reading the output,
        # from a split that never reuses anything.
        if prefix_summary is not None:
            out["point"].prefix = prefix_summary
        return out
    if closed_loop:
        clients = inference_config.request_config.resolved_max_concurrency()
        if mooncake_rows is not None or (num_prefixes or 0) > 0 or (num_instances or 1) > 1:
            # There is a block cache to model, so the requests have to be built
            # and warmed before the arrivals are replaced by client slots.
            out["point"] = simulate_multi_instance(
                inference_config,
                projector,
                rate_per_s=0.0,
                arrival_model="poisson",
                num_requests=num_requests,
                seed=seed,
                warmup_frac=warmup_frac,
                warmup_requests=warmup_requests,
                burstiness=burstiness,
                range_ratio=range_ratio,
                kv_cache_tokens=kv_cache_tokens,
                num_instances=max(1, num_instances or 1),
                routing=routing if routing in _ROUTING_POLICIES else "round_robin",
                overlap_weight=overlap_weight,
                num_prefixes=max(0, num_prefixes or 0),
                prefix_len=max(0, prefix_len or 0),
                prefix_zipf=max(0.0, prefix_zipf or 0.0),
                block_size=max(0, block_size or 0),
                cache_blocks=eff_cache_blocks,
                admit_backlog_only=admit_backlog_only,
                mooncake_rows=mooncake_rows,
                closed_loop_clients=clients,
                closed_loop_think_ms=closed_loop_think_ms,
                duration_ms=duration_ms,
                prefill_exclusive=prefill_exclusive,
                cache_shares_pool=cache_shares_pool,
                whole_context_residency=whole_context_residency,
            )
            return out
        out["point"] = simulate_once(
            inference_config,
            projector,
            rate_per_s=0.0,
            arrival_model="closed",
            num_requests=num_requests,
            seed=seed,
            warmup_frac=warmup_frac,
            warmup_requests=warmup_requests,
            range_ratio=range_ratio,
            kv_cache_tokens=kv_cache_tokens,
            record_steps=record_steps,
            closed_loop_clients=clients,
            closed_loop_think_ms=closed_loop_think_ms,
            whole_context_residency=whole_context_residency,
            prefill_exclusive=prefill_exclusive,
            new_seqs_per_step=new_seqs_per_step,
            duration_ms=duration_ms,
        )
        return out
    # A block cache is modelled whenever there is a fleet, a synthetic prefix
    # pool, or a trace to replay.
    multi = (num_instances or 1) > 1 or (num_prefixes or 0) > 0 or mooncake_rows is not None
    if multi and not workload_file:
        eff_rate = rate_per_s
        if mooncake_rows:
            span_s = (max(r[0] for r in mooncake_rows) - min(r[0] for r in mooncake_rows)) / 1000.0
            eff_rate = (len(mooncake_rows) / span_s) if span_s > 0 else rate_per_s
        out["point"] = simulate_multi_instance(
            inference_config,
            projector,
            rate_per_s=eff_rate,
            arrival_model=arrival_model
            if arrival_model in ("poisson", "deterministic")
            else "poisson",
            num_requests=num_requests,
            seed=seed,
            warmup_frac=warmup_frac,
            warmup_requests=warmup_requests,
            burstiness=burstiness,
            range_ratio=range_ratio,
            kv_cache_tokens=kv_cache_tokens,
            num_instances=max(1, num_instances or 1),
            routing=routing if routing in _ROUTING_POLICIES else "round_robin",
            overlap_weight=overlap_weight,
            num_prefixes=max(0, num_prefixes or 0),
            prefix_len=max(0, prefix_len or 0),
            prefix_zipf=max(0.0, prefix_zipf or 0.0),
            block_size=max(0, block_size or 0),
            cache_blocks=eff_cache_blocks,
            admit_backlog_only=admit_backlog_only,
            mooncake_rows=mooncake_rows,
            duration_ms=duration_ms,
            prefill_exclusive=prefill_exclusive,
            cache_shares_pool=cache_shares_pool,
            whole_context_residency=whole_context_residency,
        )
        return out
    out["point"] = simulate_once(
        inference_config,
        projector,
        rate_per_s=rate_per_s,
        arrival_model=arrival_model,
        num_requests=num_requests,
        seed=seed,
        warmup_frac=warmup_frac,
        warmup_requests=warmup_requests,
        burstiness=burstiness,
        range_ratio=range_ratio,
        kv_cache_tokens=kv_cache_tokens,
        workload_file=workload_file,
        record_steps=record_steps,
        duration_ms=duration_ms,
    )

    if sweep and not workload_file:
        steady = projector.project()
        osl = max(1, inference_config.request_config.output_seq_len)
        mu = (steady.decode_throughput_tps / osl) if steady.decode_throughput_tps > 0 else 0.0
        curve: list[DESResult] = []
        if mu > 0:
            sweep_n = min(num_requests, 300)
            fracs = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
            rates = sorted({round(f * mu, 4) for f in fracs} | {round(rate_per_s, 4)})
            for lam in rates:
                if lam <= 0:
                    continue
                curve.append(
                    simulate_once(
                        inference_config,
                        projector,
                        rate_per_s=lam,
                        arrival_model=arrival_model,
                        num_requests=sweep_n,
                        seed=seed,
                        warmup_frac=warmup_frac,
                        warmup_requests=warmup_requests,
                        burstiness=burstiness,
                        range_ratio=range_ratio,
                        kv_cache_tokens=kv_cache_tokens,
                    )
                )
        out["curve"] = curve
        out["max_sustainable_rate"] = mu
    return out
