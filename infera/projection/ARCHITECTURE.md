# InferaSim: simulating the serving stack before spending GPUs

A serving deployment is not one decision, it is a stack of them. Tensor,
expert and pipeline shape. Whether prefill and decode share GPUs or sit in
separate pools. The scheduler's per-step token budget and how many sequences it
will hold resident. How much HBM the engine is allowed to claim, and how the KV
cache is paged inside it. How requests are routed across replicas, and how much
of a prompt a replica already has cached. And underneath all of it, the shape of
the traffic itself.

These interact. Raising the token budget improves prefill throughput and
pollutes inter-token latency. Routing for cache locality raises prefix reuse and
concentrates decode pressure on whichever replica holds the popular prefix.
Lowering tensor parallelism improves per-GPU throughput right up to the point
where the model no longer fits. A local improvement usually moves the
bottleneck rather than removing it, and you only find out where it went by
measuring the whole thing.

Measuring the whole thing is the problem. One honest experiment on a large model
occupies a full node for minutes, and the interesting questions are asked in the
thousands. InferaSim exists so that the search happens in simulation and the
hardware is spent only on the shortlist.

## What it is

This document describes how it works. For installation and task-oriented usage,
see [README.md](README.md).

InferaSim is a workload-driven simulator and projector for the serving stack.
It is deliberately two coupled models over one measured foundation:

- an **analytical projector** that solves a serving configuration in closed
  form and answers steady-state questions — time to first token, inter-token
  latency, throughput, KV and weight memory, and whether the configuration fits
  at all;
- a **discrete-event simulator** (DES) that runs the same cost model on a
  virtual clock and answers the questions a closed form structurally cannot —
  latency *distributions*, queueing under an offered load, and the behaviour of
  a fleet of replicas sharing a cache and a router.

It is not a purely analytical estimate, and it is not a bit-exact hardware
emulator. The target is fidelity at the granularity of a forward pass: get the
duration of each pass from measurement or from a calibrated cost model, decide
what goes *into* each pass with a real scheduler policy, and let everything
above that emerge.

The governing idea is **measure sparsely, transport analytically**. A small
number of cheap sub-scale benchmarks are harvested into anchors; every other
configuration is projected from the nearest applicable anchor rather than
measured.

## Architecture: one timeline, composed parts

There is no monolithic model. Four concerns compose:

| Layer | Question it answers |
|---|---|
| Cost kernel | How long does *this* forward pass take? |
| Scheduler | What is *in* each forward pass? |
| Fleet | Which replica serves a request, and what has it already cached? |
| Workload | When do requests arrive, and what is in them? |

The cost kernel is the time axis for everything above it. The analytical
projector consumes it directly; the DES consumes it once per simulated step.
Because both sit on the same kernel, a measured anchor loaded for one is
automatically honoured by the other.

### The virtual clock

The DES gives the simulation a virtual clock and an event queue. Nothing waits
in real time. A step is scheduled with a modelled duration, the clock jumps to
the next timestamp, state advances, and the components affected schedule
further work. A router decision changes a replica's queue; an admission
decision changes when decode can begin; an eviction changes what the next
request has to recompute.

### A request's journey

1. **Arrival.** Either a timestamp from a replayed trace, or a sample from a
   configured arrival process.
2. **Routing.** A policy picks the replica — round robin, random, prefix-aware
   hashing, or KV-overlap scoring.
3. **Cache lookup.** The prompt is an ordered sequence of content-addressed
   block hashes. A hit is the longest *contiguous leading run* of blocks
   already resident on that replica; matched tokens are seeded as already
   computed.
4. **Admission.** The scheduler admits from the waiting set subject to a
   resident-sequence cap and a full-ISL KV reservation against a finite token
   pool. A request that does not fit waits.
5. **Packing.** Each forward pass first advances every running request, then
   admits new work, all sharing one per-step token budget. Chunked prefill is
   not special-cased — it *emerges* from the budget, so a step naturally mixes
   prefill chunks with decodes.
6. **Duration.** The step's cost comes from the cost kernel, which distinguishes
   a uniform decode step from a mixed prefill+decode one.
7. **Completion.** Per-request latencies land in the collector, along with
   per-step batch composition if requested.

### Driving it: workloads

Two sources of request content, and several arrival processes.

**Trace replay** consumes a Mooncake-format trace — JSONL or JSON carrying
`timestamp`, `input_length`, `output_length` and `hash_ids`. Because
consecutive requests that share a system prompt share leading `hash_ids`, replay
drives genuine content-addressed reuse rather than an assumed hit rate. A trace
supplies its own arrivals, so no offered rate is required.

**A synthetic prefix pool** generates the same structure parametrically: `P`
distinct shared prefixes of `L` tokens, blockified at the configured block size,
with a Zipf parameter to skew popularity the way a few hot system prompts
dominate real traffic.

**Arrival processes** cover closed-loop (no queue, steady state only), Poisson,
deterministic, and gamma-bursty via a burstiness shape parameter. The open-loop
processes run the DES for percentiles; all of them still report the analytical
queueing mean alongside, so the closed form and the simulation stay visible to
each other rather than being alternatives you pick between. Per-request lengths
are heterogeneous around the configured ISL/OSL by a range ratio. An
offered-load sweep re-runs the workload across fractions of the maximum
sustainable rate and emits a throughput-versus-latency curve, which is how you
find its knee rather than assuming where it sits.

## Single engine: the scheduler is the point

A single engine is not a tokens-per-second number. What dominates tail latency
is how requests wait, batch, chunk and enter prefill — so the scheduler is
modelled rather than summarised.

The DES implements the unified-batch policy, modelled on vLLM V1: every forward
pass advances already-running work (decodes and in-progress prefill chunks)
under a shared per-step token budget, then admits new waiting requests against
the resident cap and the KV reservation. Both phases share one budget, which is
precisely what makes chunked prefill and mixed steps fall out of the model
instead of being bolted on.

The analytical path offers the same distinction more cheaply through a serving
model selector: `continuous` charges the inter-token cost of mixed
prefill+decode steps, while `static` treats decode as an idealised pure-decode
batch with prefill charged once as TTFT. The first is what a real continuous-
batching engine does; the second is the clean upper bound worth comparing
against.

Several engine behaviours that materially move latency are modelled as
first-class terms rather than ignored:

- the fraction of HBM the engine may claim, which bounds usable memory and
  therefore maximum concurrency;
- paged-KV block sizing, including the internal fragmentation of a partially
  filled final block;
- the streaming flush interval, which delays the client's first token — and
  therefore measured TTFT — until several tokens have been decoded;
- decode-scheduler admission granularity, the delay between a prefill finishing
  and the request joining the running batch, which is a pure TTFT term and
  dominates in disaggregated serving where prefill is off the critical path;
- a penalty for mixed steps reflecting the less efficient CUDA-graph path
  relative to uniform decode steps;
- prefix-cache reuse, expressed either as a static hit fraction on the
  analytical path or as an emergent property of the block cache in the DES.

## The time axis: measure sparsely, transport analytically

The cost kernel runs in one of three modes: `simulate` (no GPU, analytical
backends for GEMM and attention), `benchmark` (real GPU measurement), or `both`
(run each and report side by side).

Pure simulation is the fast path and needs no accelerator at all. Calibration is
what makes it trustworthy, and the mechanism is the **anchor store**: a
directory of measured benchmark artifacts indexed by a **regime signature**.

Recipe parameters split in two. **Regime-defining** parameters swap the
kernel or execution path — dtype, backend, the graph or attention library in
use — and two recipes differing on any of them are not transportable from one
another; each regime needs its own anchor. **Transportable** parameters —
layer count, parallel shape, batch, sequence length — move analytically from an
existing measurement. Reconstruction picks the nearest in-regime anchor and
transports it to the target recipe by driving the same projector, so the physics
lives in one place.

A single-GPU anchor cannot observe cross-GPU communication, which is exactly
the cost that matters at scale. The **confidence ladder** handles this by
climbing the benchmark GPU count until per-GPU decode is flat within a tolerance
across an adjacent pair of rungs, bounding the restore error. A flat pair at
rung `g` certifies targets up to `2g`, so a handful of cheap runs certify a
full-node target. The ladder is capped by default; targets beyond the top rung
are still projected but reported as extrapolated rather than certified, which
keeps the distinction visible instead of silently confident.

## Multi-engine: fleet behaviour

Replicas behind a router are simulated together, because the interesting
behaviour only exists across them.

Each replica owns a **content-addressed, paged KV block cache** with finite
capacity and LRU eviction. Hit rate is therefore *emergent* — a function of
workload content, capacity and routing — rather than a number you supply. The
routing policies differ in exactly this respect:

- **KV-aware scoring** minimises the deployed router's own cost function: an
  overlap weight times the blocks a replica would have to compute, plus what it
  is already carrying. The weight is the dial between reuse and balance, so the
  simulator answers what retuning it in production would cost.
- **Prefix-aware hashing** consistently maps a leading block to a home replica,
  so same-prefix requests co-locate and misses scale with the number of
  distinct prefixes rather than with fleet size.
- **Round robin** and **random** ignore locality entirely, so every replica
  re-warms the same prefixes.

The point of simulating all four is that the trade-off is real and not
one-sided: locality-seeking policies maximise reuse but can overconcentrate a
hot prefix on one replica, raising its decode pressure and lengthening the
makespan. The fleet report surfaces both halves — pooled latency distributions
alongside the spread of per-replica hit rates — so the trade is visible rather
than argued.

Setting the fleet to a single replica is the way to study one engine's automatic
prefix caching as temporal reuse across a stream; constraining block capacity is
the way to study eviction pressure.

## Disaggregation

Prefill and decode can be modelled as separate worker pools, each with its own
parallel shape, with the KV-cache transfer cost charged when a request migrates
from a prefill worker to a decode worker. Disabled, the projector runs the
standard colocated two-phase model. This is where the decode admission term
above matters most, since prefill compute is off the critical path and what
remains visible in TTFT is the waiting.

## What comes out

The analytical path reports TTFT, inter-token latency, decode throughput in
total and per GPU, memory per GPU, and feasibility. Infeasible configurations
are *kept* and annotated with the reason rather than dropped, so a search can
distinguish "does not fit" from "was never tried".

The DES reports offered and achieved request rate, utilisation, makespan, system
throughput, and whether the configuration saturated — plus full distributions
for TTFT measured from admission, TTFT measured from arrival (including queue
wait), the queue wait itself, inter-token latency, time per output token, and
end-to-end latency. Separating the two TTFT definitions matters: one is the
engine's behaviour, the other is what a client experiences, and under load they
are not the same number. Batch-composition and prefix-reuse summaries come
alongside, with optional per-step records for inspecting how the scheduler
actually packed each pass.

## Search: simulation as a scoring function

Once a configuration can be scored without hardware, search becomes a loop over
that score.

The built-in sweep enumerates parallel shapes and concurrencies, applies a
caller-supplied validity filter to express rules the cost model should not have
to know — expert parallelism not exceeding tensor parallelism, a fixed GPU
budget — and projects everything that survives. Results can be ranked and
shortlisted by objective.

Above that sits a tuning agent that uses the projector as an oracle. It starts
from a deterministic seed sweep for a warm start, then runs an LLM-driven search
that continues from the warm-started incumbent, proposing recipes and scoring
them through the projector with no GPU in the default path. The search space is
large and awkward to enumerate, so the agent navigates rather than grids.

## Simulate, then verify

The goal is not to replace hardware validation but to aim it. Simulation is the
inner loop: sweep broadly, rank, shortlist. Hardware is the outer loop: verify
the shortlist, and harvest what you measured back into the anchor store so the
next sweep is better calibrated than the last. The loop closes because anchors
are artifacts, not one-off runs.

Memory is deliberately outside that loop. Weights and KV bytes are counted from
the model shape and the parallel layout rather than fitted, so an anchor has
nothing to contribute to them, and capacity measured on a warmup's reduced
parallelism does not describe the target's. The memory projection is analytical
at every parallelism, and its correctness is a question about the formulas
rather than about calibration.

## Boundaries

What a tool does not model matters as much as what it does.

- **One scheduler policy.** The DES implements the unified-batch policy. Engine
  families with materially different admission — radix-cache-aware admission,
  prefix-preserving decode retraction — are not modelled as separate scheduler
  cores, though their budget and memory knobs are honoured.
- **No capacity control.** There is no autoscaler, no SLA-driven replica
  scaling, and no model of worker startup delay. The fleet size is what you set
  it to.
- **One cache tier.** KV lives in device memory. There is no host-memory or SSD
  tier, no offload or onboard traffic, and no distributed cache target.
- **Only what enters the serving spec.** The projection is built from parallel
  shape, concurrency, sequence lengths and precision. Server flags that do not
  enter that spec are passed through untouched and project to the same number,
  so the tool cannot rank kernel- or scheduler-level flags against each other.
  Those belong on hardware, and treating a projection as a ranking over them
  would produce confident noise.
- **Calibration is per regime.** An anchor certifies its own execution regime.
  Change dtype or kernel backend and the previous anchor does not transfer;
  harvest another.

These are the honest edges of the tool. Inside them it is fast enough to make
exhaustive search routine; outside them, the answer is still a GPU.
