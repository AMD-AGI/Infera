# Boundaries and verification

```{admonition} One-pager
:class: tip
**What:** the honest edges of the simulator, and the loop that closes them.
**Why:** what a tool does not model matters as much as what it does — a
confident number from outside its scope is worse than no number.
**Cost:** reading one page before you trust a result.
```

## Simulate, then verify

The goal is not to replace hardware validation but to **aim** it.

```{graphviz}
digraph verify_loop {
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc" fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.22,0.13"];
  edge [fontname="Helvetica,Arial,sans-serif" fontsize=10 color="#5577cc"];

  SWEEP [label="Sweep broadly\nno GPU"];
  RANK  [label="Rank and shortlist"];
  HW    [label="Verify on hardware\nthe shortlist only" fillcolor="#fff3cd" color="#caa300"];
  STORE [label="Harvest into\nthe anchor store" fillcolor="#e6f5ec" color="#2fa36b"];

  SWEEP -> RANK -> HW -> STORE;
  STORE -> SWEEP [label="next sweep is better calibrated" constraint=false style=dashed];
}
```

Simulation is the inner loop: sweep, rank, shortlist. Hardware is the outer
loop: verify the shortlist, and harvest what you measured back into the
[anchor store](anchors.md) so the next sweep is better calibrated than the last.
The loop closes because anchors are artifacts, not one-off runs.

## What is deliberately not modelled

**One scheduler policy.** The DES implements the unified-batch policy. Engine
families with materially different admission — radix-cache-aware admission,
prefix-preserving decode retraction — are not modelled as separate scheduler
cores, though their budget and memory knobs are honoured.

**No capacity control.** There is no autoscaler, no SLA-driven replica scaling,
and no model of worker startup delay. The fleet size is what you set it to.

**One stateful cache tier in the DES.** The analytical projector can price a
bounded host-KV offload allowance and its transfer bandwidth, but the DES has one
device-resident block cache. It does not simulate tier promotion, asynchronous
prefetch, host or SSD eviction, distributed cache ownership, or contention on the
offload link. For those, see [KV cache offload](../features/kv_cache_offload.md)
and measure.

**Disaggregation is analytical only.** No independent prefill/decode event
queues, no transfer-contention events, no role-specific schedulers — so no PDD
tail distributions. See [Disaggregated serving](disaggregation.md).

**Only what enters the serving spec.** The projection is built from parallel
shape, concurrency, sequence lengths and precision. Server flags that do not
enter that spec are passed through untouched and project to the same number, so
the tool cannot rank kernel- or scheduler-level flags against each other. Those
belong on hardware, and treating a projection as a ranking over them would
produce confident noise.

**Calibration is per regime.** An anchor certifies its own execution regime.
Change dtype or kernel backend and the previous anchor does not transfer;
harvest another.

## Memory is outside the calibration loop

Weights and KV bytes are counted from the model shape and the parallel layout
rather than fitted, so an anchor has nothing to contribute to them — and capacity
measured at a warmup's reduced parallelism would not describe the target's
anyway. The memory projection is analytical at **every** parallelism, and its
correctness is a question about the formulas rather than about calibration.

This is why the parallel layout has to include how attention itself is split;
see [data-parallel attention](projection_runs.md#data-parallel-attention-mla).

## Troubleshooting

**It says no GPUs are visible and refuses to run.** `--profiling-mode` defaults
to `benchmark`, so a projection on a machine without an accelerator has nothing
to measure on. Either give it a measurement (`--load-benchmark anchor.json`, or
`--anchor-store <dir>` so it finds one for this regime itself) or ask for the
uncalibrated analytical path explicitly with `--profiling-mode simulate`. It
refuses rather than downgrading, because a silently analytical number reads
exactly like a measured one. See
[Fidelity sources](overview.md#fidelity-sources).

**Concurrency is far higher than I asked for.** Pass `--max-concurrency`.
`--inference-batch-size` does not cap serving concurrency; see
[Control concurrency](projection_runs.md#control-concurrency).

**Every sweep point is infeasible.** Check `p.reason`. The usual causes are an
HBM budget too small for the weights at that parallel shape, or a `valid=`
predicate that rejects everything.

**The DES says `[SATURATED]`.** Offered load exceeds capacity, so the queue grows
without bound and the latency percentiles describe a backlog rather than an
operating point. Lower `--request-rate` below the reported max sustainable rate,
or add replicas with `--des-instances`.

**Prefix hit rate is 0% in the DES.** The requests have no shared content.
Provide either `--des-num-prefixes`/`--des-prefix-len` or a trace with
`hash_ids`. `--prefix-cache-hit-rate` is the *analytical* knob and does not drive
the block cache — see [Workloads and traces](workloads.md).

**Anchor results look wrong after changing dtype or backend.** An anchor
certifies its own execution regime. Harvest another; see
[Regimes](anchors.md#regimes-when-an-anchor-stops-transferring).

**TTFT reads early against a benchmark harness.** The analytical path prices
prefill *service* time, not client-observed response time. Use
`--des-closed-loop` and set `--chunked-prefill-size` — see
[Reproduce a fixed-concurrency benchmark](simulation_runs.md#reproduce-a-fixed-concurrency-benchmark).

## Environment variables

Everything is `INFERASIM_*`. The ones you will actually set:

| Variable | Meaning |
|---|---|
| `INFERASIM_MODEL` | architecture preset (e.g. `gpt_oss_120B`) |
| `INFERASIM_TP` / `_EP` / `_PP` / `_CP` / `_VP` | parallel shape |
| `INFERASIM_GPU_ARCH` | target GPU architecture |
| `INFERASIM_ROOT` | repo/config root override |
| `INFERASIM_ANCHOR_STORE` | directory of measured anchors |
| `INFERASIM_SEQ_LENGTH` | default sequence length |
| `INFERASIM_TEAM` / `_USER` / `_EXP_NAME` / `_WORKSPACE` | launcher identity fields |

Further `INFERASIM_*` variables exist for kernel-model and benchmark internals
(`INFERASIM_GEMM_BACKEND`, `INFERASIM_BENCH_*`, `INFERASIM_MOE_*`,
`INFERASIM_DEBUG_*` and others). They are advanced overrides.

Inside these boundaries the tool is fast enough to make exhaustive search
routine. Outside them, the answer is still a GPU.
