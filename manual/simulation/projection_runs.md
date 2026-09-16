# Projection runs

```{admonition} One-pager
:class: tip
**What:** the analytical path — one serving recipe in, steady-state means out.
**Why:** it costs milliseconds, so it is what you run thousands of.
**Cost:** means only. For a p99, use a [simulation run](simulation_runs.md).
```

## Your first projection

The command below passes `--profiling-mode simulate` explicitly, which is what
makes it runnable on a laptop: the default is to *measure*, because only the
calibrated kernel has an established correlation against real serving. Read
[Fidelity sources](overview.md#fidelity-sources) before you quote a number from
this mode — and once you have an anchor, drop the flag and
[project from the anchor](anchors.md) instead.

```bash
INFERASIM_MODEL=gpt_oss_120B INFERASIM_TP=2 INFERASIM_EP=2 \
inferasim inference \
  --config infera/projection/examples/exp_pretrain.yaml \
  --inference-mode performance --serving-model continuous \
  --input-len 1024 --output-len 1024 --max-concurrency 32 \
  --gpu-arch mi355x --hbm-capacity-gb 288 \
  --profiling-mode simulate
```

The report ends with:

```text
[inferasim:Inference] Performance Projection
  Workload: input=1024 tok, output=1024 tok, batch=1
  Serving model: CONTINUOUS BATCHING (concurrency=32)
  Profiling source: SIMULATION
  Max sustainable concurrency: 4808  (HBM=288 GB via --hbm-capacity-gb)
  Concurrency used: 32
  TTFT (time to first token):      42.32 ms
  ITL / TPOT (per token):          14.49 ms
  Interactivity (per user):        69.0 tok/s/user
  Decode step latency (pure):      13.68 ms  | mixed: 38.90 ms
    Mixed-step fraction:           3.12%  → TPOT pollution: 8.4%
  End-to-end request latency:      14870.59 ms
  Per-request decode throughput:   69.0 tok/s
  Aggregate decode throughput:     2209.8 tok/s
  Decode throughput / GPU:         1104.9 tok/s/gpu
  Prefill throughput:              40576.9 tok/s
  Replica GPUs (TP×PP):            2
  Communication breakdown (exposed ms/forward):
    prefill:  TP-AR 3.22 | EP-A2A 14.85 | PP-P2P 0.00 | total 18.07
    decode:   TP-AR 0.00 | EP-A2A 0.01 | PP-P2P 0.00 | total 0.02
```

## Reading the report

**Max sustainable concurrency** is what fits in HBM after weights. It is a
capacity ceiling, not a recommendation — running there maximises throughput and
destroys latency.

**Mixed-step fraction / TPOT pollution** is the continuous-batching tax. Steps
that carry prefill chunks alongside decodes are slower, and this is how much
that lifts your per-token latency. It is the number that disappears if you
[disaggregate prefill from decode](disaggregation.md), which makes it the honest
way to decide whether disaggregation is worth its KV transfer.

**Decode throughput / GPU** is the right axis for comparing recipes with
different GPU counts. Aggregate throughput alone will always favour whichever
recipe was handed more GPUs.

**Communication breakdown** shows *exposed* — that is, non-overlapped —
collective time, so you can see whether a recipe is compute-bound or comm-bound
before you change its parallel shape.

## The memory projection

`--inference-mode` selects which of the two projections runs — `performance`,
`memory`, or `both` (the default). The memory half answers "will this fit, and
how many sequences can it hold?", and it is worth running on its own when you
are sizing rather than timing:

```bash
inferasim inference ... --inference-mode memory --hbm-capacity-gb 288
```

```text
  Params (this rank):       58.9412 B
  Weights (fp8):            58.9412 GB
  KV cache (fp8):           12.4180 GB (concurrency=32, ctx=2048, layers/rank=36)
    KV per sequence:        0.3881 GB
  Activation working set:   3.1120 GB
  Projected Total Memory:   74.4712 GB
  HBM capacity:             288.0000 GB
  Max concurrent sequences: 4808
```

Serving memory is a much simpler story than training memory — weights only, no
gradients, no optimizer state — plus a KV cache that grows with resident
concurrency × context, and a comparatively small forward activation working set
for the in-flight batch. `--max-context-len` sets the KV sizing horizon if it
should be something other than input+output, and `--kv-cache-memory-fraction`
bounds how much of HBM the engine is allowed to claim.

Two things to know about these numbers. **They are never calibrated by a
measurement** — see
[Memory is outside the calibration loop](boundaries.md#memory-is-outside-the-calibration-loop).
And for a [disaggregated](disaggregation.md) deployment the ceiling reported is
the *decode* pool's, because that is where the KV cache lives and what caps
concurrency.

## Control concurrency

This is the most common source of confusing results. **`--max-concurrency` is
what sets the operating point.**

```bash
--max-concurrency 32        # project at 32 in-flight requests
```

Without it, the projector runs at the maximum concurrency that fits in HBM —
which for a small model on a large GPU can be thousands of requests, producing
enormous TTFT and near-zero per-user interactivity that look like bugs but are
just saturation.

`--inference-batch-size` describes the workload's batch shape; it does **not**
cap serving concurrency. If the report says `Concurrency used:` followed by a
number far larger than you expected, this is why.

## Choose a model and hardware

`INFERASIM_MODEL` selects an architecture preset — `gpt_oss_120B`,
`deepseek_v3`, `llama3.1_405B`, `kimi_k2` and the `qwen3_*` family are among the
presets that ship. List what is available on your checkout with:

```bash
ls infera/projection/configs/models/megatron/
```

Hardware is `--gpu-arch` plus `--hbm-capacity-gb`. The HBM figure bounds what
fits and therefore the maximum concurrency, so it is not a detail — a projection
against the wrong HBM budget is a projection of a different machine.

## Set the serving recipe

Parallelism comes from the environment so a sweep can vary it per point:

```bash
INFERASIM_TP=8 INFERASIM_EP=8 INFERASIM_PP=1 inferasim inference ...
```

A replica occupies **TP × PP** GPUs. Expert parallelism is placed *within* the
tensor-parallel GPUs, so raising EP does not add GPUs. Dtypes are
`--weight-dtype` and `--kv-cache-dtype`.

### Data-parallel attention (MLA)

```bash
--attention-dp-size 8       # split requests across the 8 tensor-parallel ranks
```

Attention runs data-parallel while the MLP and experts stay tensor- and
expert-parallel: each rank owns a subset of the in-flight requests and holds
their whole KV cache. The size must divide TP, which it subdivides rather than
adds to, so the replica's GPU count does not change.

This is how MLA models are actually served. Tensor parallelism shards a GQA
model's cache because it shards KV heads, but MLA caches one compressed latent
that every head reads, so TP replicates it — at TP=8, DeepSeek-R1 stores the
same cache eight times. Splitting by request instead stores it once, and the
capacity that frees is the whole answer to a long-context or agentic sizing
question: 277 concurrent sequences on an MI355X node becomes 2216. A rank whose
attention output is its own also has nothing to all-reduce, so the per-layer TP
collective count drops with it.

For GQA models the axis is close to a no-op — TP already sharded the heads.

## Declare prefix reuse

Agentic and multi-turn traffic shares long prefixes — system prompts, tool
schemas, conversation history — that the engine keeps resident via automatic
prefix caching. On the analytical path, state the reuse directly:

```bash
inferasim inference ... --input-len 4096 --prefix-cache-hit-rate 0.8
```

The cached prefix (`R × input_len` tokens) skips prefill compute; the remaining
suffix still attends over the full context. So **TTFT and the prefill share of
continuous-batching pollution scale with `(1 - R)`**, while decode and KV sizing
are unchanged. `R = 0` is a cold cache. At least one token is always prefilled,
so `R` is clamped below 1.

This works in both analytical and anchor-calibrated modes. Set it per workload
in a YAML `inference:` block (`prefix_cache_hit_rate: 0.8`) so every recipe in a
search is scored under the same assumption.

For reuse that *emerges* from workload content rather than being asserted, use
the DES block cache — see [Fleet and routing](fleet_and_routing.md).

## Price the projection

Give the projector a GPU-hour price and it reports throughput in the unit a
serving budget is quoted in:

```bash
--gpu-cost-per-hour 2.50
```

```text
  Cost basis:                      $2.5/GPU-h x 2 GPU
  Cost / 1M output tokens:         $0.629
  Cost / 1M in+out tokens:         $0.314
```

The whole replica is charged, so a recipe that buys throughput with GPUs is
billed for them. That is the point: tokens/s ranks recipes by speed, and the
faster recipe is not always the cheaper one. The `in+out` line blends over the
workload's own mix — at 3072 in / 1024 out, four tokens are billed for every one
emitted, so it is a quarter of the output-only figure.

Nothing is priced without a price. Omit the flag and the cost lines are absent
rather than defaulted, since an invented GPU-hour rate would be read as a
measured one.

## Next steps

- The means look plausible but you need a tail: [Simulation runs](simulation_runs.md).
- You want the numbers grounded in hardware: [Anchors and calibration](anchors.md).
- You want to compare many recipes at once: [Sweeps and tuning](sweeps.md).
