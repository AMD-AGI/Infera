# InferaSim — serving simulator and projector

**InferaSim** answers "how would this serving configuration behave?" without
standing up the configuration. It projects TTFT, inter-token latency,
throughput and KV-cache footprint for a serving recipe, and simulates a fleet
of engines under an arrival-driven load — on a laptop, with no GPU in the
default path.

The point is to make the search cheap. Screen thousands of candidate
deployments in simulation, then spend GPU time only on the shortlist.

The governing idea is **measure sparsely, transport analytically**: benchmark
one cheap sub-scale anchor on a single GPU, then project every other recipe
(TP/EP/PP, batch, concurrency, dtype) from it instead of re-measuring.

- [Install](#install)
- [Concepts in two minutes](#concepts-in-two-minutes)
- [Quickstart](#quickstart-your-first-projection)
- [How-to guides](#how-to-guides)
- [Environment variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

For how the pieces work internally — the cost kernel, the scheduler model,
anchors and regimes, and what the tool deliberately does not model — see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Install

```bash
pip install ".[projection]"          # projection + simulation
pip install ".[projection-tuning]"   # + the DSPy tuning agent
```

`torch` and the serving engine (`vllm`) come from the engine base image, not
pip. Neither is needed for the no-GPU path.

Two console scripts are installed:

| Command | Purpose |
|---|---|
| `inferasim` | projection + discrete-event simulation |
| `inferasim-tune` | LLM-driven recipe search |

`infera-projection` and `infera-tuning` remain as aliases. Everything below can
also be run as `python -m infera.projection.cli` if you would rather not
install the scripts.

## Concepts in two minutes

**Four things describe a run.** The *model* (an architecture preset), the
*hardware* (GPU arch and HBM budget), the *recipe* (parallel shape, dtypes,
concurrency — the thing you are searching over), and the *workload*
(input/output lengths, arrival pattern, prefix reuse).

**Two engines answer different questions.**

| Engine | Answers | How to get it |
|---|---|---|
| Analytical projector | steady-state means: TTFT, ITL/TPOT, throughput, memory, feasibility | default |
| Discrete-event simulator (DES) | distributions (p50/p90/p99), queueing under offered load, fleet behaviour | add `--arrival-model poisson` or a trace |

They share one cost model, so a measured anchor loaded for one is honoured by
the other. The DES report is printed *in addition to* the analytical one.

**Three fidelity sources**, selected with `--profiling-mode`:

| Mode | Needs a GPU | Meaning |
|---|---|---|
| `simulate` | no | analytical kernel models (default for sweeps) |
| `benchmark` | yes | measured on real hardware |
| `both` | yes | run each and report side by side |

Sitting between them is the **anchor**: a saved artifact from one cheap
measured run that calibrates the analytical path. See
[Calibrate against a GPU](#calibrate-against-a-gpu-anchors).

## Quickstart: your first projection

No GPU required.

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

```
[inferasim:Inference] Performance Projection
  Workload: input=1024 tok, output=1024 tok, batch=32
  Serving model: CONTINUOUS BATCHING (concurrency=32)
  Profiling source: SIMULATION
  Max sustainable concurrency: 6117  (HBM=288 GB via --hbm-capacity-gb)
  Concurrency used: 32
  TTFT (time to first token):      42.32 ms
  ITL / TPOT (per token):          14.49 ms
  Interactivity (per user):        69.0 tok/s/user
  Decode step latency (pure):      13.68 ms  | mixed: 38.90 ms
    Mixed-step fraction:           3.12%  → TPOT pollution: 8.4%
  Per-request decode throughput:   69.0 tok/s
  Aggregate decode throughput:     2209.8 tok/s
  Decode throughput / GPU:         1104.9 tok/s/gpu
  Replica GPUs (TP×PP):            2
  Communication breakdown (exposed ms/forward):
    prefill:  TP-AR 3.22 | EP-A2A 14.85 | PP-P2P 0.00 | total 18.07
    decode:   TP-AR 0.10 | EP-A2A 0.46 | PP-P2P 0.00 | total 0.56
```

Reading it:

- **Max sustainable concurrency** is what fits in HBM after weights. It is a
  capacity ceiling, not a recommendation — running there maximises throughput
  and destroys latency.
- **Mixed-step fraction / TPOT pollution** is the continuous-batching tax:
  steps that carry prefill chunks alongside decodes are slower, and this is how
  much that lifts your per-token latency. It is the number that disappears if
  you disaggregate prefill from decode.
- **Decode throughput / GPU** is the right axis for comparing recipes with
  different GPU counts; aggregate throughput alone will always favour more GPUs.
- **Communication breakdown** shows *exposed* (non-overlapped) collective time,
  so you can see whether a recipe is compute-bound or comm-bound before
  changing its parallel shape.

## How-to guides

### Choose a model and hardware

`INFERASIM_MODEL` selects an architecture preset resolved from
`configs/models/<framework>/<model>.yaml` (65 presets ship, including
`gpt_oss_120B`, `deepseek_v3`, `llama3.1_405B`, `kimi_k2`, `qwen3_*`). Hardware
is `--gpu-arch` plus `--hbm-capacity-gb`; the HBM figure bounds what fits and
therefore the maximum concurrency.

```bash
ls infera/projection/configs/models/megatron/     # available presets
```

### Set the serving recipe

Parallelism comes from the environment so a sweep can vary it per point:

```bash
INFERASIM_TP=8 INFERASIM_EP=8 INFERASIM_PP=1 inferasim inference ...
```

A replica occupies **TP × PP** GPUs. Expert parallelism is placed *within* the
tensor-parallel GPUs, so raising EP does not add GPUs. Dtypes are
`--weight-dtype` and `--kv-cache-dtype`.

### Control concurrency

This is the most common source of confusing results. **`--max-concurrency` is
what sets the operating point.** Without it, the projector runs at the maximum
concurrency that fits in HBM — which for a small model on a large GPU can be
thousands of requests, producing enormous TTFT and near-zero per-user
interactivity that look like bugs but are just saturation.

```bash
--max-concurrency 32        # project at 32 in-flight requests
```

`--inference-batch-size` describes the workload's batch shape; it does not cap
the serving concurrency. If the report says `Concurrency used:` and a number
far larger than you expected, this is why.

### Calibrate against a GPU (anchors)

Harvest once on any ROCm host with vLLM:

```bash
inferasim anchor --model gpt_oss_120B --benchmark-gpus 1 --save anchor.json
```

Then project any recipe from it, with no GPU:

```bash
inferasim inference ... --load-benchmark anchor.json
```

The anchor JSON is engine-neutral (a `"backend"` field plus per-batch
decode/prefill measurements), so a different harvester can be added without
touching the projector.

**The confidence ladder.** A 1-GPU anchor cannot observe cross-GPU
communication (TP all-reduce, EP all-to-all). The ladder climbs the benchmark
GPU count per recipe until per-GPU decode is flat within ±5% across an adjacent
pair, which bounds the restore error. A flat pair at rung `g` certifies targets
up to `2g`, so rungs 1/2/4 certify an 8-GPU target from three cheap runs. The
ladder stops at 4 GPUs by default; beyond that, results are reported as
extrapolated (`capped`) rather than certified. Raise
`INFERASIM_LADDER_MAX_GPUS` when a larger benchmark host is available.

### Model prefix reuse

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
the DES block cache below.

### Simulate load and get percentiles

The analytical path gives means at a fixed concurrency. To model requests
arriving on their own schedule, queueing, and tail latency, run the DES:

```bash
inferasim inference ... \
  --request-rate 1.5 --arrival-model poisson --des-num-requests 120
```

```
[inferasim:Inference] Discrete-Event Simulation (arrival-driven)
  Arrivals: poisson @ 1.5 req/s offered  (achieved 1.20 req/s, utilization 99%)
  Simulated: 120 requests over 100.20 s  → system throughput 1226 tok/s
  metric                        mean         p50         p90         p99
  TTFT (from admit)          31.22 ms     31.49 ms     32.48 ms     35.72 ms
    queue wait                4.59 ms      4.42 ms      7.85 ms     21.02 ms
    TTFT (from arrival)      35.80 ms     35.31 ms     39.89 ms     53.52 ms
  TPOT (per token)            9.11 ms      9.42 ms     10.25 ms     10.40 ms
  ITL (inter-token)           9.14 ms      9.30 ms     10.29 ms     29.94 ms
  End-to-end latency       9360.36 ms   9669.08 ms  10526.00 ms  10673.39 ms
  Batch packing: avg batch 10.5 (max 19) | avg prefill/decode reqs 0.0/10.5 | ...
```

Reading it:

- **Two TTFT rows.** *From admit* is the engine's behaviour once the request is
  running; *from arrival* adds the queue wait and is what a client actually
  experiences. Under load these diverge sharply, and quoting the first as if it
  were the second is the classic way to make a saturated system look healthy.
- **`[SATURATED]`** on the arrivals line means offered load exceeds capacity.
  Latencies then reflect an unbounded queue and are not meaningful as a
  operating point — lower `--request-rate` below the reported max sustainable
  rate.
- **Batch packing** shows what the scheduler actually assembled, which is how
  you tell a latency problem caused by batching from one caused by queueing.

Arrival options: `--arrival-model` takes `closed` (no queue, steady state
only), `poisson`, or `deterministic`; `--des-burstiness` makes arrivals gamma-
distributed (1.0 = Poisson, lower = burstier); `--des-range-ratio` spreads
per-request lengths around the configured ISL/OSL. Add `--des-sweep` to sweep
offered load and emit a throughput-versus-latency curve rather than a single
point.

### Simulate a fleet

Several engine replicas behind a router, with a real content-addressed KV block
cache per replica:

```bash
inferasim inference ... \
  --request-rate 6 --arrival-model poisson --des-num-requests 200 \
  --des-instances 4 --des-routing kv \
  --des-num-prefixes 8 --des-prefix-len 2048 --des-block-size 512
```

```
  Fleet: 4 instance(s), routing=kv | prefix pool: 8 prefixes
  KV block cache: block_size=512 tok, capacity/instance=unbounded, evictions=0, block-reuse=48.0%
  Prefix-cache hit rate: 96.0% of requests (avg 1966 cached tok/req; per-instance 95–97%)
```

A prompt is an ordered sequence of block-hash ids, and a hit is the longest
*contiguous leading run* of blocks already resident. The cache is finite and
LRU-evicts under pressure, so **hit rate is an emergent property** of content,
capacity and routing rather than a number you supply.

Routing policies (`--des-routing`):

| Policy | Behaviour |
|---|---|
| `kv` | route to the replica holding the most of the request's leading blocks (ties → least loaded). Maximises reuse. |
| `prefix_aware` | consistently hash the leading block to a home replica, so same-prefix requests co-locate. Misses ≈ number of prefixes, independent of fleet size. |
| `round_robin` / `random` | ignore locality, so every replica re-warms every prefix. Misses ≈ prefixes × replicas. |

The trade-off is real in both directions: locality-seeking policies maximise
reuse but can overconcentrate a hot prefix onto one replica, raising its decode
pressure and lengthening the makespan. The report shows pooled latencies
alongside the per-replica hit-rate spread so you can see both halves.

Set `--des-instances 1` to study a single engine's automatic prefix caching as
temporal reuse across a stream. Add `--des-kv-blocks N` to cap per-replica
capacity and study eviction pressure; `--des-prefix-zipf` skews prefix
popularity the way a few hot system prompts dominate real traffic.

### Replay a real trace

```bash
inferasim inference ... \
  --des-mooncake-trace trace.jsonl \
  --des-instances 4 --des-routing kv --des-block-size 512
```

A Mooncake trace is JSONL/JSON with `timestamp` (ms), `input_length`,
`output_length` and `hash_ids` (the block-hash sequence). Requests sharing a
system prompt share leading `hash_ids`, which drives genuine content-addressed
reuse instead of an assumed rate. The trace supplies its own arrivals, so
`--request-rate` is not needed.

`--des-workload-file` replays a simpler JSON/CSV workload without hash ids.

### Model disaggregated serving

Give prefill and decode their own pools and parallel shapes, and charge the KV
handoff between them:

```bash
inferasim inference ... --disaggregate \
  --prefill-tp 4 --prefill-ep 4 --prefill-replicas 1 \
  --decode-tp 8  --decode-ep 8  --decode-replicas 2 \
  --kv-transfer-bw-gbps 400
```

This is where `--decode-admission-steps` matters most: with prefill off the
critical path, what remains visible in TTFT is how long a finished prefill waits
to join a decode batch.

### Sweep the configuration space

From Python, for scripted searches:

```python
from infera.projection.core.projection.inference_projection.sweep import sweep

res = sweep(
    "gpt_oss_120B",
    tp=[1, 2, 4, 8], ep=[1, 2, 4, 8], pp=[1],
    concurrency=[1, 8, 32, 128],
    isl=1024, osl=1024,
    gpu_arch="mi355x", hbm_gb=288.0,
    valid=lambda tp, ep, pp: ep <= tp,      # your own legality rules
)

for p in res.points:
    if p.feasible:
        print(p.tp, p.ep, p.concurrency, round(p.ttft_ms, 1),
              round(p.tpot_ms, 2), round(p.decode_tps_per_gpu, 1))
```

Sweeps force `--profiling-mode simulate`, so they need **zero GPUs**.
Infeasible points are *kept* and annotated with `p.reason` rather than dropped,
so you can tell "did not fit" from "was never tried". Pass `workload=` to
project against your own experiment config instead of the packaged default.

### Search with the tuning agent

```bash
inferasim-tune --workload <workload.yaml> --target-cluster <cluster.yaml>
```

Two stages: a deterministic seed sweep for a warm start, then an LLM-driven
search that continues from the warm-started incumbent, proposing recipes and
scoring them through the projector. No GPU in the default path. Use
`--seed-only` to run just the deterministic stage. See
`agents/tuning_agent/` for configuration.

## Environment variables

Everything is `INFERASIM_*`. The ones you will actually set:

| Variable | Meaning |
|---|---|
| `INFERASIM_MODEL` | architecture preset (e.g. `gpt_oss_120B`) |
| `INFERASIM_TP` / `_EP` / `_PP` / `_CP` / `_VP` | parallel shape |
| `INFERASIM_GPU_ARCH` | target GPU architecture |
| `INFERASIM_ROOT` | config root, for resolving workload YAMLs kept outside the repo |
| `INFERASIM_ANCHOR_STORE` | directory of measured anchors |
| `INFERASIM_LADDER_MAX_GPUS` | confidence-ladder cap (default 4) |
| `INFERASIM_SEQ_LENGTH` | default sequence length |
| `INFERASIM_TEAM` / `_USER` / `_EXP_NAME` / `_WORKSPACE` | launcher identity fields |

Further `INFERASIM_*` variables exist for kernel-model and benchmark internals
(`INFERASIM_GEMM_BACKEND`, `INFERASIM_BENCH_*`, `INFERASIM_MOE_*`,
`INFERASIM_DEBUG_*` and others). They are advanced overrides, documented at
their read sites.

## Troubleshooting

**Concurrency is far higher than I asked for.** Pass `--max-concurrency`.
`--inference-batch-size` does not cap serving concurrency; see
[Control concurrency](#control-concurrency).

**Every sweep point is infeasible.** Check `p.reason`. The usual causes are an
HBM budget too small for the weights at that parallel shape, or a `valid=`
predicate rejecting everything.

**The DES says `[SATURATED]`.** Offered load exceeds capacity, so the queue
grows without bound and the latency percentiles describe a backlog rather than
an operating point. Lower `--request-rate` below the reported max sustainable
rate, or add replicas with `--des-instances`.

**Prefix hit rate is 0% in the DES.** Requests have no shared content. Provide
either `--des-num-prefixes`/`--des-prefix-len` or a trace with `hash_ids`;
`--prefix-cache-hit-rate` is the analytical knob and does not drive the block
cache.

**Anchor results look wrong after changing dtype or backend.** An anchor
certifies its own execution regime. Change dtype, kernel backend or graph mode
and the previous anchor no longer transfers — harvest another.

## Repository layout

| Path | Contents |
|---|---|
| `cli.py` | the `inferasim` entry point (projection + DES) |
| `core/`, `modules/`, `platforms/` | projection engine and simulator |
| `agents/tuning_agent/` | recipe-search agent (`inferasim-tune`) |
| `configs/` | model and preset tree resolved at runtime |
| `examples/` | example experiment configs and workloads |
| `ARCHITECTURE.md` | how it works, and what it does not model |

The Megatron/training closure is intentionally **not** included; the inference
path never needs it.
