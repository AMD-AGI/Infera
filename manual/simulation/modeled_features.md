# Modeled features

```{admonition} One-pager
:class: tip
**What:** the map of what InferaSim actually models, grouped by concern, with
the flag that controls each. **Why:** the task pages show the common paths; this
page is so nothing is invisible. **Cost:** none — almost everything here
defaults to the engine's or the model's own behaviour.
```

`inferasim inference --help` is the authoritative list (133 flags). This page
groups them by what question they answer, so you can find the lever without
reading all of it. For what is deliberately *not* modeled, see
[Boundaries and verification](boundaries.md).

Most of these only need setting when you are modelling a deployment that
differs from the default. The ones that change an answer most often are marked
**⚑**.

## Two projections

`--inference-mode {performance,memory,both}` selects which runs; the default is
`both`.

| Projection | Reports |
|---|---|
| `performance` | TTFT, ITL/TPOT, throughput, interactivity, step latency, comm breakdown |
| `memory` | weights, KV cache, activation working set, total per-GPU HBM, and whether the recipe fits |

Memory is analytical at every parallelism and is never calibrated by a
measurement — see [Memory is outside the calibration loop](boundaries.md#memory-is-outside-the-calibration-loop).

## Serving model

| Flag | Meaning |
|---|---|
| `--serving-model {continuous,static}` ⚑ | `continuous` (default) models continuous batching with mixed prefill+decode steps, and therefore TPOT pollution. `static` is an idealized pure-decode batch with prefill charged once as TTFT. |

If you are comparing against a number from a real server, you want
`continuous`. `static` is for isolating decode from interference.

## Model architecture and attention

| Feature | Flags |
|---|---|
| Data-parallel attention (MLA) ⚑ | `--attention-dp-size` — see [the explanation](projection_runs.md#data-parallel-attention-mla) |
| Sliding-window / local attention | `--sliding-window` (KV tokens attended; `0` forces full attention), `--sliding-window-layer-fraction` (for models interleaving local and global layers) |
| Native sparse attention | `--sparse-attention-topk` — DeepSeek V3.2/V4-style NSA; attention scales toward `topk/context` at long context. `0` is dense. |
| Attention kernel library | `--attention-backend {aiter,triton,ck,hip}` — a compute multiplier relative to the Triton baseline |
| KV sizing horizon | `--max-context-len` — largest prompt+generated context for KV sizing, defaulting to input+output |

## Parallelism

| Feature | Flags |
|---|---|
| Core shape ⚑ | `INFERASIM_TP` / `_EP` / `_PP` / `_CP` / `_VP`, or `--target-ep-size` to override EP for the target |
| Attention DP | `--attention-dp-size` |
| Multi-node target | `--target-nodes` / `--target-num-nodes`, `--hardware-config` |
| Pipeline scheduling | `--pipeline-schedule-algorithm` (`auto`, `zerobubble`, `zbv-*`, `seaailab-ilp`, …), `--enable-zero-bubble`, `--num-virtual-stages-per-pipeline-rank`, `--micro-batch-size`, `--global-batch-size` |

## Precision

Four independent dtype axes, because real deployments mix them:

| Flag | What it governs |
|---|---|
| `--weight-dtype` ⚑ | resident weight precision — sizes the whole checkpoint |
| `--kv-cache-dtype` ⚑ | KV-cache precision, so it sets KV footprint and therefore concurrency |
| `--linear-weight-dtype` | the non-expert linears: attention projections and the dense MLP |
| `--moe-expert-dtype` | expert grouped-GEMM compute precision |
| `--act-quant-dtype` | precision activations are cast to before each low-precision GEMM (a memory-bound cast cost) |

Sizing a 4-bit checkpoint at 4 bits while streaming it at 8 is the classic
error here, which is why the linear and expert precisions are separable from
`--weight-dtype`.

## Mixture of experts

| Flag | Meaning |
|---|---|
| `--ep-load-balance` ⚑ | routing imbalance: hottest-rank / mean token load. `1.0` is perfectly balanced; above that inflates expert compute at EP>1. |
| `--redundant-experts` | extra replicated expert slots (EPLB) that reduce realized imbalance |
| `--moe-routing-skew` | Zipf exponent of the router's popularity law, which sets how many distinct experts a decode step touches — the weight-bandwidth term |
| `--moe-router-coverage` | measured router coverage as JSON: how many distinct experts the real router reaches per step, versus what independent per-token routing predicts |

## Speculative decoding

| Flag | Meaning |
|---|---|
| `--speculative-num-tokens` ⚑ | draft tokens proposed per verify step (`0` disables) |
| `--speculative-acceptance-rate` ⚑ | expected per-token acceptance in [0,1] |
| `--speculative-draft-cost-factor` | draft forward cost per proposed token, as a fraction of one target decode step. Default `0`, i.e. draft cost ignored. |

Speculation is a **regime-defining** parameter, so a recipe that turns it on
needs its own anchor — see [Regimes](anchors.md#regimes-when-an-anchor-stops-transferring).

## Collectives and communication

| Flag | Meaning |
|---|---|
| `--comm-model {explicit,builtin}` | `explicit` (default) gives the knob-driven breakdown; `builtin` folds comm into layer time with no breakdown |
| `--tp-allreduce-algo` | force `ring`, `one_shot`, `two_shot`, `hierarchical` (default `auto` = fastest) |
| `--ep-a2a-algo` | force `direct`, `single_shot`, `hierarchical` (default `auto`) |
| `--prefill-comm-overlap` / `--decode-comm-overlap` ⚑ | fraction of collective time hidden behind compute, [0,1], default `0` |
| `--tp-allreduce-efficiency` / `--ep-a2a-efficiency` | time multipliers (<1 = a fused or overlapped speedup) |
| `--quick-reduce` | ROCm quick-reduce: low-latency quantized all-reduce for small messages |
| `--fuse-rmsnorm-allreduce` | hides part of the TP all-reduce behind the norm |
| `--enable-deepep` | DeepEP async all-to-all overlapped with compute |
| `--sync-free-stage` | SyncFree MoE stage 1–3 (fused router, +DeepEP+grouped, +fused act); auto-enables DeepEP |

The report's exposed-communication breakdown is what tells you whether any of
this is worth reaching for.

## Scheduler, graphs and step overheads

| Flag | Meaning |
|---|---|
| `--max-num-batched-tokens` ⚑ | scheduler per-step token budget; oversized steps split |
| `--chunked-prefill-size` ⚑ | per-request per-step prefill allowance, which decides how many steps a prompt takes |
| `--cudagraph-mode {none,piecewise,full}` ⚑ | capture preset; sets per-step overhead and mixed-batch penalty unless those are given explicitly |
| `--mixed-batch-penalty` | extra cost fraction for mixed prefill+decode steps |
| `--decode-admission-steps` | admission granularity in decode steps — how long a finished prefill waits to join a decode batch |
| `--decode-step-overhead-us` | fixed per-step host/launch overhead; CUDA graphs reduce it |
| `--kernel-launch-latency-us`, `--kernels-per-layer` | the launch-bound decode floor in pure-simulate mode; disabled by graph capture |
| `--decode-kernel-occupancy-us` | minimum time a kernel holds the device regardless of data touched — *not* cancelled by graph capture |
| `--fused-kernels` | fused RMSNorm / RoPE / quant / KV-store kernels that cut per-step launch overhead |
| `--gemm-backend origami` | GEMM simulation backend, used only in `simulate` and `both` |

Cudagraph mode is regime-defining. So is the attention backend.

## KV cache and memory

| Flag | Meaning |
|---|---|
| `--hbm-capacity-gb` ⚑ | per-GPU HBM; bounds what fits and the max concurrency |
| `--kv-cache-memory-fraction` ⚑ | fraction of HBM the engine may claim (vLLM `gpu_memory_utilization` / SGLang `mem_fraction_static`) |
| `--kv-block-size` | paged-KV page size in tokens; context rounds up to whole blocks, inflating KV bytes |
| `--prefix-cache-hit-rate` | declared prefix reuse on the analytical path |
| `--kv-offload-gb-per-gpu` | host DRAM per GPU as a second KV tier (TRT-LLM native / SGLang HiCache `dram`) |
| `--kv-offload-bw-gbps` | host↔device bandwidth for that tier (PCIe 5 x16 ≈ 64; a coherent host link ≈ 900) |

The offload tier is priced analytically as a bounded allowance plus its transfer
bandwidth. The DES has one device-resident tier — see
[Boundaries](boundaries.md#what-is-deliberately-not-modelled).

## Host-side costs on the latency path

These exist because a client-measured TTFT is not the forward pass. Quoting a
measured number against a projection that omits them is how a model reads
systematically early.

| Flag | Meaning |
|---|---|
| `--request-overhead-ms` | fixed per-request host cost on the TTFT path: accept, parse, admit, prefix lookup, KV allocation, stream open |
| `--tokenize-overhead-us` | per-prompt-token server-side tokenization, inside the TTFT clock |
| `--detokenize-overhead-us` | per-output-token detokenize + stream, added to ITL/TPOT and e2e but not throughput |
| `--stream-interval` | output tokens buffered per flush — the client's first token only arrives after the first flush |
| `--prefill-rate-us-per-token`, `--prefill-rate-lo-tokens`, `--prefill-rate-hi-tokens` | a measured prefill rate and the two prompt lengths it was fit over. A rate without its span is meaningless, so the span is required alongside it. |
| `--sampling-top-k`, `--sampling-top-p`, `--sampling-temperature`, `--no-sampling` | logits post-processing cost; top-k adds a partial sort, top-p a threshold+renormalize, temperature a fused scale |

## Discrete-event simulator

Beyond the common flags on [Simulation runs](simulation_runs.md) and
[Fleet and routing](fleet_and_routing.md):

| Flag | Meaning |
|---|---|
| `--des-exclusive-prefill` | the engine does not co-schedule prefill with decode: a prefill batch carries one request and the whole token budget |
| `--des-max-prefill-seqs` | cap on new sequences entering one prefill step (`0` = only the token budget limits admission) |
| `--des-kv-cache-tokens` | total KV token-slot pool; admission reserves full ISL+OSL per request, so shortage head-of-line blocks |
| `--des-cache-slots` | legacy alias for `--des-kv-blocks` |
| `--des-warmup-frac` | fraction of requests, in completion order, excluded from the reported distribution (default `0.1`) — a reporting convention, not engine behaviour |
| `--des-seed` | RNG seed for arrivals and acceptance sampling (default `0`, so runs are reproducible) |
| `--des-dump-steps` | write per-step batch-composition records and a packing summary to a JSON path |

`--des-dump-steps` is the one to use when you disbelieve a latency number: it
shows what was in every step.

## Disaggregation

Each pool takes its own full shape, not just its own TP:

| Flag | Meaning |
|---|---|
| `--disaggregate` | enable separate prefill/decode pools |
| `--prefill-tp` / `--prefill-ep` / `--prefill-pp` / `--prefill-replicas` | prefill pool shape |
| `--decode-tp` / `--decode-ep` / `--decode-pp` / `--decode-replicas` | decode pool shape |
| `--prefill-attention-dp` / `--decode-attention-dp` | per-pool attention-DP degree, since deployments commonly run DP attention on one pool only |
| `--transfer-backend {nixl,mooncake,mori}` | KV-transfer engine preset (sets link bandwidth and latency) |
| `--kv-transfer-bw-gbps` / `--kv-transfer-latency-us` | override that preset |

See [Disaggregated serving](disaggregation.md), including what this path does
not model.

## Calibration

| Flag | Meaning |
|---|---|
| `--profiling-mode` ⚑ | `benchmark` (default), `simulate`, `both` |
| `--anchor-store` ⚑ | directory of warmup measurements; the closest same-regime anchor calibrates this projection |
| `--load-benchmark` | use one specific anchor artifact |
| `--save-benchmark` | write the anchor artifact |
| `--load-benchmark-scaling` | extra artifacts at other `--benchmark-gpus` values, to *fit* how the step scales with TP instead of assuming TP⁻¹ |
| `--decode-floor-benchmark` | a sharded artifact whose measured decode step defines the hardware latency floor |
| `--benchmark-gpus` | how many GPUs the warmup measures on |
| `--bench-model`, `--bench-serving-backend`, `--inference-bench-layers` | what to serve while measuring, under which engine, and how many layers to chain per timing stack |
| `--gpu-clock-mhz` | override the compute clock from the hardware profile |

See [Anchors and calibration](anchors.md).

## Cost

| Flag | Meaning |
|---|---|
| `--gpu-cost-per-hour` | prices projected throughput as cost per million tokens, charging the whole replica |

## Workload shape

Set on every run, and used in the examples throughout these pages:

| Flag | Meaning |
|---|---|
| `--config` (alias `--exp`) | the experiment YAML |
| `--input-len` / `--output-len` ⚑ | prompt and generation lengths in tokens |
| `--inference-batch-size` | sequences per decode forward — **not** a cap on serving concurrency, which is `--max-concurrency` |
| `--max-concurrency` ⚑ | resident sequences, i.e. the operating point |

## Accepted aliases

| Alias | Canonical |
|---|---|
| `--exp` | `--config` |
| `--prefix-hit-fraction` | `--prefix-cache-hit-rate` |
| `--des-new-seqs-per-step` | `--des-max-prefill-seqs` |
| `--des-cache-slots` | `--des-kv-blocks` |
| `--target-num-nodes` | `--target-nodes` |

`--profile-only` and `--save-profiling` exist but have suppressed help; they are
internal profiling plumbing rather than modelling knobs.

## Next steps

- Vary any of these automatically instead of by hand: [Tuning agent](tuning_agent.md).
- What none of them can fix: [Boundaries and verification](boundaries.md).
