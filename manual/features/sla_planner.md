# SLA planner

```{admonition} One-pager
:class: tip
**What:** work out, every few minutes, how many prefill and decode replicas the
fleet needs to hold its TTFT and ITL targets. **Why:** a pool sized for peak
wastes GPUs the rest of the day, and one sized for average misses the SLA under
load. **Cost:** an offline profiling sweep of your model, plus one small process.
```

```{admonition} It decides; it does not resize
:class: important
The planner publishes a replica count and stops there. Nothing is scaled
automatically yet — read the decision from the log and act on it yourself.
`SlaPlanner` accepts an optional async decision callback, which is the single
extension point for a future Kubernetes or Slurm actuator.
```

```{graphviz}
digraph sla_planner {
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc" fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.2,0.12"];
  edge [fontname="Helvetica,Arial,sans-serif" fontsize=10];

  SRV [label="Server\n/metrics"];
  PROF [label="profile.json\noffline sweep" fillcolor="#f4f4f4" color="#999999"];
  PLN [shape=diamond style=filled fillcolor="#fff3cd" color="#caa300"
       label="Planner\ncorrect → forecast →\nsolve for replicas"];
  OUT [label="decision\nlog" fillcolor="#e2efdd" color="#6a9a4a"];

  SRV -> PLN [label="TTFT, ITL, ISL, OSL"];
  PROF -> PLN [style=dashed color="#8a8a8a"];
  PLN -> OUT [penwidth=1.8 color="#5577cc"];
}
```

## The idea

Prefill and decode are bound by different things, which is the whole reason to
disaggregate them — and also the reason a single replica count cannot serve both.
Prefill is compute-bound and roughly single-batched, so what it needs is set by
the arriving *prompt* token rate. Decode is bandwidth-bound and batched, so what
it needs is set by the *generated* token rate and by how fast ITL degrades as KV
cache fills. The two move independently: a shift from short prompts with long
answers to long prompts with short answers changes which pool is the bottleneck
without changing the request rate at all.

The planner closes that loop. Each interval it measures what the fleet actually
delivered, compares it against what an offline sweep of the same model predicted,
forecasts the next interval, and solves each pool separately for the replica
count that meets the target.

## Step 1 — profile the model, once

The planner cannot invent a performance model; it needs to know how this model on
this GPU behaves. Bring up a deployment of **exactly one replica per role**, then
sweep it:

```bash
pip3 install ".[planner]"     # adds numpy; nothing else

python3 -m infera.planner.profile \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B \
  --max-kv-tokens 262144 \
  --isl 512,1024,2048,4096 \
  --context-length 1024,4096,16384 \
  --kv-usage 0.1,0.3,0.5,0.7,0.9 \
  --output profile.json
```

`--max-kv-tokens` is the engine's total KV capacity in tokens — GPU blocks ×
block size, which the engine prints at startup. Everything else has a default.

Two sweeps run:

- **Prefill** — one request at a time at each `--isl`, so nothing queues,
  recording the TTFT of an unqueued request and the prefill throughput it implies.
  Prompts are random and never repeated, because a prefix-cache hit would measure
  the cache instead of the engine.
- **Decode** — the `context_length × kv_usage` grid. KV utilisation isn't
  something a client can ask for, so it is reached through concurrency: holding
  `c` requests of `L` tokens each in flight occupies `c × L / max_kv_tokens` of
  the cache, so `c = u × max_kv_tokens / L` lands on the target. Aggregate
  throughput is taken from the steady state (`c / itl`) rather than from wall
  time, which would fold the prefill phase in.

Re-run it after anything that moves the curves: a different engine version,
quantisation, tensor-parallel width, or GPU model.

## Step 2 — run the planner

A separate process, like `infera.kvd`. It never sits in the request path.
Start every Infera server replica with `--enable-sla-metrics` (or
`INFERA_ENABLE_SLA_METRICS=1`) so it records the four planner inputs. This is
opt-in: without it, ordinary inference requests are not modified and streaming
chunks do not enter the planner-specific parser. The instrumentation currently
requires the default Python router backend; startup rejects the flag with
`--router-backend=rust` rather than silently exposing incomplete metrics.

```bash
python3 -m infera.planner \
  --metrics-url http://127.0.0.1:8000/metrics \
  --model Qwen/Qwen3-8B \
  --profile-results ./profile.json \
  --ttft 500 --itl 50 \
  --adjustment-interval 180
```

Each interval it logs a line like:

```text
decision: prefill 2->3, decode 4->4 (corrections p=1.842 d=1.031;
          load req=612.0 isl=2048 osl=256)
```

If the correction factors are far from 1.0, the profiling data does not describe
this deployment and every decision built on it is suspect — re-profile before
acting on the counts.

## How it decides

### 1. Correct the model

Profiling measures one request at a time on an idle engine. Production does not
look like that, so each interval the planner derives a correction factor per
phase by dividing what it observed by what the model predicted for the same
workload:

```text
prefill_correction = observed_ttft / predicted_ttft(isl)
decode_correction  = observed_itl  / predicted_itl(concurrency, isl + osl/2)
```

Prefill normally lands **above 1.0**, because real TTFT includes queueing the
sweep never saw. It lands **below 1.0** when prefix-cache hits mean the engine
prefills fewer tokens than the prompt length implies — which is exactly what
[KV-aware routing](kv_aware_routing.md) is for, so the two features compound.

Decode should sit **near 1.0**. Sustained drift there usually means chunked
prefill in the decode engine is stealing decode steps.

They are included in every decision log line; a sustained value far from 1.0 is
the first sign that the profile no longer describes the deployment.

### 2. Carry the observed load forward

The minimal planner assumes the next interval repeats the one just observed.
This keeps the online loop deterministic and dependency-free. A predictor can
later transform `(request_count, ISL, OSL)` before the sizing functions without
changing the profile, metrics source, or future decision callback.

### 3. Solve each pool

```text
prefill: max(
  ceil(req × isl / interval × min(1, p_correction) / thpt_per_gpu(isl) / gpus_per_replica),
  ceil(queue_ms × replicas_now / (ttft_target − service_ms)),
)
decode:  ceil(req × osl / interval / thpt_per_gpu(itl_target / d_correction) / gpus_per_replica)
```

The `min(1, ...)` on the prefill correction is deliberate. A correction above 1.0
means requests are queueing, and adding replicas is what fixes queueing — letting
it also multiply the demand would count the same problem twice.

That leaves throughput alone unable to honour `--ttft`, which is what the second
prefill term is for. The same correction factor splits the observed TTFT into the
two parts that behave differently:

```text
service_ms = profiled_ttft(isl) × min(1, p_correction)      # cost once on an engine
queue_ms   = profiled_ttft(isl) × max(0, p_correction − 1)   # cost of waiting for one
```

Replicas divide the queue and leave service untouched, so holding the target
takes `queue_ms × replicas_now / (target − service_ms)` of them. Two cases impose
nothing: a fleet with no queueing (`p_correction ≤ 1`) already meets any target
its service time allows, and a target below `service_ms` is unreachable at any
replica count — the planner logs that and sizes for throughput instead of
scaling into a wall.

Two guardrails keep a noisy window or a near-impossible target from producing
an order-of-magnitude recommendation:

- A TTFT target must leave queueing headroom of at least **20% of the profiled
  service time**. Below that margin, profiling noise dominates the denominator
  and the linear queue model is not trustworthy, so the planner logs a warning
  and uses throughput sizing only.
- Either pool may grow by at most **4× its observed replica count per
  decision**. The log retains the unconstrained request and the limited target,
  allowing a later interval to continue scaling if the signal persists.

Decode works the other way round: the correction tightens the ITL *target*
instead of the demand, and the model is inverted to find the highest per-GPU
throughput that still fits. A deployment running at twice the profiled ITL is
aimed at half the target, and lands on the real one. The whole KV-usage axis is
searched rather than bisected, because decode throughput typically peaks partway
up and falls off as the cache fills — the most loaded compliant operating point
is often not the fastest one.

## Profiling data format

`--profile-results` takes what the sweep writes. Hand-editing it is supported;
the loader validates the shapes and refuses to start on a mismatch.

```json
{
  "prefill": {
    "isl":          [512, 1024, 2048, 4096],
    "ttft_ms":      [60, 110, 215, 430],
    "thpt_per_gpu": [8500, 9300, 9500, 9500]
  },
  "decode": {
    "kv_usage":       [0.1, 0.3, 0.5, 0.7, 0.9],
    "context_length": [1024, 4096, 16384],
    "itl_ms":       [[11, 13, 16, 21, 30], [13, 16, 20, 27, 39], [18, 23, 30, 41, 60]],
    "thpt_per_gpu": [[900, 2300, 3100, 3300, 3000], [750, 1900, 2500, 2600, 2300],
                     [520, 1300, 1700, 1750, 1500]],
    "max_kv_tokens": 262144
  },
  "prefill_engine_num_gpu": 1,
  "decode_engine_num_gpu": 1
}
```

| Field | Meaning |
|-------|---------|
| `prefill.isl` | Prompt lengths swept, ascending. |
| `prefill.ttft_ms` | TTFT of a single unqueued request at that length. |
| `prefill.thpt_per_gpu` | Prefill tokens/s one GPU sustains at that length. |
| `decode.kv_usage` | KV-cache utilisation fractions swept, ascending. |
| `decode.context_length` | Context lengths swept, ascending. |
| `decode.itl_ms` | `[i][j]` = ITL at `context_length[i]`, `kv_usage[j]`. |
| `decode.thpt_per_gpu` | `[i][j]` = decode tokens/s/GPU at the same point. |
| `decode.max_kv_tokens` | Engine's total KV capacity in tokens. |
| `*_engine_num_gpu` | GPUs per replica, i.e. the engine's parallel width. |

The decode block must be a **full rectangular grid** — every context length
crossed with every KV usage. That requirement is what lets the planner
interpolate with numpy alone instead of pulling in scipy for scattered-point
interpolation. Queries outside the swept range are clamped to the nearest edge
rather than extrapolated: extending a saturating throughput curve invents
capacity the hardware does not have.

## Metrics

The server exposes the four SLA signals the planner consumes, labeled by
`router` and `model`. The online planner reads only `router="disagg"` and the
model passed through `--model`:

| Metric | Notes |
|--------|-------|
| `infera_time_to_first_token_seconds` | Dispatch to first token. For PD this spans prefill + KV transfer + the decode engine's first forward pass. Streaming replies only. |
| `infera_inter_token_latency_seconds` | `(stream time − TTFT)` spread over the generated tokens. Requests producing at least two tokens. |
| `infera_input_sequence_tokens` | ISL. |
| `infera_output_sequence_tokens` | OSL. Its `_count` is also the planner's request rate, so the rate is over exactly the requests that contributed the averages beside it. |

All four are recorded only for successful requests: a 5xx has no latency worth
averaging, and including it would drag the window average toward zero.

```{admonition} Streaming usage
:class: note
The Infera server asks supported engines for final streaming `usage`, making ISL
and OSL exact even under round-robin routing. If an engine omits usage, OSL falls
back to content-bearing SSE frame counts and ISL to the router's KV block count.
```

```{admonition} Scraping more than one server replica
:class: warning
The planner keeps a separate cumulative baseline for every `--metrics-url`, then
sums the per-endpoint deltas. Pass the flag once per server replica. Pointing it
at a load-balanced ClusterIP Service does not work because consecutive scrapes
can hit different pods. If an endpoint disappears or returns, the planner
resets the baseline and skips that interval rather than mixing lifetime counters
into one decision window.
```

## When the planner does nothing

Intervals are skipped rather than guessed at and the reason is written to the
planner log:

| Reason | Meaning |
|--------|---------|
| `no_metrics` | The first interval (cumulative counters need a baseline), every replica unreachable, or a counter reset from a server restart. |
| `no_traffic` | No completed request, or none that generated tokens. Nothing to conclude about the SLA. |
| `no_decode_workers` | Traffic arrived but no decode replicas are registered, so the decode model cannot be calibrated. |
| `no_latency_samples` | Traffic arrived but neither TTFT nor ITL was recorded, so there is nothing to correct the model against. A whole window of this means the clients are not streaming. |
| `model_error` | The profiling data predicts a non-positive latency, i.e. it is degenerate. |

## Limitations

- **Disaggregated deployments only.** The planner sizes a prefill pool and a
  decode pool. Aggregated (`role: mixed`) fleets are not covered.
- **Streaming clients only.** A non-streaming reply arrives in one piece, so it
  has no observable first-token boundary and contributes neither TTFT nor ITL.
  Both corrections are derived from those two, so a fleet serving only
  non-streaming requests gives the planner nothing to calibrate against and
  every interval is skipped as `no_latency_samples`.
- **The TTFT queue model is linear.** Real queueing is non-linear. The 20%
  headroom check and 4× step limit prevent extreme decisions, but production
  actuation should still add cooldown and observe the next completed interval.
- **The window is in memory.** A planner restart forfeits one interval while it
  re-establishes the baseline scrape.

## See also

- [Scaling a fleet](scaling.md) — what actually happens when a replica count
  changes, and how long each direction takes.
- [Prefill-decode disaggregation](pd_disaggregation.md) — the deployment shape
  the planner sizes.
- [KV-aware routing](kv_aware_routing.md) — why the prefill correction factor
  often drops below 1.0.
