# SLA planner

```{admonition} One-pager
:class: tip
**What:** resize the prefill and decode pools every few minutes so the fleet
holds its TTFT and ITL targets. **Why:** a pool sized for peak wastes GPUs the
rest of the day, and one sized for average misses the SLA under load.
**Cost:** an offline profiling sweep of your model, plus one small process.
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
  CRD [label="InferaDeployment\nspec.services[*].replicas" fillcolor="#e2efdd" color="#6a9a4a"];
  POOLS [label="prefill / decode pools" fillcolor="#e2efdd" color="#6a9a4a"];

  SRV -> PLN [label="TTFT, ITL, ISL, OSL"];
  PROF -> PLN [style=dashed color="#8a8a8a"];
  PLN -> CRD [label="merge patch" penwidth=1.8 color="#5577cc"];
  CRD -> POOLS [label="operator reconcile"];
  POOLS -> SRV [style=dashed color="#8a8a8a"];
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

## Running it

The planner is a separate process, like `infera.kvd`. It never sits in the
request path, so restarting it or losing it does not affect serving.

```bash
pip3 install ".[planner]"     # adds numpy; nothing else

python3 -m infera.planner \
  --metrics-url http://127.0.0.1:8000/metrics \
  --profile-results ./profile.json \
  --ttft 500 --itl 50 \
  --adjustment-interval 180 \
  --connector virtual --etcd-endpoint 127.0.0.1:2379
```

On Kubernetes, apply
[`examples/k8s-deployments/sla-planner.yaml`](https://github.com/AMD-AGI/Infera/blob/main/examples/k8s-deployments/sla-planner.yaml)
next to a PD deployment. It carries the ServiceAccount, the `get`/`patch` Role on
`inferadeployments`, a ConfigMap for the profiling data, and the planner
Deployment.

```{admonition} Start with --no-operation
:class: important
`--no-operation` observes and logs the decision it *would* have made without
touching anything. Run a few intervals of real traffic through it first: if the
correction factors are far from 1.0, the profiling data does not describe this
deployment, and every decision built on it would be wrong.
```

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

Both are exported as `infera_planner_correction_factor{phase=...}`; they are the
first thing to look at when a decision seems wrong.

### 2. Forecast the next interval

Scaling on the interval that just ended always arrives one interval late, which
on a rising ramp is precisely when the SLA breaks. `--load-predictor constant`
assumes the next interval repeats the last one, which is the right choice when
the interval is long relative to how fast traffic moves. `--load-predictor ewma`
smooths through single-interval noise at the cost of lagging a sustained ramp.

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
service_ms = profiled_ttft(isl) × min(1, p_correction)   # cost once on an engine
queue_ms   = profiled_ttft(isl) × max(0, p_correction − 1)  # cost of waiting for one
```

Replicas divide the queue and leave service untouched, so holding the target
takes `queue_ms × replicas_now / (target − service_ms)` of them. Two cases impose
nothing: a fleet with no queueing (`p_correction ≤ 1`) already meets any target
its service time allows, and a target below `service_ms` is unreachable at any
replica count — the planner logs that and sizes for throughput instead of
scaling into a wall.

Decode works the other way round: the correction tightens the ITL *target*
instead of the demand, and the model is inverted to find the highest per-GPU
throughput that still fits. A deployment running at twice the profiled ITL is
aimed at half the target, and lands on the real one.

### 4. Clamp

`--min-endpoint` keeps both pools alive; `--max-gpu-budget` caps the total.
Hitting the budget means the SLA is not reachable with the GPUs on hand — the
planner cuts both pools proportionally, logs a warning, and counts it in
`infera_planner_gpu_budget_exceeded_total`. It is a signal to buy GPUs or relax
the target, not something to leave running unnoticed.

Set `--adjustment-interval` comfortably above the time a replica takes to become
ready. Scaling is non-blocking, so an interval shorter than engine startup issues
a new decision while the previous one is still rolling out.

## Profiling data

The planner cannot invent a performance model; it needs to know how this model on
this GPU behaves. That comes from an offline sweep, handed over as one JSON file
via `--profile-results`.

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

To produce the numbers, sweep with
[`bench/serve_standard/bench_vllm_serve.sh`](https://github.com/AMD-AGI/Infera/blob/main/bench/serve_standard/README.md)
against a single engine replica:

- **Prefill** — one request at a time (concurrency 1) at each ISL, recording TTFT
  and prompt tokens/s.
- **Decode** — sweep concurrency at each ISL, converting concurrency to
  `kv_usage = concurrency × context_length / max_kv_tokens`, and recording the
  mean ITL and generated tokens/s.

Re-profile after anything that moves the curves: a different engine version,
quantisation, tensor-parallel width, or GPU model.

## Actuation

| `--connector` | What it does |
|---------------|--------------|
| `kubernetes` | Merge-patches `spec.services.{prefill,decode}.replicas` on the named `InferaDeployment`. The operator's existing reconcile turns that into pod counts, so nothing in the deployment changes. Needs `get` and `patch` on `inferadeployments`. |
| `virtual` | Writes the decision to etcd at `<prefix>/planner/decision` and stops. For bare metal, Docker, or a bespoke scheduler. |

Counts are absolute, never deltas, in both cases. A connector that drops a
decision self-corrects on the next interval rather than drifting, and a `virtual`
consumer that misses one has nothing to catch up on — applying the newest is
always correct. The `decision_id` in the etcd payload increments per write so a
consumer can tell a fresh decision from a re-read:

```json
{"num_prefill_workers": 3, "num_decode_workers": 5, "decision_id": 42,
 "observed_prefill": 2, "observed_decode": 5, "gpu_budget_exceeded": false}
```

## Metrics

The server exposes the four SLA signals the planner consumes, per router mode:

| Metric | Notes |
|--------|-------|
| `infera_time_to_first_token_seconds` | Dispatch to first token. For PD this spans prefill + KV transfer + the decode engine's first forward pass. Streaming replies only. |
| `infera_inter_token_latency_seconds` | `(stream time − TTFT)` spread over the generated tokens. Requests producing at least two tokens. |
| `infera_input_sequence_tokens` | ISL. |
| `infera_output_sequence_tokens` | OSL. |

All four are recorded only for successful requests: a 5xx has no latency worth
averaging, and including it would drag the window average toward zero.

```{admonition} ISL and OSL are estimates on the streaming path
:class: note
A streaming reply carries no `usage` object, so OSL is counted as SSE data frames
(one token per frame) and ISL is derived from the KV block count the router
already computed — which rounds up to a block boundary, and is unavailable
entirely under round-robin routing. Both become exact when the engine reports
`usage`: always for non-streaming replies, and for streaming ones when the client
passes `stream_options: {"include_usage": true}`.
```

The planner exposes its own state on `--metrics-port` (9085 by default):
`infera_planner_correction_factor{phase}`,
`infera_planner_desired_replicas{role}`, `infera_planner_observed_ttft_seconds`,
`infera_planner_decisions_total{outcome}`,
`infera_planner_intervals_skipped_total{reason}` and
`infera_planner_gpu_budget_exceeded_total`.

```{admonition} Scraping more than one server replica
:class: warning
The planner sums samples across every `--metrics-url` before windowing, so pass
the flag once per server replica. Pointing it at a single ClusterIP Service does
not work: the Service load-balances each scrape to an arbitrary pod, and
differencing one pod's counters against another's produces nonsense. The planner
detects the resulting negative delta and skips the interval, so the failure mode
is a planner that never decides anything.
```

## When the planner does nothing

Intervals are skipped rather than guessed at, each counted under
`infera_planner_intervals_skipped_total`:

| Reason | Meaning |
|--------|---------|
| `no_metrics` | The first interval (cumulative counters need a baseline), every replica unreachable, or a counter reset from a server restart. |
| `no_traffic` | No completed request, or none that generated tokens. Nothing to conclude about the SLA. |
| `no_decode_workers` | Traffic arrived but no decode replicas are registered, so the decode model cannot be calibrated. |
| `no_latency_samples` | Traffic arrived but neither TTFT nor ITL was recorded, so there is nothing to correct the model against. A whole window of this means the clients are not streaming. |
| `model_error` | The profiling data predicts a non-positive latency, i.e. it is degenerate. |

A decision that matches the current replica counts is also not applied — patching
the deployment with what it already has is churn the operator would reconcile as
a no-op.

## Limitations

- **Disaggregated deployments only.** The planner sizes a prefill pool and a
  decode pool. Aggregated (`role: mixed`) fleets are not covered.
- **Streaming clients only.** A non-streaming reply arrives in one piece, so it
  has no observable first-token boundary and contributes neither TTFT nor ITL.
  Both corrections are derived from those two, so a fleet serving only
  non-streaming requests gives the planner nothing to calibrate against and
  every interval is skipped as `no_latency_samples`.
- **One planner per deployment.** Two would fight over the same replica counts.
- **The window is in memory.** A planner restart forfeits one interval while it
  re-establishes the baseline scrape.
- **Profiling is manual.** There is no automated pre-deployment sweep yet; the
  JSON is produced by hand from a benchmark run.

## See also

- [Scaling a fleet](scaling.md) — what actually happens when a replica count
  changes, and how long each direction takes.
- [Prefill-decode disaggregation](pd_disaggregation.md) — the deployment shape
  the planner resizes.
- [KV-aware routing](kv_aware_routing.md) — why the prefill correction factor
  often drops below 1.0.
- [Operator](../components/operator.md) — the CRD the Kubernetes connector
  patches.
