# SLA planner

The SLA planner is a separate Infera process that turns recent request
measurements into target sizes for the prefill and decode pools. It is useful
when traffic shape changes over time and static replica counts either waste
accelerators or miss latency objectives.

```{admonition} Recommendation only
:class: important
The current planner does not modify a deployment. It writes a reviewable
decision to its log. An optional async callback on `SlaPlanner` is the boundary
for a future Kubernetes, Slurm, or operator integration.
```

## Infera data path

The planner is not in the inference request path:

```{graphviz}
digraph planner_path {
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fillcolor="#eef2f7" color="#5577cc" fontname="Helvetica,Arial,sans-serif" fontsize=11 margin="0.2,0.12"];
  edge [fontname="Helvetica,Arial,sans-serif" fontsize=10];

  CLIENT [label="streaming clients"];
  SERVER [label="Infera server\nrequest observations"];
  METRICS [label="/metrics\ncumulative histograms"];
  WINDOW [label="planner\nwindow deltas"];
  PROFILE [label="profile.json\ncapacity envelope" fillcolor="#f4f4f4" color="#999999"];
  DECISION [label="target pool sizes\nlog / callback" fillcolor="#e2efdd" color="#6a9a4a"];

  CLIENT -> SERVER;
  SERVER -> METRICS;
  METRICS -> WINDOW;
  PROFILE -> WINDOW [style=dashed];
  WINDOW -> DECISION;
}
```

Each server records successful streaming requests. The planner reads every
configured server endpoint directly, subtracts the previous cumulative values,
and combines only the work completed during that window. This design does not
require a Prometheus server and works in a small Docker or bare-metal setup.

## Measure the deployment

A capacity decision is meaningful only when it uses measurements from the same
model, engine build, accelerator, quantisation, and parallel layout as the
running deployment. Start exactly one prefill replica and one decode replica,
then run:

```bash
pip3 install ".[planner]"

python3 -m infera.planner.profile \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B \
  --max-kv-tokens 262144 \
  --isl 512,1024,2048,4096 \
  --context-length 1024,4096,16384 \
  --kv-usage 0.1,0.3,0.5,0.7,0.9 \
  --output profile.json
```

`--max-kv-tokens` is the engine's total KV capacity in tokens. The engine
usually reports it as GPU blocks multiplied by block size.

The command builds two datasets:

- The prompt curve sends isolated one-token completions at each requested input
  length. Unique random prompts avoid measuring a prefix-cache hit.
- The generation surface samples every requested mean-context and occupied-KV
  pair. The runner derives a whole request count for the target occupancy,
  launches that batch together, and measures steady-state token spacing.

The first request at each prompt length is a warm-up. Timed prompt trials use
their median. Generation cells use the mean of the completed requests.

### Profile document

The output remains a plain JSON file so it can be inspected, versioned with a
deployment recipe, or produced by another benchmark:

```json
{
  "prefill": {
    "isl": [512, 1024, 2048, 4096],
    "ttft_ms": [60, 110, 215, 430],
    "thpt_per_gpu": [8500, 9300, 9500, 9500]
  },
  "decode": {
    "kv_usage": [0.1, 0.3, 0.5, 0.7, 0.9],
    "context_length": [1024, 4096, 16384],
    "itl_ms": [
      [11, 13, 16, 21, 30],
      [13, 16, 20, 27, 39],
      [18, 23, 30, 41, 60]
    ],
    "thpt_per_gpu": [
      [900, 2300, 3100, 3300, 3000],
      [750, 1900, 2500, 2600, 2300],
      [520, 1300, 1700, 1750, 1500]
    ],
    "max_kv_tokens": 262144
  },
  "prefill_engine_num_gpu": 1,
  "decode_engine_num_gpu": 1
}
```

The decode arrays must form a complete rectangle: rows correspond to
`context_length` and columns to `kv_usage`. Infera performs bounded linear
interpolation on this measured surface. A query outside an axis uses its nearest
edge; it never extrapolates unmeasured capacity.

## Enable request observations

Start each Infera server with `--enable-sla-metrics`, or set
`INFERA_ENABLE_SLA_METRICS=1`. The instrumentation currently requires the
Python router backend. Startup rejects the option with
`--router-backend=rust` so an incomplete metric stream cannot silently produce
recommendations.

The planner consumes these histograms:

- `infera_time_to_first_token_seconds`: dispatch to the first streamed token.
- `infera_inter_token_latency_seconds`: generation time after the first token,
  divided across the remaining output tokens.
- `infera_input_sequence_tokens`: prompt tokens.
- `infera_output_sequence_tokens`: generated tokens.

All are labeled by `router` and `model`. Only successful requests are included.
When a supported engine emits final streaming usage, its exact input and output
counts replace router estimates. Otherwise, input length falls back to KV-block
count and output length to content-bearing SSE frame count.

## Start the planner

Run one planner process outside the server fleet:

```bash
python3 -m infera.planner \
  --metrics-url http://127.0.0.1:8000/metrics \
  --model Qwen/Qwen3-8B \
  --profile-results ./profile.json \
  --ttft 500 \
  --itl 50 \
  --adjustment-interval 180
```

Pass `--metrics-url` once for every server replica. Do not point it at a
load-balanced service: two consecutive scrapes could reach different pods and
their cumulative counters would not form a valid window.

The first scrape establishes a baseline. A decision appears after the next
complete interval:

```text
capacity decision: prefill 2->3, decode 4->4
  (latency ratios p=1.842 d=1.031; load req=612.0 isl=2048 osl=256)
```

## Capacity policy

For a window of `H` seconds, let `C` be completed requests, `I` the mean prompt
length, and `O` the mean generated length. The incoming work rates are:

```text
prompt_tokens_per_second     = C × I / H
generated_tokens_per_second  = C × O / H
```

The offline envelope supplies a prompt latency and capacity at `I`. It also
supplies generation latency and capacity at the estimated in-flight load and
mean context `I + O/2`.

The planner records two latency ratios:

```text
prompt_ratio      = measured_TTFT / envelope_prompt_latency
generation_ratio  = measured_ITL  / envelope_generation_latency
```

A ratio far from `1` means the offline envelope no longer matches the running
system. Queueing commonly raises the prompt ratio; prefix-cache reuse can lower
it. Generation drift can indicate that decode steps are competing with other
work. These values stay in every decision so operators can distinguish a
traffic change from stale profiling data.

### Prefill budget

The prompt pool first receives a throughput budget. A ratio below `1` reduces
the effective prompt work because the server processed less work than raw input
length implies. A ratio above `1` is not allowed to inflate work a second time:
its queueing component is handled separately.

```text
throughput_replicas =
  ceil(prompt_tokens_per_second × min(1, prompt_ratio)
       / measured_tokens_per_second_per_replica)
```

For overloaded windows, observed TTFT is separated into measured service time
and waiting time. Replicas can divide the wait but cannot make one prompt's
service faster. The queue budget therefore asks how many copies are needed to
fit the observed wait into the target's remaining allowance. The larger of the
throughput and queue budgets wins.

### Decode budget

The generation surface is searched for the fastest sampled operating point
whose latency fits `ITL_target / generation_ratio`. Every sampled KV occupancy
is considered because throughput need not move monotonically as the cache
fills. The generated token rate divided by that per-replica capacity gives the
decode budget.

If no measured point fits, the least-occupied point is returned and the planner
logs that the requested target is below the measured hardware floor.

### Safety limits

- Queue sizing is disabled when the TTFT target leaves less than 20% of prompt
  service time as waiting allowance. Near a zero denominator, small measurement
  noise would produce a very large recommendation.
- One decision can grow either pool by at most four times its currently
  observed replica count. A later complete window can continue the move if the
  signal persists.
- Pool targets never fall below one replica.

## Skipped windows

The current deployment is left unchanged when:

- this is the baseline scrape;
- every configured metrics endpoint is unavailable;
- an endpoint appears or disappears between scrapes;
- a cumulative counter moves backwards after a server restart;
- no completed generation work was observed;
- streaming TTFT or ITL samples are absent;
- no decode replica is registered; or
- the profile contains a non-positive latency.

Every case is logged. Skipping is safer than interpreting missing measurements
as zero latency or zero demand.

## Operational limits

- Only disaggregated deployments are sized; mixed pools are not a planner
  target.
- Latency evidence requires streaming replies, and ITL requires at least two
  generated tokens.
- The current workload projection is persistence: the next window is assumed
  to resemble the latest completed window.
- Observation baselines live in memory. Restarting the planner forfeits one
  interval.
- Decisions are recommendations until an actuator is attached.

## Related guides

- [Scaling a fleet](scaling.md)
- [Prefill-decode disaggregation](pd_disaggregation.md)
- [KV-aware routing](kv_aware_routing.md)
