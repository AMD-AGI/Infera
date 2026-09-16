# Simulation runs

```{admonition} One-pager
:class: tip
**What:** the discrete-event simulator — requests arrive on their own schedule,
queue, get batched, and finish. **Why:** it is the only way to get a p99, a
queue wait, or a saturation verdict. **Cost:** seconds instead of milliseconds,
and you have to say what the load is.
```

The [analytical projector](projection_runs.md) gives means at a fixed
concurrency. Add an arrival model and the same cost kernel is driven through an
event loop instead, producing distributions.

## Arrival-driven simulation

```bash
inferasim inference ... \
  --request-rate 1.5 --arrival-model poisson --des-num-requests 120
```

```text
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

The DES report is printed *in addition to* the analytical one, not instead of
it, so you can see both descriptions of the same configuration.

## Reading the report

**There are two TTFT rows, and the difference matters.** *From admit* is the
engine's behaviour once the request is running; *from arrival* adds the queue
wait and is what a client actually experiences. Under load these diverge
sharply, and quoting the first as if it were the second is the classic way to
make a saturated system look healthy.

**`[SATURATED]`** on the arrivals line means offered load exceeds capacity. The
latencies then reflect an unbounded queue and are not meaningful as an operating
point. Lower `--request-rate` below the reported max sustainable rate, or add
replicas with `--des-instances`.

**Batch packing** shows what the scheduler actually assembled, which is how you
tell a latency problem caused by batching from one caused by queueing.

## Arrival options

| Flag | Meaning |
|---|---|
| `--arrival-model` | `poisson`, `deterministic`, or `closed` (no arrival stream — see [closed loop](#reproduce-a-fixed-concurrency-benchmark)) |
| `--request-rate` | offered load in requests per second |
| `--des-num-requests` | how many requests to simulate |
| `--des-burstiness` | makes arrivals gamma-distributed: `1.0` is Poisson, lower is burstier |
| `--des-range-ratio` | spreads per-request lengths around the configured input/output lengths |
| `--des-sweep` | sweep offered load and emit a throughput-versus-latency curve rather than a single point |

`--des-sweep` is the one to reach for when the question is "what rate can this
configuration take?" rather than "how does it behave at this rate". It walks the
load axis and shows you where the knee is.

## Reproduce a fixed-concurrency benchmark

A harness run with `--max-concurrency C` has no arrival stream: `C` clients each
submit one request, block until it completes, then submit the next.
`--des-closed-loop` simulates exactly that, which makes TTFT the difference
between two simulated timestamps rather than a closed-form prefill time:

```bash
inferasim inference ... --max-concurrency 256 \
  --max-num-batched-tokens 8192 --chunked-prefill-size 256 \
  --des-closed-loop --des-num-requests 1024
```

This matters because the analytical path prices TTFT as prefill *service* time —
how long the forward pass over the prompt takes — while a harness reports
*response* time. Under a colocated engine a prefill chunk rides a scheduler step
that is simultaneously carrying the resident decodes, so the step it waits on
dilates with the decode batch. Measured TTFT is close to a constant number of
steps across three decades of concurrency; a standalone-prefill model holds it
constant in milliseconds and therefore reads progressively early as load rises.
The step loop already mixes prefill chunks with decodes, so driving it from a
closed load puts that dilation in the answer without a separate queue term.

```{admonition} Set --chunked-prefill-size
:class: important
It is the engine's per-request per-step prefill allowance (vLLM's
`long_prefill_token_threshold`) and it decides how many steps a prompt takes, so
it is the one input this needs. Left unset, a prompt prefills in a single step
and TTFT reads early for that reason instead. It is a *declared* input: read it
off the server's launch flags rather than fitting it, or the simulation just
relocates the error it was meant to remove.
```

Concurrency is the load axis here, so there is no offered rate, no saturation
flag — the population is bounded by the client count — and no `--des-sweep`. All
`C` clients start together, which is what the harness does and why its *mean*
TTFT carries an opening-burst transient: `mean/p50` reaches 3–5.6x at high
concurrency on real runs. Match `--des-num-requests` and the warmup you score
against to whatever the harness reports over.

## Next steps

- Drive the simulation from real traffic instead of a synthetic process:
  [Workloads and traces](workloads.md).
- Put several replicas behind a router: [Fleet and routing](fleet_and_routing.md).
- Understand what the event loop does not model:
  [Boundaries and verification](boundaries.md).
