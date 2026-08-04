# Customer benchmark — AgentX Case-A at concurrency 8

The customer-supplied agentic benchmark from
[ROCm/MAD PR #173](https://github.com/ROCm/MAD/pull/173) (`scripts/AgentX_CaseA/`),
replayed **unmodified** against exactly the deployment shape this kit ships.

**This kit ships no runner for it.** The harness is the customer's, lives upstream,
and is configured entirely through environment variables — see
[Pointing it at this deployment](#pointing-it-at-this-deployment) at the bottom.

## What the benchmark is

A **deterministic, spec-constructed replay trace** plus a generic driver:

- `gen_caseA_conformance.py` synthesises 200 sessions / 1,778 requests from the
  Case-A parameters at a fixed seed, so the corpus is byte-identical every run.
- The corpus records **demand only** — per-request input/output token counts, turn
  structure, think-times, KV prefix reuse. Nothing engine- or topology-specific.
- `replay_caseA.sh` replays it via **aiperf** (a SemiAnalysis fork, scenario
  `inferencex-agentx-mvp`, `--custom-dataset-type weka_trace`) against any
  OpenAI-compatible endpoint.

It is **closed-loop**, like the infera bench: `--concurrency N` runs N workers, and a
worker does not issue its next request until the previous one has returned. Offered
load is therefore bounded by the server, not pushed at it — which the measurement
confirms, in-flight sitting at mean 5.13 against a cap of 8. What differs from the
infera bench is not the loop but *what N counts*: N is a fixed worker population here,
where the infera bench starts 8 sessions and lets the population grow.

The frozen trace is what makes this benchmark useful: replaying it against a
different deployment changes exactly one variable.

## Results at c8

Both clusters, the same frozen trace, the same deployment shape:

| | multi-rail cluster | single-rail cluster |
|---|---|---|
| profiling requests | 231 | 231 |
| window | 901 s | 907 s |
| **errors / cancelled / context overflows** | **0 / 0 / 0** | **2 / 0 / 0** |
| in-flight max / mean | 8 / 5.13 | 8 / 4.98 |
| request rate | 0.256 req/s | 0.255 req/s |
| output token rate | 268 tok/s | 265 tok/s |
| **TTFT p50** | **5,146 ms** | 6,698 ms |
| TTFT p90 | 19,780 ms | 23,871 ms |
| TTFT p99 | 31,014 ms | 33,972 ms |
| **E2E p50** | **12,556 ms** | 13,874 ms |
| E2E p90 | 44,522 ms | 42,501 ms |
| **ITL p50** | **13.81 ms** | 13.26 ms |
| ITL p90 | 27.43 ms | 18.52 ms |
| ISL p50 / mean | 70,624 / 80,989 | 69,911 / 80,811 |
| OSL p50 / mean | 223 / — | 230 / 1,050 |
| **server-reported cache hit** (per-request p50) | **88.1 %** | **88.1 %** |

Ladders are recomputed from aiperf's raw per-request records
(`profile_export.jsonl`), not copied from a summary line.

**The two clusters agree closely at this concurrency** — TTFT p50 within 1.3×, ITL
within 4 %. That is a narrower gap than the infera bench shows between the same two
clusters, which is consistent with this driver's fixed worker population holding
in-flight at ~5 on both, where the infera bench's growing session population let the
faster cluster run at higher occupancy.

### The harness's cache-hit construction does not meet the workload spec

The Case-A spec asks for a **~88–89 % prefix-cache hit rate**. The corpus generator
aims at that number per turn and reaches it *as a per-turn median* — but the corpus it
produces does not deliver it as a workload property. Measured over all 200 sessions /
1,778 requests of the shipped corpus:

| how the corpus's own `hash_ids` are counted | hit rate |
|---|---|
| per-turn ratio, median | 88.0 % |
| per-turn ratio, median, excluding turn 0 | 88.2 % |
| per-turn ratio, **mean** | 66.0 % |
| **token-weighted over the whole corpus** | **59.7 %** |

Three things in `gen_caseA_conformance.py` drive the gap, and each is worth reporting
upstream:

1. **Cold start is inside the target.** Turn 0 is all-miss by construction and is
   11.2 % of requests. The generator applies its 0.88–0.90 draw only from turn 1, so
   the corpus can hit the target per-turn and still miss it overall.
2. **No token weighting.** The target is drawn per *turn*, uniformly, but turns differ
   by two orders of magnitude in input size (ISL is sampled lognormal per turn,
   independently of the reuse draw). Cheap turns and 240K-token turns count equally
   toward a rate whose cost is entirely token-proportional.
3. **The reuse accounting does not survive a shrinking turn.** Reuse is
   `min(len(prefix_blocks), int(total_blocks × uniform(0.88, 0.90)))` against a
   `total_blocks` resampled independently each turn, and the surviving prefix is then
   *overwritten* with this turn's `hash_ids` (`prefix_blocks = hash_ids`). Because
   consecutive ISL draws are independent, **51 % of turns are shorter than their
   predecessor**, and every one of those truncates the accumulated prefix permanently.
   A real agentic session's context grows; this one random-walks.

The consequence for reading the results below: the harness's own summary line reports
**`Theoretical Prefix Cache Hit` ≈ 50.8 %**, and that figure is *not* a deployment
measurement — it is computed from the trace file's `hash_ids` and never asks the
server, so it reads the same on both clusters regardless of how the cache performs.
Its closeness to the corpus's own 59.7 % token-weighted rate is the point: both are
properties of the corpus.

The server-side number aiperf does record is `usage_prompt_cache_read_tokens`. Taken
as a per-request ratio it is **88.1 % p50 / 89.4 % p90 on both clusters** — but that
median is over the 177 of 231 records that carry the field, and the 54 that do not are
exactly the `turn_index == 0` requests, i.e. the cold ones. Counted with those in at
0 % it is 72.4 %; token-weighted over all requests, 50.3 %. **All three are real
numbers about different populations**, and the deployment comparison in this document
uses the first because it is the one that holds the population fixed across clusters.

A second, unrelated defect is worth reporting with the above: `replay_caseA.sh`
silently loses results when `OUT` is set outside the script's own directory, because
the container mounts only that directory — the sweep then prints `FAILED` for a run
that actually succeeded.

## What this benchmark establishes that the infera one cannot

**1. Per-turn attribution — the prefix cache is worth 2.5× on TTFT.** aiperf tags
every record with `turn_index`, so first-turn (cold) and later-turn (cached) requests
can be split at matched input size:

| | n | ISL p50 | TTFT p50 |
|---|---|---|---|
| first turn (cold) | 54 | 69,907 | **8,981 ms** |
| turn ≥ 1 (cached) | 177 | 70,998 | **3,568 ms** |

2.5× faster for a 1.6 % *larger* prompt. That is the prefix cache priced directly.
The infera bench has no turn index and cannot produce this.

**2. Independent confirmation of the cache rate.** 88.1 % measured from the server's
own usage field, by a third-party driver, on a third-party trace, on **both**
clusters — against the infera bench's 88.9 %. Same population caveat as above: this is
the median over the warm requests.

**3. A structurally clean input distribution.** `max(in + out) = 258,303` by
construction, below the 262,144 context, so context overflow is impossible. The
infera bench samples input and output independently with no joint clamp, which is
where its 0.5 % error rate comes from.

**4. TTFT-by-input-size, and what changes with concurrency.** Comparing c8 against a
c16 run on the same deployment, the curve changes *shape*:

| | 0–50K | 50–100K | 100–160K | 160–220K | 220–300K | spread |
|---|---|---|---|---|---|---|
| **c8** TTFT p50 | 3,177 | 6,161 | 11,305 | 20,674 | 22,639 | **7.1×** |
| **c16** TTFT p50 | 14,425 | 18,594 | 22,723 | 33,162 | 40,192 | **2.8×** |

*(single-rail cluster; the multi-rail cluster shows the same transition, 10.0× → 2.3×)*

At c8 the curve is **prefill-shaped** — monotone and super-linear, the deployment is
computing. At c16 it **flattens** and the smallest bucket already costs 14 s: a 40K
request cannot take that long to prefill on a leg that serves 240K in 40 s, so that
time is queueing rather than compute.

TTFT here is entirely server-side: `http_req_sending` p50 is 0.2 ms.

## Why its TTFT is higher than the infera bench's

At c8 the customer bench reads TTFT p50 5,146 ms against the infera bench's 1,365 ms
on the same cluster, **against the same server processes in the same hour**.

Part of it is the load model — but not all of it, and the residual is honestly
unexplained. At c8 the customer bench's mean in-flight is **5.1, less than half** the
infera bench's 12.1, and its TTFT is still 3.8× worse. Lower offered load with worse
latency is not explained by queueing.

Candidates, none of which the available data discriminates between:

| candidate | what would settle it |
|---|---|
| the 900 s window is too short for the prefill radix tree to reach the steady state the 3,600 s run measures | run the customer bench at c8 for 3,600 s; compare its first 900 s against its last |
| the scenario injects a unique marker into every trajectory's first turn, deliberately defeating cross-trajectory prefix sharing the other driver allows | run with the scenario's cache-bust disabled |
| the driver may not honour the trace's `think_time`, so turns arrive far denser than the session profile specifies | compare measured inter-arrival per conversation against the trace's think-time field |
| measurement definition — first *token* vs first *chunk* | inspect the first streamed chunk of a known request under both drivers |

**Do not pick one of these without running the experiment.**

## The right posture: run both

The two benchmarks answer different questions and should not be collapsed into one
number.

- The **infera bench** discovers capacity: its session population grows, so it finds
  the concurrency the workload naturally sustains, runs a long steady-state window,
  and captures router and KV internals. The customer bench must be *told* a worker
  count, so it measures the point you chose.
- The **customer bench** compares deployments: the frozen trace means replaying it
  against a different topology changes exactly one variable. It also models real
  per-turn structure and applies no client timeout.

They agree where they should — ITL ≈ 14 ms, cache ≈ 88 % — and diverge exactly where
the load model differs.

## Pointing it at this deployment

Get the harness from upstream ([ROCm/MAD PR #173](https://github.com/ROCm/MAD/pull/173),
`scripts/AgentX_CaseA/`) and configure it by environment only — no code change is
needed, and none should be made.

| var | set it to |
|---|---|
| `URL` | this deployment's router: `http://<prefill-data-plane-ip>:8100` |
| `SERVED` | the served-model-name this kit launches with (`glm5.2-mxfp4`), **not** the harness default |
| `TOK` | a tokenizer path the harness's container can see |
| `CONCS` | `8` for the numbers above |
| `DUR` | `900` — the scenario enforces this as its minimum and rejects the script's own default |
| `OUT` | a path **inside the harness's own directory** — see the defect noted above |

Two prerequisites the deployment side must satisfy, both of which this kit already
does: `--enable-cache-report` on the engine (or every cache-hit column reads 0), and a
context length that covers the trace's 258,303-token maximum.
