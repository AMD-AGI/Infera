# Analysis — customer AgentX Case-A replay against the infera GLM-5.2 PD deployment

Format follows `../../par8.glm52.dpaoff.packup_20260803/analysis/`.

| file | what |
|---|---|
| this file | headline, the comparison against our own bench, the verdict |
| [`sli_percentiles.md`](sli_percentiles.md) | full ladders recomputed from raw records; TTFT-by-input-size; first-turn vs cached-turn split |
| [`vs_infera_bench.md`](vs_infera_bench.md) | the two benches side by side — what each can and cannot establish |
| [`customer_method_review.md`](customer_method_review.md) | review of the customer's method + deployment recipe; what is worth adopting |

---

## Headline

**The customer's benchmark runs clean against our deployment, and it confirms our
own prefix-cache number independently — but the two benches measure different
things, and the headline latency figures are not comparable.**

| | AgentX c8 | AgentX c16 | par8 (our bench) |
|---|---|---|---|
| driver | aiperf, **open-loop trace replay** | same | `agent_throughput.py`, **closed-loop** |
| requests (measured) | **231** | **323** | 2,671 |
| window | 901 s | 922 s | 3,600 s |
| errors / cancelled | **0 / 0** | **0 / 0** | 15 (0.52 %) |
| in-flight max / mean | **8 / 5.1** | **16 / 11.1** | 22 / 12.1 |
| **TTFT p50** | **5,146 ms** | **14,394 ms** | **1,365 ms** |
| TTFT p90 | 19,780 ms | 30,567 ms | 4,903 ms |
| ITL / TPOT p50 | **13.8 ms** | **14.7 ms** | **14.8 ms** |
| ITL / TPOT p90 | 27.4 ms | 29.7 ms | 17.7 ms |
| E2E p50 | 12.6 s | 21.6 s | 7.4 s |
| per-request cache hit p50 | **88.1 %** | **88.1 %** | 88.9 % |
| ISL p50 / mean | 70,624 / 80,989 | 73,450 / 82,330 | 75,440 / 86,888 |
| OSL p50 / mean | 223 / 1,046 | 264 / 1,097 | 316 / 1,087 |

*(AgentX = profiling phase only, warmup excluded. par8 = sustain phase only.)*

## ⚠ TTFT is 3.8× worse under the customer bench — and that is not a regression

**Read this before quoting the TTFT number.** At c16 the deployment shows TTFT
p50 14,394 ms against par8's 1,365 ms on **the same hardware, the same engine
processes, the same hour**. Nothing about the server changed between the two
measurements. The difference is the load model.

**The mechanism is measured, not inferred:**

| | evidence |
|---|---|
| both AgentX points **pinned** at their concurrency ceiling | in-flight max = 8 of 8, and 16 of 16 |
| the wait is **entirely server-side** | `http_req_waiting` p50 = TTFT p50 to the millisecond; `http_req_sending` p50 = **0.2 ms** |
| at c16 TTFT decouples from input size | 0–50K bucket p50 = **11,949 ms**, 220–300K bucket p50 = 27,860 ms — only 2.3× across a 5× size span (at c8 it is 1,867 → 18,661 ms, exactly 10×) |

A flat TTFT-vs-size curve is the signature of **queueing**, not prefill. At c16
a small request waits about as long as a large one because it is waiting behind
other requests, not computing.

**Why the two benches load the server differently, at the same nominal
concurrency:**

- **par8 is closed-loop.** A session issues its next turn only after the
  previous one returns, then sleeps its think-time. Mean in-flight 12.1 is an
  *emergent* value — the workload and the server negotiate it.
- **AgentX is open-loop with a hard lane count.** aiperf keeps exactly
  `--concurrency` trajectory lanes saturated; the moment a request returns the
  lane issues the next one. In-flight is *pinned*, and the trace's think-times
  are consumed inside a lane rather than reducing offered load.

So "concurrency 16" in the customer's bench is **strictly more offered load**
than "mean 12.1 in-flight" in ours, even though the numbers look adjacent.

**The honest comparison is c8 vs par8**: in-flight mean 5.1 vs 12.1 — the
customer bench at c8 is offering *less* load, yet still shows TTFT p50 5,146 ms
against par8's 1,365 ms. **That residual gap is not explained by this data.**
The candidate explanations are listed in
[`vs_infera_bench.md`](vs_infera_bench.md); discriminating between them needs
one more run, not more analysis.

## What this run does establish, cleanly

**1. The customer's bench runs against our stack with zero code modification.**
Only environment variables (`URL`, `SERVED`, `TOK`, `CONCS`, `DUR`, `IMG`).
`md5sum replay_caseA.sh` is unchanged in the run log.

**2. Zero errors, zero cancellations, zero context overflows** across 554
measured requests. Notably better than our own workload, which produces 0.52 %
context-overflow rejections because it samples input and output independently
with no joint clamp. The customer's frozen trace has
`max(in + out) = 258,303 < 262,144` **by construction** — 0 requests can overflow.

**3. Our prefix cache is independently confirmed at 88.1 %.** This is the most
valuable single number in the kit: it comes from the *server's own*
`usage_prompt_cache_read_tokens`, reported through a **third-party driver on a
third-party trace**, and it lands on our own bench's 88.9 %. Two independent
measurement paths agreeing on cache behaviour.

**4. Decode is not the bottleneck, and this reproduces our finding.** ITL p50
13.8 ms (c8) / 14.7 ms (c16) against par8's TPOT p50 14.8 ms — the same number
from a different driver. ITL barely moves when concurrency doubles (13.8 →
14.7 ms, +6 %) while TTFT triples. Prefill is the binding resource, exactly as
par8 concluded.

**5. The cache is doing real work, and we can now price it.** At c8, splitting by
turn index at near-identical input size:

| | n | ISL p50 | TTFT p50 |
|---|---|---|---|
| first turn (cold) | 54 | 69,907 | **8,981 ms** |
| turn ≥ 1 (cached) | 177 | 70,998 | **3,568 ms** |

**2.5× on TTFT for the same prompt size.** par8 could not produce this table —
its driver does not tag turn index in the metrics stream.

## The measurement finding: the customer's README misstates its own cache metric

The Case-A README says:

> Reported cache-hit is the endpoint's realized server-side prefix hit

**The code does not do this.** `aiperf/metrics/theoretical_prefix_cache.py:22-30`
computes the reported `Theoretical Prefix Cache Hit` from the **loader's own walk
of the trace's `hash_ids`** — an infinite-cache upper bound derived from the
input file. It never asks the server.

This matters because the two numbers can disagree arbitrarily: the trace-side
value is a property of the corpus (always ~88 %), while the server-side value
depends on the deployment. In this run they happen to agree, which is a real
result — but only because we computed the server-side number ourselves from
`usage_prompt_cache_read_tokens` in the raw records.

**Recommend reporting this upstream.** It is a documentation/naming defect, not
a correctness bug: the metric is honestly *named* "Theoretical".

## Verdict against par8's stated bars

Using the same bars par8 was judged against:

| bar | target | c8 | c16 | verdict |
|---|---|---|---|---|
| success rate | ≥ 0.97 | **1.000** | **1.000** | **PASS** |
| TTFT p90 | < 30,000 ms | **19,780 ms** | **30,567 ms** | **PASS / marginal FAIL** |
| E2E p50 | < 4,500 ms | 12,556 ms | 21,620 ms | **FAIL** |

The E2E miss is the same one par8 recorded: `e2e_p50_ms: 4500` is a
**latency-floor** spec that the solo kits showed is met only at concurrency 1.
It fails identically in every loaded run on this stack.

TTFT p90 at c16 clears the bar by 567 ms — **too close to call a pass with one
sample**. If the customer intends to certify against a 30 s TTFT p90, c16 is the
operating limit of this deployment posture, not a comfortable margin.

## What to run next

| question | run |
|---|---|
| **Why is c8 TTFT 3.8× par8's at lower in-flight?** | AgentX at c4 and c2 — extend the ladder down until it meets par8's regime | 
| Does prefill DPA-on help under the customer's load? | same replay, prefill relaunched with DPA on + chunk 65536 — also par8's missing control |
| Is c16 the real TTFT-p90 cliff? | AgentX at c12 and c24 — bracket the 30 s bar |
| Does the trace's think-time reach the server? | compare inter-arrival in the records against the trace's `think_time` field |
